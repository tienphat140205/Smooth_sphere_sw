import torch

import numpy as np
import torch.nn.functional as F
from utils_vmf import rand_t_marginal



def roll_by_gather(mat,dim, shifts: torch.LongTensor):
    ## https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch

    # assumes 2D array
    n_rows, n_cols = mat.shape
    
    if dim==0:
        arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols, device=mat.device).view(( 1,n_cols)).repeat((n_rows,1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)
    

def dCost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()
    
    n = u_values.shape[-1]
    m_batch, m = v_values.shape
    
    v_cdf_theta = v_cdf -(theta - torch.floor(theta))
    
    mask_p = v_cdf_theta>=0
    mask_n = v_cdf_theta<0
         
    v_values[mask_n] += torch.floor(theta)[mask_n]+1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    ## ??
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1
    
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))

    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1,1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1,1))
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)
    
    u_index = torch.searchsorted(u_cdf, v_cdf_theta)
    u_icdf_theta = torch.gather(u_values, -1, u_index.clip(0, n-1))
    
    ## Deal with 1
    u_cdfm = torch.cat([u_cdf, u_cdf[:,0].view(-1,1)+1], dim=1)
    u_valuesm = torch.cat([u_values, u_values[:,0].view(-1,1)+1],dim=1)
    u_indexm = torch.searchsorted(u_cdfm, v_cdf_theta, right=True)
    u_icdfm_theta = torch.gather(u_valuesm, -1, u_indexm.clip(0, n))
    
    dCp = torch.sum(torch.pow(torch.abs(u_icdf_theta-v_values[:,1:]), p)
                   -torch.pow(torch.abs(u_icdf_theta-v_values[:,:-1]), p), axis=-1)
    
    dCm = torch.sum(torch.pow(torch.abs(u_icdfm_theta-v_values[:,1:]), p)
                   -torch.pow(torch.abs(u_icdfm_theta-v_values[:,:-1]), p), axis=-1)
    
    return dCp.reshape(-1,1), dCm.reshape(-1,1)


def Cost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()
    
    m_batch, m = v_values.shape
    n_batch, n = u_values.shape

    v_cdf_theta = v_cdf -(theta - torch.floor(theta))
    
    mask_p = v_cdf_theta>=0
    mask_n = v_cdf_theta<0
    
    v_values[mask_n] += torch.floor(theta)[mask_n]+1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    
    ## ??
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1
    
    ## Put negative values at the end
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))# .tolist()

    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1,1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1,1))
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)  
    
    ## Compute abscisse
    cdf_axis, cdf_axis_sorter = torch.sort(torch.cat((u_cdf, v_cdf_theta), -1), -1)
    cdf_axis_pad = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis_pad[..., 1:] - cdf_axis_pad[..., :-1]

    ## Compute icdf
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
        
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)
    v_index = torch.searchsorted(v_cdf_theta, cdf_axis)
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m))
    
    if p == 1:
        ot_cost = torch.sum(delta*torch.abs(u_icdf-v_icdf), axis=-1)
    elif p == 2:
        ot_cost = torch.sum(delta*torch.square(u_icdf-v_icdf), axis=-1)
    else:
        ot_cost = torch.sum(delta*torch.pow(torch.abs(u_icdf-v_icdf), p), axis=-1)
    return ot_cost



def binary_search_circle(u_values, v_values, u_weights=None, v_weights=None, p=1, 
                         Lm=10, Lp=10, tm=-1, tp=1, eps=1e-6, require_sort=True):
    r"""
    Computes the Wasserstein distance on the circle using the Binary search algorithm proposed in [1].

    Parameters:
    u_values : ndarray, shape (n_batch, n_samples_u)
        samples in the source domain
    v_values : ndarray, shape (n_batch, n_samples_v)
        samples in the target domain
    u_weights : ndarray, shape (n_batch, n_samples_u), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n_batch, n_samples_v), optional
        samples weights in the target domain
    p : float, optional
        Power p used for computing the Wasserstein distance
    Lm : int, optional
        Lower bound dC
    Lp : int, optional
        Upper bound dC
    tm: float, optional
        Lower bound theta
    tp: float, optional
        Upper bound theta
    eps: float, optional
        Stopping condition
    require_sort: bool, optional
        If True, sort the values.

    [1] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    """
    ## Matlab Code : https://users.mccme.ru/ansobol/otarie/software.html
    
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]
    
    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)
    
    L = max(Lm,Lp)
    
    tm = tm * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1,1)
    tm = tm.repeat(1, m)
    tp = tp * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1,1)
    tp = tp.repeat(1, m)
    tc = (tm+tp)/2
    
    done = torch.zeros((u_values.shape[0],m))
        
    cpt = 0
    while torch.any(1-done):
        cpt += 1
        
        dCp, dCm = dCost(tc, u_values, v_values, u_cdf, v_cdf, p)
        done = ((dCp*dCm)<=0) * 1
        
        mask = ((tp-tm)<eps/L) * (1-done)
        
        if torch.any(mask):
            ## can probably be improved by computing only relevant values
            dCptp, dCmtp = dCost(tp, u_values, v_values, u_cdf, v_cdf, p)
            dCptm, dCmtm = dCost(tm, u_values, v_values, u_cdf, v_cdf, p)
            Ctm = Cost(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
            Ctp = Cost(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
        
            mask_end = mask * (torch.abs(dCptm-dCmtp)>0.001)
            tc[mask_end>0] = ((Ctp-Ctm+tm*dCptm-tp*dCmtp)/(dCptm-dCmtp))[mask_end>0]
            done[torch.prod(mask, dim=-1)>0] = 1
        ## if or elif?
        elif torch.any(1-done):
            tm[((1-mask)*(dCp<0))>0] = tc[((1-mask)*(dCp<0))>0]
            tp[((1-mask)*(dCp>=0))>0] = tc[((1-mask)*(dCp>=0))>0]
            tc[((1-mask)*(1-done))>0] = (tm[((1-mask)*(1-done))>0]+tp[((1-mask)*(1-done))>0])/2
    
    return Cost(tc.detach(), u_values, v_values, u_cdf, v_cdf, p)


def emd1D_circle(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    if p == 1:
        ## Code inspired from https://gitlab.gwdg.de/shundri/circularOT/-/tree/master/
        values_sorted, values_sorter = torch.sort(torch.cat((u_values, v_values), -1), -1)
        
        cdf_diff = torch.cumsum(torch.gather(torch.cat((u_weights, -v_weights),-1),-1,values_sorter),-1)
        cdf_diff_sorted, cdf_diff_sorter = torch.sort(cdf_diff, axis=-1)
        
        values_sorted = torch.nn.functional.pad(values_sorted, (0,1), value=1)
        delta = values_sorted[..., 1:]-values_sorted[..., :-1]
        weight_sorted = torch.gather(delta, -1, cdf_diff_sorter)

        sum_weights = torch.cumsum(weight_sorted, axis=-1)-0.5
        sum_weights[sum_weights<0] = np.inf
        inds = torch.argmin(sum_weights, axis=-1)
            
        levMed = torch.gather(cdf_diff_sorted, -1, inds.view(-1,1))
        
        return torch.sum(delta * torch.abs(cdf_diff - levMed), axis=-1)



def sliced_cost(Xs, Xt, Us, p=2, u_weights=None, v_weights=None):
    """
        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        Xt: ndarray, shape (n_samples_v, dim)
            Samples in the target domain
        Us: ndarray, shape (num_projections, d, 2)
            Independent samples of the Uniform distribution on V_{d,2}
        p: float
            Power
    """
    n_projs, d, k = Us.shape
    n, _ = Xs.shape
    m, _ = Xt.shape    

    
    ## Projection on S^1
    ## Projection on plane
    Xps = torch.matmul(torch.transpose(Us,1,2)[:,None], Xs[:,:,None]).reshape(n_projs, n, 2)
    Xpt = torch.matmul(torch.transpose(Us,1,2)[:,None], Xt[:,:,None]).reshape(n_projs, m, 2)
        
    ## Projection on sphere
    Xps = F.normalize(Xps, p=2, dim=-1)
    Xpt = F.normalize(Xpt, p=2, dim=-1)
    
    ## Get coords
    Xps = (torch.atan2(-Xps[:,:,1], -Xps[:,:,0])+np.pi)/(2*np.pi)
    Xpt = (torch.atan2(-Xpt[:,:,1], -Xpt[:,:,0])+np.pi)/(2*np.pi)
        
    if p==1:
        w1 = emd1D_circle(Xps, Xpt, u_weights=u_weights, v_weights=v_weights)
    else:
        w1 = binary_search_circle(Xps, Xpt, p=p, u_weights=u_weights, v_weights=v_weights)

    return torch.mean(w1)
    

def sliced_wasserstein_sphere(Xs, Xt, num_projections, device, u_weights=None, v_weights=None, p=2):
    """
        Compute the sliced-Wasserstein distance on the sphere.

        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        Xt: ndarray, shape (n_samples_v, dim)
            Samples in the target domain
        num_projections: int
            Number of projections
        device: str
        p: float
            Power of SW. Need to be >= 1.
    """
    d = Xs.shape[1]
    
    ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
    Z = torch.randn((num_projections,d,2), device=device)
    U, _ = torch.linalg.qr(Z)
            
    return sliced_cost(Xs, Xt, U, p=p, u_weights=u_weights, v_weights=v_weights)


def w2_unif_circle_approx(u_values):
    """
    	Approximation 
        weights 1/n
        Compute u_values vs Uniform distribution
        
        Parameters:
        u_values: ndarray, shape (n_batch, n_samples)
    """
    
    n = u_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    u_values, _ = torch.sort(u_values, -1)
    u_weights = torch.full((n,), 1/n, dtype=u_values.dtype, device=u_values.device)
    u_cdf = torch.cumsum(u_weights, -1)
    
    alpha = torch.mean(u_values, axis=-1)-1/2

    ot_cost = torch.mean(torch.square(u_values-u_cdf-alpha[:,None]), axis=-1)
    return ot_cost
    
    
def w2_unif_circle(u_values):
    """
    	Closed-form
    	
        weights 1/n
        Compute u_values vs Uniform distribution
        
        Parameters:
        u_values: ndarray, shape (n_batch, n_samples)
    """
    
    n = u_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    u_values, _ = torch.sort(u_values, -1)
    u_weights = torch.full((n,), 1/n, dtype=u_values.dtype, device=u_values.device)
    u_cdf = torch.cumsum(u_weights, -1)
    
    cpt1 = torch.mean(u_values**2, axis=-1)
    x_mean = torch.mean(u_values, axis=-1)
    
#    ns = torch.tensor(range(1, n+1), dtype=torch.float)
#    cpt2 = torch.sum((n+1-2*ns)*u_values, axis=-1)/n**2
    
    ns_n2 = torch.arange(n-1, -n, -2, dtype=torch.float, device=u_values.device)/n**2
    cpt2 = torch.sum(ns_n2 * u_values, dim=-1)
    
    return cpt1 - x_mean**2 +cpt2 + 1/12


def sliced_wasserstein_sphere_unif(Xs, num_projections, device):
    """
        Compute the SSW2 on the sphere w.r.t. a uniform distribution.

        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        num_projections: int
            Number of projections
        device: str.
    """
    n, d = Xs.shape
    
    ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
    Z = torch.randn((num_projections,d,2), device=device)
    U, _ = torch.linalg.qr(Z)
    
    ## Projection on S^1
    ## Projection on plane
    Xps = torch.matmul(torch.transpose(U,1,2)[:,None], Xs[:,:,None]).reshape(num_projections, n, 2)
    ## Projection on sphere
    Xps = F.normalize(Xps, p=2, dim=-1)
    ## Get coords
    Xps = (torch.atan2(-Xps[:,:,1], -Xps[:,:,0])+np.pi)/(2*np.pi)
        
    return torch.mean(w2_unif_circle(Xps))


# ---------------------------------------------------------------------------
# Global Smoothed Spherical Sliced-Wasserstein (GSSW)
# Definition: GSSW_{p,kappa}(mu, nu) = SSW_p(mu * K_kappa, nu * K_kappa)
# Empirical approximation: sample x ~ mu, then x' ~ K(x, kappa)
# ---------------------------------------------------------------------------

def sample_vmf_batch(x, kappa):
    """
    Sample x' ~ vMF(x_i, kappa) for each x_i in a batch, using Householder
    rotation. The rotation step is differentiable w.r.t. x.

    Parameters:
    x     : Tensor, shape (n, d) — batch of unit vectors on S^{d-1}
    kappa : float — concentration parameter (kappa=0 -> uniform on S^{d-1})

    Returns:
    Tensor, shape (n, d) — smoothed samples on S^{d-1}
    """
    n, d = x.shape
    device = x.device
    dtype = x.dtype

    # Sample t from the vMF marginal distribution (Wood 1994).
    # rand_t_marginal is numpy-based; no gradient needed through t itself.
    t_np = rand_t_marginal(float(kappa), d, n)          # (n, 1) numpy
    t = torch.tensor(t_np, dtype=dtype, device=device).squeeze(1)  # (n,)

    # Sample a uniform direction on S^{d-2} (orthogonal complement of e_1)
    v = F.normalize(torch.randn(n, d - 1, dtype=dtype, device=device), p=2, dim=-1)

    # Build z in the canonical frame aligned with e_1 = [1, 0, ..., 0]
    # z = [t, sqrt(1 - t^2) * v]
    rad = torch.sqrt(torch.clamp(1.0 - t ** 2, min=1e-7)).unsqueeze(1)  # (n, 1)
    z = torch.cat([t.unsqueeze(1), rad * v], dim=-1)   # (n, d)

    # Householder reflection: rotate z from e_1 frame to x frame.
    # H(u) z = z - 2 (z . u) u,  where u = normalize(e_1 - x)
    e1 = torch.zeros_like(x)
    e1[:, 0] = 1.0
    diff = e1 - x                                              # (n, d)
    diff_norm = torch.linalg.norm(diff, dim=-1, keepdim=True)  # (n, 1)
    # When x ≈ e1, diff_norm ≈ 0 → no rotation needed (z already aligned)
    u = torch.where(diff_norm > 1e-7, diff / diff_norm, torch.zeros_like(diff))

    x_smooth = z - 2.0 * (z * u).sum(dim=-1, keepdim=True) * u  # (n, d)
    return x_smooth


def sample_ps_batch(x, kappa):
    """
    Sample x' ~ PowerSpherical(x_i, kappa) for each x_i in a batch.
    Uses PowerSpherical.rsample() which supports pathwise (reparameterised)
    gradients through x.

    Parameters:
    x     : Tensor, shape (n, d) — batch of unit vectors on S^{d-1}
    kappa : float — concentration parameter

    Returns:
    Tensor, shape (n, d) — smoothed samples on S^{d-1}
    """
    from power_spherical import PowerSpherical
    scale = torch.full((x.shape[0],), float(kappa), dtype=x.dtype, device=x.device)
    ps = PowerSpherical(x, scale)
    return ps.rsample()


def gssw_sphere(Xs, Xt, num_projections, device, kappa, p=2, kernel='vmf'):
    """
    Compute the Global Smoothed Spherical Sliced-Wasserstein distance between
    two empirical distributions on S^{d-1}.

    GSSW_{p,kappa}(mu, nu) = SSW_p(mu * K_kappa, nu * K_kappa)

    Parameters:
    Xs            : Tensor, shape (n, d) — samples from source distribution mu
    Xt            : Tensor, shape (m, d) — samples from target distribution nu
    num_projections: int   — number of random projections
    device        : str
    kappa         : float  — smoothing concentration (kappa -> inf recovers SSW)
    p             : float  — power of the Wasserstein distance (>= 1)
    kernel        : str    — 'vmf' (von Mises-Fisher) or 'ps' (Power Spherical)

    Returns:
    Scalar Tensor — GSSW_{p,kappa}(mu, nu)
    """
    sample_fn = sample_vmf_batch if kernel == 'vmf' else sample_ps_batch
    Xs_smooth = sample_fn(Xs, kappa)
    Xt_smooth = sample_fn(Xt, kappa)
    return sliced_wasserstein_sphere(Xs_smooth, Xt_smooth, num_projections, device, p=p)


def gssw_sphere_unif(Xs, num_projections, device, kappa, kernel='vmf'):
    """
    Compute the Global Smoothed SSW between an empirical distribution on
    S^{d-1} and the uniform distribution.

    Because Uniform * K_kappa = Uniform for any rotation-invariant kernel K,
    only Xs is smoothed.

    Parameters:
    Xs            : Tensor, shape (n, d) — samples from source distribution mu
    num_projections: int   — number of random projections
    device        : str
    kappa         : float  — smoothing concentration
    kernel        : str    — 'vmf' or 'ps' (Power Spherical)

    Returns:
    Scalar Tensor — GSSW_{2,kappa}(mu, Uniform(S^{d-1}))
    """
    sample_fn = sample_vmf_batch if kernel == 'vmf' else sample_ps_batch
    Xs_smooth = sample_fn(Xs, kappa)
    return sliced_wasserstein_sphere_unif(Xs_smooth, num_projections, device)

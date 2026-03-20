import torch
import torch.nn.functional as F
import numpy as np

try:
    from ..utils.power_spherical import PowerSpherical
    from ..utils.vmf import sample_vmf_batch
except ImportError:
    from utils.power_spherical import PowerSpherical
    from utils.vmf import sample_vmf_batch


# ---------------------------------------------------------------------------
# Helpers: circle Wasserstein (unchanged from original)
# ---------------------------------------------------------------------------

def roll_by_gather(mat, dim, shifts: torch.LongTensor):
    n_rows, n_cols = mat.shape
    if dim == 0:
        arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim == 1:
        arange1 = torch.arange(n_cols, device=mat.device).view((1, n_cols)).repeat((n_rows, 1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)


def dCost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()
    n = u_values.shape[-1]
    m_batch, m = v_values.shape
    v_cdf_theta = v_cdf - (theta - torch.floor(theta))
    mask_p = v_cdf_theta >= 0
    mask_n = v_cdf_theta < 0
    v_values[mask_n] += torch.floor(theta)[mask_n] + 1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))
    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1, 1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1, 1))
    v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)
    u_index = torch.searchsorted(u_cdf, v_cdf_theta)
    u_icdf_theta = torch.gather(u_values, -1, u_index.clip(0, n - 1))
    u_cdfm = torch.cat([u_cdf, u_cdf[:, 0].view(-1, 1) + 1], dim=1)
    u_valuesm = torch.cat([u_values, u_values[:, 0].view(-1, 1) + 1], dim=1)
    u_indexm = torch.searchsorted(u_cdfm, v_cdf_theta, right=True)
    u_icdfm_theta = torch.gather(u_valuesm, -1, u_indexm.clip(0, n))
    dCp = torch.sum(torch.pow(torch.abs(u_icdf_theta - v_values[:, 1:]), p)
                    - torch.pow(torch.abs(u_icdf_theta - v_values[:, :-1]), p), axis=-1)
    dCm = torch.sum(torch.pow(torch.abs(u_icdfm_theta - v_values[:, 1:]), p)
                    - torch.pow(torch.abs(u_icdfm_theta - v_values[:, :-1]), p), axis=-1)
    return dCp.reshape(-1, 1), dCm.reshape(-1, 1)


def Cost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()
    m_batch, m = v_values.shape
    n_batch, n = u_values.shape
    v_cdf_theta = v_cdf - (theta - torch.floor(theta))
    mask_p = v_cdf_theta >= 0
    mask_n = v_cdf_theta < 0
    v_values[mask_n] += torch.floor(theta)[mask_n] + 1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))
    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1, 1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1, 1))
    v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)
    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf_theta), -1), -1)
    cdf_axis_pad = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis_pad[..., 1:] - cdf_axis_pad[..., :-1]
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n - 1))
    v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)
    v_index = torch.searchsorted(v_cdf_theta, cdf_axis)
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m))
    if p == 1:
        return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
    elif p == 2:
        return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)
    else:
        return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)


def binary_search_circle(u_values, v_values, u_weights=None, v_weights=None, p=1,
                         Lm=10, Lp=10, tm=-1, tp=1, eps=1e-6, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]
    device = u_values.device
    dtype = u_values.dtype
    if u_weights is None:
        u_weights = torch.full((n,), 1 / n, dtype=dtype, device=device)
    if v_weights is None:
        v_weights = torch.full((m,), 1 / m, dtype=dtype, device=device)
    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)
        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]
    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)
    L = max(Lm, Lp)
    tm = (tm * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1, 1)).repeat(1, m)
    tp = (tp * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1, 1)).repeat(1, m)
    tc = (tm + tp) / 2
    done = torch.zeros((u_values.shape[0], m))
    while torch.any(1 - done):
        dCp, dCm = dCost(tc, u_values, v_values, u_cdf, v_cdf, p)
        done = ((dCp * dCm) <= 0) * 1
        mask = ((tp - tm) < eps / L) * (1 - done)
        if torch.any(mask):
            dCptp, dCmtp = dCost(tp, u_values, v_values, u_cdf, v_cdf, p)
            dCptm, dCmtm = dCost(tm, u_values, v_values, u_cdf, v_cdf, p)
            Ctm = Cost(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
            Ctp = Cost(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
            mask_end = mask * (torch.abs(dCptm - dCmtp) > 0.001)
            tc[mask_end > 0] = ((Ctp - Ctm + tm * dCptm - tp * dCmtp) / (dCptm - dCmtp))[mask_end > 0]
            done[torch.prod(mask, dim=-1) > 0] = 1
        elif torch.any(1 - done):
            tm[((1 - mask) * (dCp < 0)) > 0] = tc[((1 - mask) * (dCp < 0)) > 0]
            tp[((1 - mask) * (dCp >= 0)) > 0] = tc[((1 - mask) * (dCp >= 0)) > 0]
            tc[((1 - mask) * (1 - done)) > 0] = (tm[((1 - mask) * (1 - done)) > 0]
                                                   + tp[((1 - mask) * (1 - done)) > 0]) / 2
    return Cost(tc.detach(), u_values, v_values, u_cdf, v_cdf, p)


def emd1D_circle(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]
    device = u_values.device
    dtype = u_values.dtype
    if u_weights is None:
        u_weights = torch.full((n,), 1 / n, dtype=dtype, device=device)
    if v_weights is None:
        v_weights = torch.full((m,), 1 / m, dtype=dtype, device=device)
    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)
        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]
    if p == 1:
        values_sorted, values_sorter = torch.sort(torch.cat((u_values, v_values), -1), -1)
        cdf_diff = torch.cumsum(torch.gather(torch.cat((u_weights, -v_weights), -1), -1, values_sorter), -1)
        cdf_diff_sorted, cdf_diff_sorter = torch.sort(cdf_diff, axis=-1)
        values_sorted = torch.nn.functional.pad(values_sorted, (0, 1), value=1)
        delta = values_sorted[..., 1:] - values_sorted[..., :-1]
        weight_sorted = torch.gather(delta, -1, cdf_diff_sorter)
        sum_weights = torch.cumsum(weight_sorted, axis=-1) - 0.5
        sum_weights[sum_weights < 0] = np.inf
        inds = torch.argmin(sum_weights, axis=-1)
        levMed = torch.gather(cdf_diff_sorted, -1, inds.view(-1, 1))
        return torch.sum(delta * torch.abs(cdf_diff - levMed), axis=-1)


def w2_unif_circle(u_values):
    n = u_values.shape[-1]
    u_values, _ = torch.sort(u_values, -1)
    ns_n2 = torch.arange(n - 1, -n, -2, dtype=torch.float, device=u_values.device) / n ** 2
    cpt2 = torch.sum(ns_n2 * u_values, dim=-1)
    return torch.mean(u_values ** 2, axis=-1) - torch.mean(u_values, axis=-1) ** 2 + cpt2 + 1 / 12


# ---------------------------------------------------------------------------
# Core helpers (extracted to eliminate repetition)
# ---------------------------------------------------------------------------

def _get_kernel_fn(kernel: str):
    """Return the smoothing sample function for the given kernel name."""
    if kernel == 'vmf':
        return sample_vmf_batch
    elif kernel == 'ps':
        return _sample_ps_batch
    else:
        raise ValueError(f"Unsupported kernel '{kernel}'. Expected 'vmf' or 'ps'.")


def _sample_ps_batch(x, kappa):
    """Sample x' ~ PowerSpherical(x_i, kappa) for each x_i in a batch."""
    scale = torch.full((x.shape[0],), float(kappa), dtype=x.dtype, device=x.device)
    return PowerSpherical(x, scale).rsample()


def _sample_stiefel(num_projections, d, device):
    """Sample a random orthonormal frame from the Stiefel manifold V_{d,2}."""
    Z = torch.randn((num_projections, d, 2), device=device)
    U, _ = torch.linalg.qr(Z)
    return U


def _project_to_circle_coords(X, U):
    """
    Project points on S^{d-1} onto S^1 via a random 2D subspace U,
    returning angles normalized to [0, 1].

    Parameters:
    X : Tensor, shape (n, d)
    U : Tensor, shape (num_projections, d, 2)

    Returns:
    Tensor, shape (num_projections, n)  — angles in [0, 1]
    """
    n_projs = U.shape[0]
    n = X.shape[0]
    Xp = torch.matmul(torch.transpose(U, 1, 2)[:, None], X[:, :, None]).reshape(n_projs, n, 2)
    Xp = F.normalize(Xp, p=2, dim=-1)
    return (torch.atan2(-Xp[:, :, 1], -Xp[:, :, 0]) + np.pi) / (2 * np.pi)


def _project_and_smooth(X, U, kappa):
    """
    Project S^{d-1} -> S^1, smooth with VonMises(kappa), return [0,1] coords.
    vMF on S^1 == VonMises on angle, so we smooth directly on the scalar angle
    after atan2 instead of on the 2D vector — mathematically equivalent, cheaper.

    Parameters:
    X     : Tensor, shape (n, d)
    U     : Tensor, shape (num_projections, d, 2)
    kappa : float

    Returns:
    Tensor, shape (num_projections, n)  — smoothed angles in [0, 1]
    """
    n_projs, n = U.shape[0], X.shape[0]

    # Project onto plane then normalize to S^1 (2D)
    Xp = torch.matmul(torch.transpose(U, 1, 2)[:, None], X[:, :, None]).reshape(n_projs, n, 2)
    Xp = F.normalize(Xp, p=2, dim=-1)

    # Convert to scalar angle in [0, 1]
    coords = (torch.atan2(-Xp[:, :, 1], -Xp[:, :, 0]) + np.pi) / (2 * np.pi)

    # Smooth with VonMises — equivalent to vMF on S^1, no rejection sampling
    noise = torch.distributions.VonMises(
        torch.zeros_like(coords),
        torch.full_like(coords, kappa)
    ).sample() / (2 * np.pi)

    return (coords + noise) % 1.0


def _wasserstein_circle(Xps_coords, Yps_coords, p):
    """Compute batched W_p on the circle, shape (num_projections,)."""
    if p == 1:
        return emd1D_circle(Xps_coords, Yps_coords)
    else:
        return binary_search_circle(Xps_coords, Yps_coords, p=p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sliced_wasserstein_sphere(Xs, Xt, num_projections, device, u_weights=None, v_weights=None, p=2):
    """Spherical Sliced-Wasserstein distance (no smoothing)."""
    U = _sample_stiefel(num_projections, Xs.shape[1], device)
    Xps = _project_to_circle_coords(Xs.to(device), U)
    Xpt = _project_to_circle_coords(Xt.to(device), U)
    wd = _wasserstein_circle(Xps, Xpt, p)
    return torch.mean(wd)


def sliced_wasserstein_sphere_unif(Xs, num_projections, device):
    """SSW2 between an empirical distribution on S^{d-1} and the uniform."""
    U = _sample_stiefel(num_projections, Xs.shape[1], device)
    Xps = _project_to_circle_coords(Xs, U)
    return torch.mean(w2_unif_circle(Xps))


def gsssw(Xs, Xt, num_projections, device, kappa, p=2, kernel='vmf'):
    """
    Global Smoothed Spherical Sliced-Wasserstein.
    GSSSW_{p,kappa}(mu, nu) = SSW_p(mu * K_kappa, nu * K_kappa)
    """
    sample_fn = _get_kernel_fn(kernel)
    Xs_smooth = sample_fn(Xs, kappa)
    Xt_smooth = sample_fn(Xt, kappa)
    sw = sliced_wasserstein_sphere(Xs_smooth, Xt_smooth, num_projections, device, p=p)
    return sw


def gsssw_unif(Xs, num_projections, device, kappa, kernel='vmf'):
    """
    Global Smoothed SSW between an empirical distribution and the uniform.
    Uniform * K_kappa = Uniform, so only Xs is smoothed.
    """
    sample_fn = _get_kernel_fn(kernel)
    Xs_smooth = sample_fn(Xs, kappa)
    return sliced_wasserstein_sphere_unif(Xs_smooth, num_projections, device)

def psssw(Xs, Xt, num_projections, device, kappa, p=2):
    """
    Projection Smoothed Spherical Sliced-Wasserstein.
    PSSW_{p,kappa}(mu, nu) = E_P[W_p(P#mu * vMF_kappa, P#nu * vMF_kappa)]
    Smoothing is applied after projection onto S^1 via VonMises (= vMF on S^1).
    """
    U = _sample_stiefel(num_projections, Xs.shape[1], device)
    Xps_coords = _project_and_smooth(Xs, U, kappa)
    Yps_coords = _project_and_smooth(Xt, U, kappa)
    wd = _wasserstein_circle(Xps_coords, Yps_coords, p)
    return torch.mean(wd)


def psssw_unif(Xs, num_projections, device, kappa):
    """
    Projection Smoothed SSW between an empirical distribution and the uniform.
    Uniform * vMF_kappa = Uniform on S^1, so only Xs is smoothed.
    """
    U = _sample_stiefel(num_projections, Xs.shape[1], device)
    Xps_coords = _project_and_smooth(Xs, U, kappa)
    return torch.mean(w2_unif_circle(Xps_coords))
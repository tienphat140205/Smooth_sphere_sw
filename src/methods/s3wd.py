import torch
import torch.nn as nn
import torch.optim as optim
import geotorch
import numpy as np

from utils.misc import generate_rand_projs
from utils.s3w import get_stereo_proj_torch, epsilon_projection, hStar, RotationPool, unif_hypersphere


# ---------------------------------------------------------------------------
# Kernel helper (eliminates repeated import blocks)
# ---------------------------------------------------------------------------

def _get_kernel_fn(kernel: str):
    if kernel == 'vmf':
        try:
            from ..utils.vmf import sample_vmf_batch
        except ImportError:
            from utils.vmf import sample_vmf_batch
        return sample_vmf_batch
    elif kernel == 'ps':
        try:
            from .sswd import _sample_ps_batch
        except ImportError:
            from methods.sswd import _sample_ps_batch
        return _sample_ps_batch
    else:
        raise ValueError(f"Unsupported kernel '{kernel}'. Expected 'vmf' or 'ps'.")


# ---------------------------------------------------------------------------
# Base S3W distances (no smoothing)
# ---------------------------------------------------------------------------

def s3wd(X, Y, p, h=None, n_projs=1000, device='cpu', eps=1e-6):
    if h is None: h = hStar()
    X, Y = X.to(device), Y.to(device)
    X_sp = get_stereo_proj_torch(epsilon_projection(X, eps)).to(device)
    Y_sp = get_stereo_proj_torch(epsilon_projection(Y, eps)).to(device)
    s1_h = h(X_sp).double()
    s2_h = h(Y_sp).double()
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.transpose(0, 1), dim=1).values -
                  torch.sort(s2_h_rp.transpose(0, 1), dim=1).values)
    return d.pow(p).sum(dim=1).mean(dim=-1)


def s3wd_unif(X, p, h=None, n_projs=1000, device='cpu', eps=1e-6):
    if h is None: h = hStar()
    X = X.to(device)
    X_sp = get_stereo_proj_torch(epsilon_projection(X, eps)).to(device)
    s1_h = h(X_sp).double()
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp = s1_h @ projs.T
    Y_unif = unif_hypersphere((X.shape[0], X.shape[-1]), device=device)
    Y_unif_sp = get_stereo_proj_torch(epsilon_projection(Y_unif)).to(device)
    s2_h_rp = h(Y_unif_sp).double() @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.transpose(0, 1), dim=1).values -
                  torch.sort(s2_h_rp.transpose(0, 1), dim=1).values)
    return d.pow(p).sum(dim=1).mean(dim=-1)


def ri_s3wd(X, Y, p, h=None, n_projs=1000, n_rotations=1, device='cpu', eps=1e-6):
    if h is None: h = hStar()
    X, Y = X.to(device), Y.to(device)
    n = X.shape[-1]
    rot_matrices = torch.stack([
        geotorch.SO(torch.Size([n, n])).sample('uniform') for _ in range(n_rotations)
    ]).to(device)
    X_rot = (rot_matrices @ X.T).permute(0, 2, 1)
    Y_rot = (rot_matrices @ Y.T).permute(0, 2, 1)
    X_sp = get_stereo_proj_torch(epsilon_projection(X_rot, eps)).to(device)
    Y_sp = get_stereo_proj_torch(epsilon_projection(Y_rot, eps)).to(device)
    s1_h = h(X_sp).double()
    s2_h = h(Y_sp).double()
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values -
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)
    return d.pow(p).sum(dim=-1).mean(dim=-1).mean()


def ri_s3wd_unif(X, p, h=None, n_projs=1000, n_rotations=1, device='cpu', eps=1e-6):
    if h is None: h = hStar()
    X = X.to(device)
    n = X.shape[-1]
    rot_matrices = torch.stack([
        geotorch.SO(torch.Size([n, n])).sample('uniform') for _ in range(n_rotations)
    ]).to(device)
    X_rot = (rot_matrices @ X.T).permute(0, 2, 1)
    X_sp = get_stereo_proj_torch(epsilon_projection(X_rot, eps)).to(device)
    Y_unif = unif_hypersphere((n_rotations, X.shape[0], n), device=device)
    Y_unif_sp = get_stereo_proj_torch(epsilon_projection(Y_unif)).to(device)
    s1_h = h(X_sp).double()
    s2_h = h(Y_unif_sp).double()
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values -
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)
    return d.pow(p).sum(dim=-1).mean(dim=-1).mean()


def ari_s3wd(X, Y, p, h=None, n_projs=1000, n_rotations=1, pool_size=100, device='cpu', eps=1e-6):
    if h is None: h = hStar()
    assert pool_size >= n_rotations
    X, Y = X.to(device), Y.to(device)
    n = X.shape[-1]
    rot_matrices = RotationPool.get(n, pool_size, device=device)[
        torch.randperm(pool_size)[:n_rotations]
    ]
    X_rot = (rot_matrices @ X.T).permute(0, 2, 1)
    Y_rot = (rot_matrices @ Y.T).permute(0, 2, 1)
    X_sp = get_stereo_proj_torch(epsilon_projection(X_rot, eps)).to(device)
    Y_sp = get_stereo_proj_torch(epsilon_projection(Y_rot, eps)).to(device)
    s1_h = h(X_sp).double()
    s2_h = h(Y_sp).double()
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values -
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)
    return d.pow(p).sum(dim=-1).mean(dim=-1).mean()


def ari_s3wd_unif(X, p, h=None, n_projs=1000, n_rotations=1, pool_size=100, device='cpu', eps=1e-6):
    if h is None: h = hStar()
    assert pool_size >= n_rotations
    X = X.to(device)
    n = X.shape[-1]
    rot_matrices = RotationPool.get(n, pool_size, device=device)[
        torch.randperm(pool_size)[:n_rotations]
    ].to(device)
    X_rot = (rot_matrices @ X.T).permute(0, 2, 1)
    X_sp = get_stereo_proj_torch(epsilon_projection(X_rot, eps)).to(device)
    Y_unif = unif_hypersphere((n_rotations, X.shape[0], n), device=device)
    Y_unif_sp = get_stereo_proj_torch(epsilon_projection(Y_unif)).to(device)
    s1_h = h(X_sp).double()
    s2_h = h(Y_unif_sp).double()
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values -
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)
    return d.pow(p).sum(dim=-1).mean(dim=-1).mean()


# ---------------------------------------------------------------------------
# Global Smoothed S3W  (prefix: gs)
# GS: smooth mu, nu with K_kappa BEFORE passing to base distance
# ---------------------------------------------------------------------------

def gs_s3wd(Xs, Xt, kappa, p=2, h=None, n_projs=1000, device='cpu', eps=1e-6, kernel='vmf'):
    """Global Smoothed S3W. GS_{p,kappa}(mu,nu) = S3W_p(mu*K, nu*K)"""
    sample_fn = _get_kernel_fn(kernel)
    return s3wd(sample_fn(Xs, kappa), sample_fn(Xt, kappa),
                p=p, h=h, n_projs=n_projs, device=device, eps=eps)


def gs_s3wd_unif(Xs, kappa, p=2, h=None, n_projs=1000, device='cpu', eps=1e-6, kernel='vmf'):
    """Global Smoothed S3W vs Uniform. Uniform*K = Uniform, so only Xs smoothed."""
    return s3wd_unif(
        _get_kernel_fn(kernel)(Xs, kappa),
        p=p, h=h, n_projs=n_projs, device=device, eps=eps,
    )


def gs_ri_s3wd(Xs, Xt, kappa, p=2, h=None, n_projs=1000, n_rotations=1, device='cpu', eps=1e-6, kernel='vmf'):
    """Global Smoothed Rotation-Invariant S3W."""
    sample_fn = _get_kernel_fn(kernel)
    return ri_s3wd(sample_fn(Xs, kappa), sample_fn(Xt, kappa),
                   p=p, h=h, n_projs=n_projs, n_rotations=n_rotations, device=device, eps=eps)


def gs_ri_s3wd_unif(Xs, kappa, p=2, h=None, n_projs=1000, n_rotations=1, device='cpu', eps=1e-6, kernel='vmf'):
    """Global Smoothed RI-S3W vs Uniform."""
    return ri_s3wd_unif(
        _get_kernel_fn(kernel)(Xs, kappa),
        p=p, h=h, n_projs=n_projs, n_rotations=n_rotations, device=device, eps=eps,
    )


def gs_ari_s3wd(Xs, Xt, kappa, p=2, h=None, n_projs=1000, n_rotations=1, pool_size=100, device='cpu', eps=1e-6, kernel='vmf'):
    """Global Smoothed Averaged-RI S3W."""
    sample_fn = _get_kernel_fn(kernel)
    return ari_s3wd(sample_fn(Xs, kappa), sample_fn(Xt, kappa),
                    p=p, h=h, n_projs=n_projs, n_rotations=n_rotations,
                    pool_size=pool_size, device=device, eps=eps)


def gs_ari_s3wd_unif(Xs, kappa, p=2, h=None, n_projs=1000, n_rotations=1, pool_size=100, device='cpu', eps=1e-6, kernel='vmf'):
    """Global Smoothed ARI-S3W vs Uniform."""
    return ari_s3wd_unif(
        _get_kernel_fn(kernel)(Xs, kappa),
        p=p, h=h, n_projs=n_projs, n_rotations=n_rotations,
        pool_size=pool_size, device=device, eps=eps,
    )


# ---------------------------------------------------------------------------
# Projection Smoothed S3W  (prefix: ps, Definition 3 in paper)
# PS3W_{p,kappa}(mu,nu) = E_P[W_p^p(P#mu * N_{1/kappa}, P#nu * N_{1/kappa})]
# P: S^{d-1} -> R (stereographic), smooth with Gaussian AFTER projection
# ---------------------------------------------------------------------------

def _stereo_project(X, h, eps, device):
    """Stereographic project S^{d-1} -> R^{d-1}, apply feature map h."""
    X_sp = get_stereo_proj_torch(epsilon_projection(X, eps)).to(device)
    return h(X_sp).double()


def ps_s3wd(Xs, Xt, kappa, p=2, h=None, n_projs=1000, device='cpu', eps=1e-6):
    """
    Projection Smoothed S3W (Definition 3).
    Smooth with N(0, 1/kappa) AFTER stereographic projection onto R.
    """
    if h is None: h = hStar()
    Xs, Xt = Xs.to(device), Xt.to(device)

    s1_h = _stereo_project(Xs, h, eps, device)   # (n, d')
    s2_h = _stereo_project(Xt, h, eps, device)   # (m, d')

    # Convolve with N(0, 1/kappa): add iid Gaussian noise
    sigma = (1.0 / kappa) ** 0.5
    s1_h = s1_h + sigma * torch.randn_like(s1_h)
    s2_h = s2_h + sigma * torch.randn_like(s2_h)

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(0, 1), dim=1).values -
                  torch.sort(s2_h_rp.transpose(0, 1), dim=1).values)
    return d.pow(p).sum(dim=1).mean(dim=-1)


def ps_s3wd_unif(Xs, kappa, p=2, h=None, n_projs=1000, device='cpu', eps=1e-6):
    """Projection Smoothed S3W vs Uniform (Definition 3)."""
    if h is None: h = hStar()
    Xs = Xs.to(device)

    s1_h = _stereo_project(Xs, h, eps, device)
    sigma = (1.0 / kappa) ** 0.5
    s1_h = s1_h + sigma * torch.randn_like(s1_h)

    Y_unif = unif_hypersphere((Xs.shape[0], Xs.shape[-1]), device=device)
    s2_h = _stereo_project(Y_unif, h, eps, device)
    s2_h = s2_h + sigma * torch.randn_like(s2_h)

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(0, 1), dim=1).values -
                  torch.sort(s2_h_rp.transpose(0, 1), dim=1).values)
    return d.pow(p).sum(dim=1).mean(dim=-1)


def _ps_sliced_cost(s1_h, s2_h, kappa, p, n_projs, device):
    """
    Shared core for all ps_*_s3wd variants:
    add Gaussian noise then compute sliced Wasserstein on R.
    s1_h, s2_h: already stereographic-projected + h-mapped, shape (..., n/m, d')
    """
    sigma = (1.0 / kappa) ** 0.5
    s1_h = s1_h + sigma * torch.randn_like(s1_h)
    s2_h = s2_h + sigma * torch.randn_like(s2_h)
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp = s1_h @ projs.T
    s2_h_rp = s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values -
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)
    return d.pow(p).sum(dim=-1).mean(dim=-1).mean()


def ps_ri_s3wd(Xs, Xt, kappa, p=2, h=None, n_projs=1000, n_rotations=1, device='cpu', eps=1e-6):
    """Projection Smoothed Rotation-Invariant S3W (Definition 3)."""
    if h is None: h = hStar()
    Xs, Xt = Xs.to(device), Xt.to(device)
    n = Xs.shape[-1]
    rot_matrices = torch.stack([
        geotorch.SO(torch.Size([n, n])).sample('uniform') for _ in range(n_rotations)
    ]).to(device)
    Xs_rot = (rot_matrices @ Xs.T).permute(0, 2, 1)
    Xt_rot = (rot_matrices @ Xt.T).permute(0, 2, 1)
    s1_h = h(get_stereo_proj_torch(epsilon_projection(Xs_rot, eps)).to(device)).double()
    s2_h = h(get_stereo_proj_torch(epsilon_projection(Xt_rot, eps)).to(device)).double()
    return _ps_sliced_cost(s1_h, s2_h, kappa, p, n_projs, device)


def ps_ri_s3wd_unif(Xs, kappa, p=2, h=None, n_projs=1000, n_rotations=1, device='cpu', eps=1e-6):
    """Projection Smoothed RI-S3W vs Uniform (Definition 3)."""
    if h is None: h = hStar()
    Xs = Xs.to(device)
    n = Xs.shape[-1]
    rot_matrices = torch.stack([
        geotorch.SO(torch.Size([n, n])).sample('uniform') for _ in range(n_rotations)
    ]).to(device)
    Xs_rot = (rot_matrices @ Xs.T).permute(0, 2, 1)
    Y_unif = unif_hypersphere((n_rotations, Xs.shape[0], n), device=device)
    s1_h = h(get_stereo_proj_torch(epsilon_projection(Xs_rot, eps)).to(device)).double()
    s2_h = h(get_stereo_proj_torch(epsilon_projection(Y_unif, eps)).to(device)).double()
    return _ps_sliced_cost(s1_h, s2_h, kappa, p, n_projs, device)


def ps_ari_s3wd(Xs, Xt, kappa, p=2, h=None, n_projs=1000, n_rotations=1, pool_size=100, device='cpu', eps=1e-6):
    """Projection Smoothed Averaged-RI S3W (Definition 3)."""
    if h is None: h = hStar()
    assert pool_size >= n_rotations
    Xs, Xt = Xs.to(device), Xt.to(device)
    n = Xs.shape[-1]
    rot_matrices = RotationPool.get(n, pool_size, device=device)[
        torch.randperm(pool_size)[:n_rotations]
    ]
    Xs_rot = (rot_matrices @ Xs.T).permute(0, 2, 1)
    Xt_rot = (rot_matrices @ Xt.T).permute(0, 2, 1)
    s1_h = h(get_stereo_proj_torch(epsilon_projection(Xs_rot, eps)).to(device)).double()
    s2_h = h(get_stereo_proj_torch(epsilon_projection(Xt_rot, eps)).to(device)).double()
    return _ps_sliced_cost(s1_h, s2_h, kappa, p, n_projs, device)


def ps_ari_s3wd_unif(Xs, kappa, p=2, h=None, n_projs=1000, n_rotations=1, pool_size=100, device='cpu', eps=1e-6):
    """Projection Smoothed ARI-S3W vs Uniform (Definition 3)."""
    if h is None: h = hStar()
    assert pool_size >= n_rotations
    Xs = Xs.to(device)
    n = Xs.shape[-1]
    rot_matrices = RotationPool.get(n, pool_size, device=device)[
        torch.randperm(pool_size)[:n_rotations]
    ].to(device)
    Xs_rot = (rot_matrices @ Xs.T).permute(0, 2, 1)
    Y_unif = unif_hypersphere((n_rotations, Xs.shape[0], n), device=device)
    s1_h = h(get_stereo_proj_torch(epsilon_projection(Xs_rot, eps)).to(device)).double()
    s2_h = h(get_stereo_proj_torch(epsilon_projection(Y_unif, eps)).to(device)).double()
    return _ps_sliced_cost(s1_h, s2_h, kappa, p, n_projs, device)
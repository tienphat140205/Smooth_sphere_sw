import torch
import torch.nn.functional as F
import numpy as np
from utils.misc import rand_t_marginal, rand_u_hypersphere
from scipy.linalg import null_space

def pdf_vmf(x, mu, kappa):
    return torch.exp(kappa * torch.matmul(mu, x.T))[0]

def pdf_von_Mises(theta,mu,kappa):
    pdf = np.exp(kappa * np.cos(theta - mu)) / (2.0*np.pi*iv(0,kappa))
    return pdf

def rand_vmf(mu, kappa, N=1):
    """Sample from vMF using null_space rotation (NumPy, non-differentiable)."""
    if torch.is_tensor(mu): mu = mu.numpy()  
    p = len(mu)
    mu = np.reshape(mu, (p, 1)).astype(float)
    mu /= np.linalg.norm(mu, axis=0)
    samples = np.zeros((N, p))
    t = rand_t_marginal(kappa, p, N) 
    xi = rand_u_hypersphere(N, p - 1) 
    samples[:, [0]] = t
    samples[:, 1:] = np.sqrt(1 - t**2) * xi
    O = null_space(mu.T)
    R = np.concatenate((mu, O), axis=1)
    return np.dot(R, samples.T).T


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


def fibonacci_sphere(samples=10):
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
    y = 1 - 2*np.arange(samples)/float(samples-1)
    radius = np.sqrt(1-y**2)
    theta = phi*np.arange(samples)
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    points = np.stack([x,y,z],axis=-1)
    return points

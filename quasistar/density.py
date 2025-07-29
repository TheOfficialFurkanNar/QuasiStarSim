# density.py

"""
Module for computing accretion disk density profiles in QuasiStarSim.
Includes:
  - Shakura–Sunyaev model
  - Power-law profile
  - Scale height H(r)
  - Parallel radial‐bin evaluation with joblib & tqdm
  - Per‐step logging
"""

from typing import Dict, Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from quasistar.constants import (
    BOLTZMANN_CONSTANT,
    GRAVITATIONAL_CONSTANT,
    PROTON_MASS,
    SPEED_OF_LIGHT,
)
from physics import _validate_positive
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "shakura_sunyaev_density",
    "power_law_density",
    "scale_height",
    "calculate_density_profile",
]


def shakura_sunyaev_density(
    r: float,
    M: float,
    m_dot: float,
    alpha: float,
    T: float
) -> float:
    """
    Compute volume density ρ(r) via Shakura–Sunyaev (1973) model.

    Parameters
    ----------
    r       radius (m)
    M       black hole mass (kg)
    m_dot   mass accretion rate (kg/s)
    alpha   viscosity parameter (0 < alpha ≤ 1)
    T       local disk temperature (K)

    Returns
    -------
    ρ       volume density (kg/m³)
    """
    _validate_positive(r=r, M=M, m_dot=m_dot, T=T)
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0,1]. Got {alpha}")

    # Gravitational radius and Kepler frekansı
    r_g = GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2
    Omega_K = np.sqrt(GRAVITATIONAL_CONSTANT * M / r**3)

    # Ölçek yüksekliği
    H = scale_height(r, T, M)

    # Surface density Σ ve volume density ρ
    Sigma = m_dot / (3 * np.pi * alpha * Omega_K * H**2)
    rho = Sigma / (2 * H)

    logger.debug(f"ShSu bin: r={r:.3e}, rho={rho:.3e}")
    return rho


def power_law_density(
    r: float,
    rho0: float,
    r0: float,
    p: float
) -> float:
    """
    Power-law volume density: ρ(r) = ρ0 * (r/r0)^(-p).
    """
    _validate_positive(r=r, rho0=rho0, r0=r0)
    rho = rho0 * (r / r0) ** (-p)
    logger.debug(f"PowerLaw bin: r={r:.3e}, rho={rho:.3e}, p={p}")
    return rho


def scale_height(
    r: float,
    T: float,
    M: float,
    mu: float = 0.615
) -> float:
    """
    Disk scale height H = c_s / Ω_K, where c_s is sound speed.
    """
    _validate_positive(r=r, T=T, M=M)
    c_s = np.sqrt(BOLTZMANN_CONSTANT * T / (mu * PROTON_MASS))
    Omega_K = np.sqrt(GRAVITATIONAL_CONSTANT * M / r**3)
    H = c_s / Omega_K
    logger.debug(f"ScaleHeight bin: r={r:.3e}, H={H:.3e}")
    return H


def calculate_density_profile(
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Compute full radial density profile in parallel.

    Required params keys:
      - num_radial_bins: int
      - inner_radius: float
      - outer_radius: float
      - n_jobs: int
      - density_model: "shakura_sunyaev" or "power_law"
    Model-specific params:
      * if "shakura_sunyaev":
          - mass (kg), m_dot (kg/s), alpha, T_init (float or array)
      * if "power_law":
          - rho0, r0, p

    Returns
    -------
    np.ndarray of shape (num_radial_bins,)
    """
    # Temel doğrulamalar
    req = ["num_radial_bins", "inner_radius", "outer_radius", "n_jobs", "density_model"]
    missing = [k for k in req if k not in params]
    if missing:
        raise KeyError(f"Missing params for density profile: {missing}")

    n_bins = params["num_radial_bins"]
    inner_r = params["inner_radius"]
    outer_r = params["outer_radius"]
    n_jobs = params["n_jobs"]
    model = params["density_model"]

    # Radial grid
    r_grid = np.linspace(inner_r, outer_r, n_bins)

    # Model seçimi ve paralel hesap
    if model == "shakura_sunyaev":
        M     = params["mass"]
        m_dot = params["m_dot"]
        alpha = params.get("alpha", 0.1)
        T_ini = params["T_init"]
        # T_init sabit ise tüm grid’e uygula
        if np.isscalar(T_ini):
            T_grid = np.full(n_bins, T_ini)
        else:
            T_grid = np.asarray(T_ini)
            if T_grid.shape[0] != n_bins:
                raise ValueError("T_init array length must equal num_radial_bins")

        logger.info(f"Computing Shakura–Sunyaev density ({n_bins} bins, {n_jobs} jobs)")
        func = lambda i: shakura_sunyaev_density(
            r_grid[i], M, m_dot, alpha, T_grid[i]
        )

    elif model == "power_law":
        rho0, r0, p = params["rho0"], params["r0"], params["p"]
        logger.info(f"Computing power-law density ({n_bins} bins, {n_jobs} jobs)")
        func = lambda i: power_law_density(r_grid[i], rho0, r0, p)

    else:
        raise ValueError(f"Unknown density_model '{model}'")

    # Paralel ve ilerleme çubuğu
    densities = Parallel(n_jobs=n_jobs)(
        delayed(func)(i) for i in tqdm(range(n_bins), desc="Density bins")
    )

    profile = np.array(densities)
    logger.info("Density profile computation completed")
    return profile
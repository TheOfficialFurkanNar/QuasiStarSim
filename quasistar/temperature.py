# temperature.py

"""
Module for computing accretion disk temperature profiles in QuasiStarSim.
Provides:
  - effective_temperature: steady‐state α‐disk surface temperature
  - iterative_temperature: Newton‐Raphson solution of σT⁴ = viscous flux
  - solve_temperature_distribution: parallel radial‐bin evaluation
"""

from typing import Dict, Any, Union, Optional

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from quasistar.constants import GRAVITATIONAL_CONSTANT, STEFAN_BOLTZMANN_CONSTANT
from quasistar.physics import isco_radius, _validate_positive
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "effective_temperature",
    "iterative_temperature",
    "solve_temperature_distribution",
]


def effective_temperature(
    r: Union[float, np.floating],
    M: Union[float, np.floating],
    m_dot: Union[float, np.floating],
    r_in: Optional[Union[float, np.floating]] = None,
    a: float = 0.0
) -> float:
    """
    Surface effective temperature T_eff(r) of a steady‐state α‐disk.

    T_eff(r) = [3 G M ṁ / (8 π σ r³) * (1 − √(r_in / r))]^(1/4)

    Parameters
    ----------
    r     : radius (m)
    M     : black hole mass (kg)
    m_dot : mass accretion rate (kg/s)
    r_in  : inner radius (m). If None, uses ISCO(a, M)
    a     : spin parameter (−1 ≤ a ≤ 1)

    Returns
    -------
    Teff  : effective temperature (K)
    """
    # Determine inner radius
    if r_in is None:
        if abs(a) > 1:
            raise ValueError(f"Spin |a| ≤ 1 required. Got {a}")
        r_in = isco_radius(a, M)

    # Sanity checks
    _validate_positive(r=r, M=M, m_dot=m_dot, r_in=r_in)
    if r <= r_in:
        raise ValueError(f"r ({r}) must be > r_in ({r_in})")

    # Flux calculation
    num = 3 * GRAVITATIONAL_CONSTANT * M * m_dot
    den = 8 * np.pi * STEFAN_BOLTZMANN_CONSTANT * r**3
    factor = num / den * (1 - np.sqrt(r_in / r))

    Teff = factor**0.25
    logger.debug(f"Effective_T bin: r={r:.3e}, T={Teff:.3e}")
    return Teff


def iterative_temperature(
    r: Union[float, np.floating],
    M: Union[float, np.floating],
    m_dot: Union[float, np.floating],
    alpha: float = 0.1,
    a: float = 0.0,
    initial_T: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> float:
    """
    Solve for surface temperature T via Newton‐Raphson on σT⁴ = F_visc(r).

    Parameters
    ----------
    r         : radius (m)
    M         : black hole mass (kg)
    m_dot     : mass accretion rate (kg/s)
    alpha     : viscosity parameter (0 < α ≤ 1)
    a         : spin (−1 ≤ a ≤ 1)
    initial_T : initial guess (K). If None, uses effective_temperature
    max_iter  : max iterations
    tol       : convergence tolerance on |ΔT|

    Returns
    -------
    T         : surface temperature (K)
    """
    # Validations
    _validate_positive(r=r, M=M, m_dot=m_dot)
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0,1]. Got {alpha}")
    if abs(a) > 1:
        raise ValueError(f"Spin |a| ≤ 1 required. Got {a}")

    # Inner radius and bounds
    r_in = isco_radius(a, M)
    if r <= r_in:
        raise ValueError(f"r ({r}) must be > r_in ({r_in})")

    # Initial guess
    if initial_T is None:
        initial_T = effective_temperature(r, M, m_dot, r_in=r_in, a=a)
    _validate_positive(initial_T=initial_T)

    T = float(initial_T)
    for i in range(max_iter):
        F_visc = (3 * GRAVITATIONAL_CONSTANT * M * m_dot /
                 (8 * np.pi * r**3) *
                 (1 - np.sqrt(r_in / r)))
        F_rad = STEFAN_BOLTZMANN_CONSTANT * T**4

        resid = F_rad - F_visc
        dresdT = 4 * STEFAN_BOLTZMANN_CONSTANT * T**3

        delta = resid / dresdT
        T -= delta

        logger.debug(f"Iter_T bin r={r:.3e}, iter={i}, T={T:.3e}, ΔT={delta:.3e}")
        if abs(delta) < tol:
            return T

    raise RuntimeError(f"iterative_temperature did not converge after {max_iter} iterations")


def solve_temperature_distribution(
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Compute radial temperature profile in parallel over bins.

    Required params keys:
      - num_radial_bins: int
      - inner_radius: float
      - outer_radius: float
      - n_jobs: int
      - temp_model: "effective" or "iterative"
    Model-specific params:
      - mass, m_dot
      - for iterative: alpha, spin, initial_T, max_iter, tol

    Returns
    -------
    1D numpy array of temperatures for each radial bin.
    """
    # Parameter checks
    req = ["num_radial_bins", "inner_radius", "outer_radius", "n_jobs", "temp_model"]
    missing = [k for k in req if k not in params]
    if missing:
        raise KeyError(f"Missing params for temperature distribution: {missing}")

    n_bins = params["num_radial_bins"]
    inner_r = params["inner_radius"]
    outer_r = params["outer_radius"]
    n_jobs = params["n_jobs"]
    model = params["temp_model"]

    # Build radial grid
    r_grid = np.linspace(inner_r, outer_r, n_bins)

    # Choose model function
    if model == "effective":
        func = lambda i: effective_temperature(
            r_grid[i],
            params["mass"],
            params["m_dot"],
            params.get("inner_radius"),
            params.get("spin", 0.0)
        )
        logger.info(f"Computing effective temperature ({n_bins} bins, {n_jobs} jobs)")

    elif model == "iterative":
        func = lambda i: iterative_temperature(
            r_grid[i],
            params["mass"],
            params["m_dot"],
            params.get("alpha", 0.1),
            params.get("spin", 0.0),
            params.get("initial_T", None),
            params.get("max_iter", 100),
            params.get("tol", 1e-6)
        )
        logger.info(f"Computing iterative temperature ({n_bins} bins, {n_jobs} jobs)")

    else:
        raise ValueError(f"Unknown temp_model '{model}'")

    # Parallel execution with progress bar
    temps = Parallel(n_jobs=n_jobs)(
        delayed(func)(i)
        for i in tqdm(range(n_bins), desc="Temperature bins")
    )

    profile = np.array(temps)
    logger.info("Temperature distribution computation completed")
    return profile

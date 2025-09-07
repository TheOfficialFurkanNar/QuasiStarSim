import numpy as np
from pint import Quantity

from quasistar.constants import (
    PLANCK_CONSTANT,
    BOLTZMANN_CONSTANT,
    SPEED_OF_LIGHT,
    PROTON_MASS,
    THOMSON_CROSS_SECTION,
)
from temperature import effective_temperature
from quasistar.physics import _validate_positive
from quasistar.config_logging import get_logger

logger = get_logger(__name__)

__all__ = ["planck_nu", "disk_spectrum", "generate_spectrum"]

def planck_nu(nu: float | np.ndarray | Quantity, T: float | Quantity) -> float | np.ndarray | Quantity:
    """Planck dağılımı per birim frekans: B_nu(T)."""
    _validate_positive(nu=nu, T=T)
    nu = Quantity(nu).to("hertz").magnitude if hasattr(nu, "to") else np.atleast_1d(nu)
    T = Quantity(T).to("kelvin").magnitude if hasattr(T, "to") else T
    num = 2 * PLANCK_CONSTANT * nu**3 / SPEED_OF_LIGHT**2
    expo = np.exp(PLANCK_CONSTANT * nu / (BOLTZMANN_CONSTANT * T)) - 1
    return num / expo

def disk_spectrum(nu: float | np.ndarray | Quantity, r: float | np.ndarray | Quantity, M: float | Quantity, m_dot: float | Quantity, r_out: float | Quantity = None, a: float = 0.0, inclination: float = 0.0, distance: float | Quantity = None) -> Quantity:
    """Accretion disk spektrumu: L_nu veya gözlenen F_nu."""
    _validate_positive(M=M, m_dot=m_dot)
    if distance is not None:
        _validate_positive(distance=distance)
    if not (-1.0 <= a <= 1.0):
        raise ValueError(f"Spin |a| ≤ 1 olmalı. Got {a}.")
    if not (0 <= inclination <= np.pi):
        raise ValueError(f"Inclination must be between 0 and π radians. Got {inclination}.")

    if r_out is None:
        r_out = np.max(r)
    _validate_positive(r=r, r_out=r_out)
    if np.any(r <= 0) or r_out <= 0:
        raise ValueError("r ve r_out pozitif olmalı.")
    if np.any(np.diff(r) < 0):
        raise ValueError("r must be monotonically increasing for integration.")

    # Eddington limit check
    L_Edd = 4 * np.pi * G * M * PROTON_MASS / THOMSON_CROSS_SECTION
    if m_dot * SPEED_OF_LIGHT**2 > L_Edd:
        logger.warning(f"m_dot exceeds Eddington limit. L = {m_dot * SPEED_OF_LIGHT**2:.2e} W, L_Edd = {L_Edd:.2e} W")

    mu = np.cos(inclination)
    T_eff = effective_temperature(r, M, m_dot, a=a)
    B_nu = planck_nu(nu=np.atleast_1d(nu)[:, None], T=T_eff[None, :])
    integrand = 2 * np.pi * B_nu * r[None, :] * mu
    L_nu = np.trapz(integrand, x=r, axis=1, dx=np.diff(r).mean())  # Approximate dx

    if distance is not None:
        distance = Quantity(distance).to("meter").magnitude if hasattr(distance, "to") else distance
        return L_nu / (4 * np.pi * distance**2)
    return L_nu

def generate_spectrum(params: dict) -> dict:
    logger.info("Generating spectrum")
    freqs = params["frequencies"]
    radii = params["radii"]
    M = params["mass"]
    m_dot = params["m_dot"]
    r_out = params.get("r_out", None)
    a = params.get("spin", 0.0)
    inclination = params.get("inclination", 0.0)
    distance = params.get("distance", None)

    freqs = Quantity(freqs).to("hertz").magnitude if hasattr(freqs, "to") else np.atleast_1d(freqs)
    radii = Quantity(radii).to("meter").magnitude if hasattr(radii, "to") else np.atleast_1d(radii)
    M = Quantity(M).to("kilogram").magnitude if hasattr(M, "to") else M
    m_dot = Quantity(m_dot).to("kilogram/second").magnitude if hasattr(m_dot, "to") else m_dot
    if r_out is not None:
        r_out = Quantity(r_out).to("meter").magnitude if hasattr(r_out, "to") else r_out
    if distance is not None:
        distance = Quantity(distance).to("meter").magnitude if hasattr(distance, "to") else distance

    spectrum_values = disk_spectrum(nu=freqs, r=radii, M=M, m_dot=m_dot, r_out=r_out, a=a, inclination=inclination, distance=distance)
    logger.info(f"Spectrum generated with shape {spectrum_values.shape}")

    return {
        "frequency": freqs,
        "flux_or_luminosity": spectrum_values
    }
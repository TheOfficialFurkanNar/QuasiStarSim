"""
density.py

Accretion disk yoğunluk profilleri:
Shakura–Sunyaev modeli (hacim yoğunluğu),
güç‐kanunlu profil ve ölçek yüksekliği hesaplamaları.

Tüm fonksiyonlar constants.py’daki sabitleri,
physics._validate_positive helper’ını kullanır.
"""

from constants import (
    BOLTZMANN_CONSTANT,
    GRAVITATIONAL_CONSTANT,
    PROTON_MASS,
    SPEED_OF_LIGHT,
)
from physics import _validate_positive
import numpy as np

__all__ = [
    "shakura_sunyaev_density",
    "power_law_density",
    "scale_height",
]


def shakura_sunyaev_density(r, M, m_dot, alpha=0.1, T):
    """
    Shakura–Sunyaev (1973) modeliyle hacim yoğunluğu ρ(r).

    Parameters
    ----------
    r : float or Quantity
        Yarıçap (m).
    M : float or Quantity
        Kara delik kütlesi (kg).
    m_dot : float or Quantity
        Kütle akış hızı (kg/s).
    alpha : float, optional
        Viskozite parametresi (0 < alpha ≤ 1).
    T : float or Quantity
        Disk sıcaklığı (K).

    Returns
    -------
    float or Quantity
        Hacim yoğunluğu ρ(r) (kg/m³).

    Raises
    ------
    ValueError
        Eğer r, M, m_dot veya T pozitif değilse,
        ya da alpha geçersizse.
    """
    _validate_positive(r=r, M=M, m_dot=m_dot, T=T)
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0,1]. Got {alpha}.")

    # Gravitational radius
    r_g = GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2

    # Kepler açısal frekansı
    Omega_K = np.sqrt(GRAVITATIONAL_CONSTANT * M / r**3)

    # Ölçek yüksekliği
    H = scale_height(r, T, M)

    # Surface density Σ = ṁ / (3π α Ω_K H^2)
    Sigma = m_dot / (3 * np.pi * alpha * Omega_K * H**2)

    # Volume density ρ = Σ / (2 H)
    return Sigma / (2 * H)


def power_law_density(r, rho0, r0, p):
    """
    Güç‐kanunlu hacim yoğunluk profili: ρ(r) = ρ0 * (r / r0)^(-p).

    ...
    """
    _validate_positive(r=r, rho0=rho0, r0=r0)
    return rho0 * (r / r0) ** (-p)


def scale_height(r, T, M, mu=0.615):
    """
    Disk ölçek yüksekliği H = c_s / Ω_K.

    ...
    """
    _validate_positive(r=r, T=T, M=M)

    c_s = np.sqrt(BOLTZMANN_CONSTANT * T / (mu * PROTON_MASS))
    Omega_K = np.sqrt(GRAVITATIONAL_CONSTANT * M / r**3)
    return c_s / Omega_K
"""
temperature.py

Accretion disk sıcaklık profilleri:
- Etkin sıcaklık (steady–state α-disk)
- T–ν iteratif yüzey sıcaklığı çözümü
"""

import numpy as np
from pint import Quantity

from constants import (
    GRAVITATIONAL_CONSTANT,
    STEFAN_BOLTZMANN_CONSTANT,
)
from physics import isco_radius, _validate_positive

__all__ = [
    "effective_temperature",
    "iterative_temperature",
]


def effective_temperature(
    r: float | Quantity,
    M: float | Quantity,
    m_dot: float | Quantity,
    r_in: float | Quantity = None,
    a: float = 0.0
) -> float | Quantity:
    """
    Shakura–Sunyaev diskinin yüzey etkin sıcaklığı T_eff(r).

    T_eff(r) = [3 G M ṁ / (8 π σ r³) * (1 − √(r_in / r))]^(1/4)

    Parameters
    ----------
    r    : float veya pint.Quantity
           Disk yarıçapı (m).
    M    : float veya pint.Quantity
           Kara delik kütlesi (kg).
    m_dot: float veya pint.Quantity
           Kütle akış hızı (kg/s).
    r_in : float veya pint.Quantity, optional
           İç yarıçap (m). None ise ISCO(a,M) hesaplanır.
    a    : float, optional
           Spin parametresi (−1 ≤ a ≤ 1).

    Returns
    -------
    Etkin yüzey sıcaklığı (K).

    Raises
    ------
    ValueError
        - r, M, m_dot veya r_in pozitif değilse
        - |a| > 1
        - r ≤ r_in
    """
    # İç yarıçap belirleme ve spin kontrolü
    if r_in is None:
        if abs(a) > 1:
            raise ValueError(f"Spin |a| ≤ 1 olmalı. Got {a}.")
        r_in = isco_radius(a, M)

    # Pozitiflik kontrolü
    _validate_positive(r=r, M=M, m_dot=m_dot, r_in=r_in)

    if r <= r_in:
        raise ValueError(f"r ({r}) > r_in ({r_in}) olmalı.")

    # Hesaplamalar
    prefactor    = 3 * GRAVITATIONAL_CONSTANT * M * m_dot
    denominator  = 8 * np.pi * STEFAN_BOLTZMANN_CONSTANT * r**3
    flux_term    = prefactor / denominator
    shape_factor = 1 - np.sqrt(r_in / r)

    return (flux_term * shape_factor) ** 0.25


def iterative_temperature(
    r: float | Quantity,
    M: float | Quantity,
    m_dot: float | Quantity,
    alpha: float = 0.1,
    a: float = 0.0,
    initial_T: float | Quantity = None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> float | Quantity:
    """
    Newton yöntemiyle T–ν dengesi: σ T⁴ = viscous flux.

    σ T⁴ = F_visc(r) = 3GMṁ/(8πr³) * [1 − √(r_in/r)]

    Parameters
    ----------
    r        : float veya pint.Quantity
               Disk yarıçapı (m).
    M        : float veya pint.Quantity
               Kara delik kütlesi (kg).
    m_dot    : float veya pint.Quantity
               Kütle akış hızı (kg/s).
    alpha    : float, optional
               Viskozite parametresi (0 < α ≤ 1).
    a        : float, optional
               Spin parametresi (−1 ≤ a ≤ 1).
    initial_T: float veya pint.Quantity, optional
               Başlangıç tahmini (K). None ise effective_temperature kullanılır.
    max_iter : int, optional
               Maks. iterasyon sayısı.
    tol      : float, optional
               Tolerans (|ΔT| < tol koşulu).

    Returns
    -------
    Yüzey sıcaklığı T (K).

    Raises
    ------
    ValueError
        - r, M, m_dot veya initial_T pozitif değilse
        - 0 < α ≤ 1 veya |a| ≤ 1 değilse
        - r ≤ r_in
    RuntimeError
        - max_iter boyunca yakınsama sağlanamazsa
    """
    # Temel validasyon
    _validate_positive(r=r, M=M, m_dot=m_dot)
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0,1]. Got {alpha}.")
    if abs(a) > 1:
        raise ValueError(f"Spin |a| ≤ 1 olmalı. Got {a}.")

    # ISCO ve yarıçap kontrolü
    r_in = isco_radius(a, M)
    if r <= r_in:
        raise ValueError(f"r ({r}) > r_in ({r_in}) olmalı.")

    # Başlangıç sıcaklığı
    if initial_T is None:
        initial_T = effective_temperature(r, M, m_dot, r_in=r_in, a=a)
    _validate_positive(initial_T=initial_T)

    # Newton iterasyonu
    T = initial_T
    for i in range(max_iter):
        F_visc = (3 * GRAVITATIONAL_CONSTANT * M * m_dot /
                  (8 * np.pi * r**3) *
                  (1 - np.sqrt(r_in / r)))
        F_rad  = STEFAN_BOLTZMANN_CONSTANT * T**4

        # Rezidü ve türev
        res    = F_rad - F_visc
        dresdT = 4 * STEFAN_BOLTZMANN_CONSTANT * T**3

        delta  = res / dresdT
        T     -= delta

        if abs(delta) < tol:
            return T

    raise RuntimeError(
        f"iterative_temperature did not converge after {max_iter} iterations."
    )
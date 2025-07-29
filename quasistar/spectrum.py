"""
spectrum.py

Accretion disk spektrumu:
- Planck dağılım fonksiyonu (per Hz)
- Disk spektrumunun (L_nu, F_nu) hesaplanması

Tüm fonksiyonlar constants.py, temperature.py ve physics.py modüllerinden
sabitleri ve yardımcı fonksiyonları kullanır.
"""

import numpy as np
from pint import Quantity

from quasistar.constants import (
    PLANCK_CONSTANT,
    BOLTZMANN_CONSTANT,
    SPEED_OF_LIGHT,
)
from temperature import effective_temperature
from quasistar.physics import _validate_positive

__all__ = [
    "planck_nu",
    "disk_spectrum",
]


def planck_nu(
    nu: float | np.ndarray | Quantity,
    T: float | Quantity
) -> float | np.ndarray | Quantity:
    """
    Planck dağılımı per birim frekans: B_nu(T).

    B_nu = (2 h ν^3 / c^2) / (exp(h ν / k_B T) - 1)

    Parameters
    ----------
    nu : float, np.ndarray veya pint.Quantity
         Frekans (Hz).
    T  : float veya pint.Quantity
         Sıcaklık (K).

    Returns
    -------
    float, np.ndarray veya pint.Quantity
        Spectral radiance B_nu (W·sr⁻¹·m⁻²·Hz⁻¹).

    Raises
    ------
    ValueError
        Eğer nu veya T pozitif değilse.

    Note
    ----
    Uses _validate_positive from physics.py for input validation.
    Units are automatically converted with pint if available.

    Examples
    --------
    >>> from quasistar.constants import ureg
    >>> planck_nu(3e14 * ureg.Hz, 5500 * ureg.kelvin)
    """
    # Pozitiflik kontrolü
    _validate_positive(nu=nu, T=T)

    # Numerik değerler (units-preserving)
    num = 2 * PLANCK_CONSTANT * nu**3 / SPEED_OF_LIGHT**2
    expo = np.exp(PLANCK_CONSTANT * nu / (BOLTZMANN_CONSTANT * T)) - 1
    return num / expo


def disk_spectrum(
    nu: float | np.ndarray | Quantity,
    r: float | np.ndarray | Quantity,
    M: float | Quantity,
    m_dot: float | Quantity,
    r_out: float | Quantity = None,
    a: float = 0.0,
    inclination: float = 0.0,
    distance: float | Quantity = None
) -> Quantity:
    """
    Accretion disk spektrumu: L_nu veya gözlenen F_nu.

    L_nu = 2π cos(i) ∫_{r_in}^{r_out} B_nu[T_eff(r)] · r dr
    F_nu = L_nu / (4π D^2)   (eğer distance (D) verilmişse)

    Parameters
    ----------
    nu : float, np.ndarray veya pint.Quantity
         Frekans dizisi (Hz).
    r : float, np.ndarray veya pint.Quantity
        Yarıçap dizisi (m). Monotonik artan olmalı, r_in ile başlar.
    M : float veya pint.Quantity
        Kara delik kütlesi (kg).
    m_dot : float veya pint.Quantity
        Kütle akış hızı (kg/s).
    r_out : float veya pint.Quantity, optional
        Dış yarıçap (m). None ise r.max() kullanılır.
    a : float, optional
        Spin parametresi (−1 ≤ a ≤ 1).
    inclination : float, optional
        Disk eğim açısı (radyan). 0 → yüzeye dik.
    distance : float veya pint.Quantity, optional
        Uzaklık (m). Verilirse akı F_nu döner.

    Returns
    -------
    Quantity
        L_nu (W·Hz⁻¹) veya F_nu (W·m⁻²·Hz⁻¹) biriminde.

    Raises
    ------
    ValueError
        - M, m_dot veya distance pozitif değilse
        - |a| > 1
        - 0 ≤ inclination ≤ π değilse
        - r veya r_out pozitif değilse
        - r monotonik artmıyorsa

    Note
    ----
    Uses _validate_positive from physics.py for input validation.
    Units are automatically converted with pint if available.
    Ignores relativistic effects and Doppler shifts (to be added later).

    Examples
    --------
    >>> from quasistar.constants import SOLAR_MASS, ureg
    >>> r_vals = np.linspace(1e10, 1e11, 100) * ureg.meter
    >>> disk_spectrum(3e14 * ureg.Hz, r_vals, SOLAR_MASS, 1e20 * ureg.kg / ureg.s)
    """
    # Unit ve pozitiflik kontrolü
    _validate_positive(M=M, m_dot=m_dot)
    if distance is not None:
        _validate_positive(distance=distance)
    if not (-1.0 <= a <= 1.0):
        raise ValueError(f"Spin |a| ≤ 1 olmalı. Got {a}.")
    if not (0 <= inclination <= np.pi):
        raise ValueError(f"Inclination must be between 0 and π radians. Got {inclination}.")

    # Entegrasyon yarıçapları
    if r_out is None:
        r_out = np.max(r)
    _validate_positive(r=r, r_out=r_out)
    if np.any(r <= 0) or r_out <= 0:
        raise ValueError("r ve r_out pozitif olmalı.")
    if np.any(np.diff(r) < 0):
        raise ValueError("r must be monotonically increasing for integration.")

    # Eğim faktörü
    mu = np.cos(inclination)

    # Yüzey sıcaklığı
    T_eff = effective_temperature(r, M, m_dot, a=a)

    # Planck fonksiyonu değerleri (broadcast: nu[:, None] × r[None, :])
    B_nu = planck_nu(nu=np.atleast_1d(nu)[:, None], T=T_eff[None, :])

    # İntegrand: 2π r · B_nu · cos(i)
    integrand = 2 * np.pi * B_nu * r[None, :] * mu

    # Entegrasyon over r ekseni
    L_nu = np.trapz(integrand, x=r, axis=1)

    if distance is not None:
        # Gözlenen akı
        F_nu = L_nu / (4 * np.pi * distance**2)
        return F_nu
    return L_nu
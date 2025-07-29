"""
physics.py

Planck yasası, Wien yer değiştirme, ISCO yarıçapı,
gravitational redshift, Doppler faktörü hesaplayıcıları
ve ek fonksiyonlar: planck_frequency, bolometric_luminosity.

Tüm fonksiyonlar constants.py’daki sabitleri kullanır.
Uses _validate_positive helper for input validation.

Usage:
    >>> from quasistar.constants import SOLAR_MASS, ureg
    >>> isco_radius(0.5, SOLAR_MASS)
    >>> planck_frequency(3e14 * ureg.Hz, 5500 * ureg.kelvin)
    >>> bolometric_luminosity(1e20 * ureg.kg / ureg.s, eta=0.1)
"""

from quasistar.constants import (
    BOLTZMANN_CONSTANT,
    GRAVITATIONAL_CONSTANT,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    WIEN_DISPLACEMENT,
)
import numpy as np

__all__ = [
    "planck_lambda",
    "planck_frequency",
    "wien_wavelength_peak",
    "isco_radius",
    "gravitational_redshift",
    "doppler_factor",
    "bolometric_luminosity",
]


def _validate_positive(**kwargs):
    """
    Helper to ensure given keyword arguments are positive.
    Raises ValueError if any value <= 0.
    """
    for name, value in kwargs.items():
        if value is None or value <= 0:
            raise ValueError(f"{name} must be positive. Got {value}.")


def planck_lambda(wavelength, T):
    """
    Spektral radyans B_lambda(T) dalga boyuna göre.

    Parameters
    ----------
    wavelength : float or pint.Quantity
        Dalga boyu (m).
    T : float or pint.Quantity
        Sıcaklık (K).

    Returns
    -------
    float or pint.Quantity
        Spektral radyans (W·sr⁻¹·m⁻³).

    Raises
    ------
    ValueError
        Eğer wavelength veya T pozitif değilse.
    """
    _validate_positive(wavelength=wavelength, T=T)

    exponent = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (
        wavelength * BOLTZMANN_CONSTANT * T
    )
    numerator = 2 * PLANCK_CONSTANT * SPEED_OF_LIGHT**2
    denominator = wavelength**5 * (np.exp(exponent) - 1)

    return numerator / denominator


def planck_frequency(freq, T):
    """
    Spektral radyans B_nu(T) frekansa göre.

    Parameters
    ----------
    freq : float or pint.Quantity
        Frekans (Hz).
    T : float or pint.Quantity
        Sıcaklık (K).

    Returns
    -------
    float or pint.Quantity
        Spektral radyans (W·sr⁻¹·m⁻²·Hz⁻¹).

    Raises
    ------
    ValueError
        Eğer freq veya T pozitif değilse.

    Note
    ----
    Units are automatically converted with pint if available.
    """
    _validate_positive(freq=freq, T=T)

    exponent = (PLANCK_CONSTANT * freq) / (BOLTZMANN_CONSTANT * T)
    numerator = 2 * PLANCK_CONSTANT * freq**3 / SPEED_OF_LIGHT**2
    denominator = np.exp(exponent) - 1

    return numerator / denominator


def wien_wavelength_peak(T):
    """
    Wien yer değiştirme kanunu: λ_max = b / T.

    Parameters
    ----------
    T : float or pint.Quantity
        Sıcaklık (K).

    Returns
    -------
    float or pint.Quantity
        En yoğun dalga boyu (m).

    Raises
    ------
    ValueError
        Eğer T pozitif değilse.
    """
    _validate_positive(T=T)
    return WIEN_DISPLACEMENT / T


def isco_radius(a, M):
    """
    Karadeliğin spin parametresi a ve kütlesi M için ISCO yarıçapı.

    Parameters
    ----------
    a : float
        Boyutsuz spin parametresi (-1 <= a <= 1).
    M : float or pint.Quantity
        Karadelik kütlesi (kg).

    Returns
    -------
    float or pint.Quantity
        ISCO yarıçapı (m).

    Raises
    ------
    ValueError
        Eğer |a|>1 veya M pozitif değilse.
    """
    if abs(a) > 1:
        raise ValueError(f"Spin parameter must be within [-1,1]. Got {a}.")
    _validate_positive(M=M)

    r_g = GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2
    z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    z2 = np.sqrt(3 * a**2 + z1**2)
    sign_a = np.sign(a)
    term = np.sqrt((3 - z1) * (3 + z1 + 2 * z2))

    return r_g * (3 + z2 - sign_a * term)


def gravitational_redshift(r, M, a):
    """
    Simplified Kerr spacetime gravitational redshift factor.

    Note
    ----
    Bu fonksiyon tam geodesic entegrasyonu yerine basitleştirilmiş bir yaklaşımdır.

    Parameters
    ----------
    r : float or pint.Quantity
        Yarıçap (m).
    M : float or pint.Quantity
        Karadelik kütlesi (kg).
    a : float
        Boyutsuz spin parametresi (-1 <= a <= 1).

    Returns
    -------
    float or pint.Quantity
        Kırmızıya kayma faktörü.

    Raises
    ------
    ValueError
        Eğer r veya M pozitif değilse, ya da |a|>1.
    """
    _validate_positive(r=r, M=M)
    if abs(a) > 1:
        raise ValueError(f"Spin parameter must be within [-1,1]. Got {a}.")

    r_g = GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2
    term1 = 1 - 2 * r_g / r
    term2 = np.sqrt(1 - a**2 * (1 - r_g / r)**2)

    return np.sqrt(term1 / (1 + term2))


def doppler_factor(r, M, a, inclination=0.0):
    """
    Simplified relativistic Doppler factor for Kerr spacetime.

    Parameters
    ----------
    r : float or pint.Quantity
        Yarıçap (m).
    M : float or pint.Quantity
        Karadelik kütlesi (kg).
    a : float
        Boyutsuz spin parametresi (-1 <= a <= 1).
    inclination : float, optional
        Gözlemci eğim açısı (radyan, 0=face-on).

    Returns
    -------
    float or pint.Quantity
        Doppler faktörü.

    Raises
    ------
    ValueError
        Eğer r veya M pozitif değilse, |a|>1,
        veya inclination 0 ile π arasında değilse.
    """
    _validate_positive(r=r, M=M)
    if abs(a) > 1:
        raise ValueError(f"Spin parameter must be within [-1,1]. Got {a}.")
    if not (0 <= inclination <= np.pi):
        raise ValueError(
            f"Inclination must be between 0 and π radians. Got {inclination}."
        )

    v = np.sqrt(GRAVITATIONAL_CONSTANT * M / r)
    beta = v / SPEED_OF_LIGHT
    gamma = 1.0 / np.sqrt(1 - beta**2)
    mu = np.cos(inclination)

    return 1.0 / (gamma * (1 - beta * mu))


def bolometric_luminosity(m_dot, eta=0.1):
    """
    Bolometric luminosity L = η * ṁ * c^2.

    Parameters
    ----------
    m_dot : float or pint.Quantity
        Kütle akı hızı (kg/s).
    eta : float, optional
        Verimlilik faktörü (default=0.1).

    Returns
    -------
    float or pint.Quantity
        Bolometrik luminosite (W).

    Raises
    ------
    ValueError
        Eğer m_dot pozitif değilse veya eta 0 ile 1 arasında değilse.

    Note
    ----
    Units are automatically converted with pint if available.
    """
    _validate_positive(m_dot=m_dot)
    if not 0 <= eta <= 1:
        raise ValueError(f"Efficiency eta must be between 0 and 1. Got {eta}.")
    c_squared = SPEED_OF_LIGHT**2
    return eta * m_dot * c_squared
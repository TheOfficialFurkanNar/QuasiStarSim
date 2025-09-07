"""
physics.py

Planck yasası, Wien yer değiştirme, ISCO yarıçapı,
gravitational redshift, Doppler faktörü hesaplayıcıları
ve ek fonksiyonlar: planck_frequency, bolometric_luminosity, planck, optical_depth.

Tüm fonksiyonlar constants.py’daki sabitleri kullanır.
Uses _validate_positive helper for input validation.
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
    "planck",
    "optical_depth",
]


def _validate_positive(**kwargs):
    """
    Helper to ensure given keyword arguments are positive.
    Supports both scalars and NumPy arrays.
    """
    for name, value in kwargs.items():
        if value is None:
            raise ValueError(f"{name} must be positive. Got None.")
        arr = np.atleast_1d(value)
        if np.any(arr <= 0):
            raise ValueError(f"{name} must be positive. Got {value}.")


def planck_lambda(wavelength, T):
    """Spektral radyans B_lambda(T) dalga boyuna göre."""
    _validate_positive(wavelength=wavelength, T=T)

    # Pint.Quantity ise uygun birime çevir ve magnitude al
    if hasattr(wavelength, "to"):
        wavelength = wavelength.to("meter").magnitude
    if hasattr(T, "to"):
        T = T.to("kelvin").magnitude

    exponent = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (
        wavelength * BOLTZMANN_CONSTANT * T
    )
    if hasattr(exponent, "to"):
        exponent = exponent.to("").magnitude  # boyutsuzlaştır

    numerator = 2 * PLANCK_CONSTANT * SPEED_OF_LIGHT**2
    if hasattr(numerator, "to"):
        numerator = numerator.to("watt * meter**3 / steradian").magnitude

    denominator = wavelength**5 * (np.exp(exponent) - 1)
    return numerator / denominator


def planck_frequency(freq, T):
    """Spektral radyans B_nu(T) frekansa göre."""
    _validate_positive(freq=freq, T=T)

    if hasattr(freq, "to"):
        freq = freq.to("hertz").magnitude
    if hasattr(T, "to"):
        T = T.to("kelvin").magnitude

    exponent = (PLANCK_CONSTANT * freq) / (BOLTZMANN_CONSTANT * T)
    numerator = 2 * PLANCK_CONSTANT * freq**3 / SPEED_OF_LIGHT**2
    denominator = np.exp(exponent) - 1
    return numerator / denominator


def planck(wavelength, T):
    """
    Planck yasası: B_lambda(T) dalga boyuna göre (kısa isim).
    QuasiStarDiskSimulator gibi modüllerde doğrudan kullanılabilir.
    """
    return planck_lambda(wavelength, T)


def optical_depth(wavelength, r, tau0=0.01, ref_wavelength=1e-9, ref_radius=None):
    """
    Dalga boyu ve yarıçap bağımlı optik derinlik.
    """
    _validate_positive(wavelength=wavelength, r=r)
    if ref_radius is None:
        ref_radius = r
    _validate_positive(ref_radius=ref_radius)
    return tau0 * (wavelength / ref_wavelength)**(-2) * (r / ref_radius)**(-1)


def wien_wavelength_peak(T):
    """Wien yer değiştirme kanunu: λ_max = b / T."""
    _validate_positive(T=T)
    return WIEN_DISPLACEMENT / T


def isco_radius(a, M):
    """Karadeliğin spin parametresi a ve kütlesi M için ISCO yarıçapı."""
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
    """Simplified Kerr spacetime gravitational redshift factor."""
    _validate_positive(r=r, M=M)
    if abs(a) > 1:
        raise ValueError(f"Spin parameter must be within [-1,1]. Got {a}.")
    r_g = GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2
    term1 = 1 - 2 * r_g / r
    term2 = np.sqrt(1 - a**2 * (1 - r_g / r)**2)
    return np.sqrt(term1 / (1 + term2))


def doppler_factor(r, M, a, inclination=0.0):
    """Simplified relativistic Doppler factor for Kerr spacetime."""
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
    """Bolometric luminosity L = η * ṁ * c^2."""
    _validate_positive(m_dot=m_dot)
    if not 0 <= eta <= 1:
        raise ValueError(f"Efficiency eta must be between 0 and 1. Got {eta}.")
    c_squared = SPEED_OF_LIGHT**2
    return eta * m_dot * c_squared
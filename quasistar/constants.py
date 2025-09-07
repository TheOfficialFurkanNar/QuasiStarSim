"""
constants.py

Merkezi fiziksel sabitler ve (opsiyonel) birim yönetimi.
Tüm QuasiStarSim modülleri bu dosyadan import eder.

Kaynaklar:
- CODATA 2018
- NIST Fundamental Physical Constants

Usage:
    from constants import GRAVITATIONAL_CONSTANT as G
    print(G)  # 6.67430e-11 m³/kg/s² or Quantity with pint
    Note: Enables unit-aware plotting with matplotlib if pint is available.
"""

__version__ = "0.1.1"

try:
    from pint import UnitRegistry
except ImportError:
    UnitRegistry = None

if UnitRegistry:
    ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
    ureg.setup_matplotlib()  # Enable unit-aware plotting
    Q_ = ureg.Quantity

    # 🔹 Ek birim tanımlamaları
    # Güneş kütlesi (M☉)
    ureg.define("solar_mass = 1.98847e30 * kilogram = M_sun")
    # Güneş yarıçapı (opsiyonel, istersen eklersin)
    ureg.define("solar_radius = 6.957e8 * meter = R_sun")

else:
    ureg = None

    def Q_(value, unit):
        return value

    import warnings
    warnings.warn(
        "pint kütüphanesi yüklenemedi. Sabitler ham float olarak tanımlanacak.",
        ImportWarning,
        stacklevel=2
    )

# Temel sabitler
SPEED_OF_LIGHT            = Q_(2.99792458e8, "meter/second")
PLANCK_CONSTANT           = Q_(6.62607015e-34, "joule*second")
BOLTZMANN_CONSTANT        = Q_(1.380649e-23, "joule/kelvin")
GRAVITATIONAL_CONSTANT    = Q_(6.67430e-11, "meter**3 / kilogram / second**2")
STEFAN_BOLTZMANN_CONSTANT = Q_(5.670374419e-8, "watt / meter**2 / kelvin**4")
WIEN_DISPLACEMENT         = Q_(2.897771955e-3, "meter*kelvin")
ELECTRON_CHARGE           = Q_(1.602176634e-19, "coulomb")
VACUUM_PERMITTIVITY       = Q_(8.854187817e-12, "farad/meter")
GAS_CONSTANT              = Q_(8.314462618, "joule / mole / kelvin")

# Astronomik sabitler
SOLAR_MASS                = Q_(1.98847e30, "kilogram")
SOLAR_RADIUS              = Q_(6.957e8, "meter")

# Parçacık kütleleri
ELECTRON_MASS             = Q_(9.10938356e-31, "kilogram")
PROTON_MASS               = Q_(1.6726219e-27, "kilogram")
AVOGADRO_CONSTANT         = Q_(6.02214076e23, "1/mole")

# Matematiksel sabit
PI = ureg.pi if ureg else 3.141592653589793

__all__ = [
    "__version__",
    "ureg", "Q_",
    # Temel sabitler
    "SPEED_OF_LIGHT",
    "PLANCK_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "GRAVITATIONAL_CONSTANT",
    "STEFAN_BOLTZMANN_CONSTANT",
    "WIEN_DISPLACEMENT",
    "ELECTRON_CHARGE",
    "VACUUM_PERMITTIVITY",
    "GAS_CONSTANT",
    # Astronomik sabitler
    "SOLAR_MASS",
    "SOLAR_RADIUS",
    # Parçacık kütleleri
    "ELECTRON_MASS",
    "PROTON_MASS",
    "AVOGADRO_CONSTANT",
    # Matematiksel sabit
    "PI",
]

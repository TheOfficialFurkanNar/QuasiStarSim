# tests/test_density.py

import math
import pytest
import density

# 1. Kütle/ Hacim Temelli Yoğunluk
def test_calculate_density_normal():
    mass = 10.0
    volume = 2.0
    expected = 5.0
    assert density.calculate_density(mass, volume) == expected

def test_calculate_density_zero_volume():
    with pytest.raises(ValueError):
        density.calculate_density(10.0, 0.0)

# 2. Atmosfer Yoğunluğu (Örnek: Üstel Model)
@pytest.mark.parametrize("altitude, surface, scale, expected", [
    (0, 1.225, 8500.0, 1.225),                              # Deniz seviyesi
    (1000, 1.225, 8500.0, 1.225 * math.exp(-1000/8500.0)),  # 1 km yükseklik
])
def test_atmospheric_density_default(altitude, surface, scale, expected):
    # density.atmospheric_density(altitude, surface_density, scale_height)
    result = density.atmospheric_density(altitude, surface_density=surface, scale_height=scale)
    assert pytest.approx(expected, rel=1e-6) == result

# 3. Özel Akışkan Yoğunluğu (Örnek: İdeal Gaz Yasası)
def test_gas_density_ideal():
    pressure = 101325      # Pa
    molar_mass = 0.029     # kg/mol (hava)
    temperature = 288.15   # K (15 °C)
    R = 8.3145             # J/(mol·K)
    expected = (pressure * molar_mass) / (R * temperature)
    assert pytest.approx(expected, rel=1e-6) == density.gas_density(pressure, molar_mass, temperature)

# 4. Hatalı Girdi Kontrolleri
@pytest.mark.parametrize("mass, volume", [
    (-1.0, 1.0),  # Negatif kütle
    (1.0, -1.0),  # Negatif hacim
])
def test_invalid_inputs_mass_volume(mass, volume):
    with pytest.raises(ValueError):
        density.calculate_density(mass, volume)

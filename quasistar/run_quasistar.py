"""
run_quasistar.py

QuasiStarSim: Kara delik akresyon diski simülasyonu.
Tam birim farkındalığı (pint) ile çalışır.

Özellikler:
- Wien tepe dalga boyu hesaplama
- Akresyon diski sıcaklık profili
- Planck spektrumları (birim farkındalıklı)
- Radyal grid üzerinden spektrum animasyonu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from quasistar.constants import (
    GRAVITATIONAL_CONSTANT as G,
    SPEED_OF_LIGHT as c,
    STEFAN_BOLTZMANN_CONSTANT as sigma,
    WIEN_DISPLACEMENT as b,
    Q_,
    ureg,
)
from quasistar.physics import (
    planck,
    isco_radius,
    gravitational_redshift,
    doppler_factor,
    optical_depth,
)
from quasistar.config_logging import get_logger

logger = get_logger(__name__)


def wien_peak_lambda(T):
    """Wien yasasına göre tepe dalga boyu (λ_max)."""
    return b / T


def _to_quantity_meter(x):
    """float veya Quantity'yi metre cinsinden Quantity yapar."""
    if x is None:
        return None
    if hasattr(x, "to"):
        return x.to("meter")
    return Q_(x, "meter")


def simulate_accretion_disk(
    M_solar,
    mdot,
    rin=None,
    rout=None,
    eta=0.1,
    tau0=0.01,
    steps=500,
    time_s=0,
    animate=False,
    a=0.0,
    inclination=0.0,
):
    """
    Akresyon diski simülasyonu.

    Args:
        M_solar (float): Kara delik kütlesi [güneş kütlesi].
        mdot (float): Kütle akış hızı [kg/s].
        rin (float|Quantity, optional): İç yarıçap (m).
        rout (float|Quantity, optional): Dış yarıçap (m).
        eta (float): Radyatif verimlilik.
        tau0 (float): Optik derinlik normalizasyonu.
        steps (int): Radyal grid adım sayısı.
        time_s (float): Başlangıç zamanı (s).
        animate (bool): Spektrum animasyonu çizilsin mi?
        a (float): Spin parametresi.
        inclination (float): Yörünge eğimi (radyan).

    Returns:
        dict: {r, T, L, wavelengths, wavelengths_q}
    """
    logger.info("Akresyon diski simülasyonu başlatılıyor...")

    # Kütle ve mdot'u Quantity yap
    M = Q_(M_solar, "solar_mass").to("kg")
    mdot_q = Q_(mdot, "kg/s")

    # Yarıçapları güvenli şekilde Quantity(m) yap
    rin_q = _to_quantity_meter(rin)
    rout_q = _to_quantity_meter(rout)

    # Varsayılan yarıçaplar
    if rin_q is None:
        rin_q = isco_radius(a, M).to("meter")
    if rout_q is None:
        rout_q = 100 * rin_q

    # Radyal grid (Quantity, m)
    r = Q_(np.linspace(rin_q.magnitude, rout_q.magnitude, steps), "meter")

    # Disk sıcaklığı (Kelvin Quantity)
    T_expr = ((3 * G * M * mdot_q) / (8 * np.pi * sigma * r**3)
              * (1 - np.sqrt(rin_q / r)))**0.25
    T = T_expr.to("kelvin")

    # Dalga boyu aralığı (1 nm – 3000 nm)
    wavelengths_q = Q_(np.linspace(1e-9, 3e-6, 500), "meter")
    wavelengths = wavelengths_q.magnitude  # optical_depth için float

    def spectrum(r_idx):
        """Belirli yarıçap için spektrum (I_λ)."""
        r_i = r[r_idx]
        T_i = T[r_idx]
        z = gravitational_redshift(r_i, M, a)
        dpl = doppler_factor(r_i, M, a, inclination=inclination)
        tau = optical_depth(wavelengths, r_i, tau0, ref_radius=rin_q)
        wl_obs = wavelengths_q / (z * dpl)
        return planck(wl_obs, T_i) * np.exp(-tau)

    # Toplam parlaklık (L ~ η * Ṁ * c²)
    L = (eta * mdot_q * c**2).to("watt")

    if animate:
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(wavelengths.min() * 1e9, wavelengths.max() * 1e9)
        ax.set_ylim(0, spectrum(int(steps / 2)).max().magnitude * 1.1)
        ax.set_xlabel("λ (nm)")
        ax.set_ylabel("I (W·m⁻³·sr⁻¹)")
        ax.set_title("Disk Spektrumu Zamanla Değişimi")

        def init():
            line.set_data([], [])
            return (line,)

        def update(frame):
            I = spectrum(frame)
            line.set_data(wavelengths * 1e9, I.magnitude)
            ax.set_title(
                f"r = {int(r[frame].to('km').magnitude)} km, "
                f"t = {int(time_s+frame)} s"
            )
            return (line,)

        animation.FuncAnimation(
            fig, update, frames=steps, init_func=init,
            blit=True, interval=50
        )
        plt.show()
    else:
        for idx in (int(steps * 0.2), int(steps * 0.5), int(steps * 0.8)):
            I = spectrum(idx)
            plt.plot(
                wavelengths * 1e9, I.magnitude,
                label=f"r={int(r[idx].to('km').magnitude)} km, "
                      f"T={T[idx].magnitude:.0f} K"
            )
        plt.xlabel("λ (nm)")
        plt.ylabel("I (W·m⁻³·sr⁻¹)")
        plt.title("Akresyon Disk Spektrumu")
        plt.legend()
        plt.grid(True)
        plt.show()

    logger.info(f"Toplam radyasyon gücü L: {L:.2e}")
    return {
        "r": r,
        "T": T,
        "L": L,
        "wavelengths": wavelengths,
        "wavelengths_q": wavelengths_q
    }


if __name__ == "__main__":
    from quasistar.__main__ import cli
    cli()



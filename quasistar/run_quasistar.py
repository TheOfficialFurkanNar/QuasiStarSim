# quasistar/run_quasistar.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Paket içinden gerekli sabitler ve fonksiyonlar
from quasistar.constants import (
    GRAVITATIONAL_CONSTANT as G,
    SPEED_OF_LIGHT as c,
    STEFAN_BOLTZMANN_CONSTANT as sigma,
    PLANCK_CONSTANT as h,
    BOLTZMANN_CONSTANT as k,
    WIEN_DISPLACEMENT as b
)
from quasistar.physics import (
    planck,
    isco_radius,
    gravitational_redshift,
    doppler_factor,
    optical_depth
)
from quasistar.config_logging import get_logger

logger = get_logger(__name__)


def wien_peak_lambda(T):
    """Wien yer değiştirme kanunu: λ_max = b / T"""
    return b / T


def simulate_accretion_disk(
    M_solar, mdot, rin=None, rout=None,
    eta=0.1, tau0=0.01, steps=500,
    time_s=0, animate=False,
    a=0.0, inclination=0.0
):
    """
    Basit akresyon diski simülasyonu.

    Parameters
    ----------
    M_solar : float
        Kara delik kütlesi (Güneş kütlesi)
    mdot : float
        Kütle akış hızı (kg/s)
    rin, rout : float, optional
        İç ve dış yarıçap (m)
    eta : float
        Verimlilik faktörü
    tau0 : float
        Optik derinlik katsayısı
    steps : int
        Radyal adım sayısı
    time_s : int
        Başlangıç zamanı (s)
    animate : bool
        True ise animasyon modunda çalışır
    a : float
        Spin parametresi (-1 <= a <= 1)
    inclination : float
        Gözlemci eğim açısı (radyan)
    """
    logger.info("Akresyon diski simülasyonu başlatılıyor...")

    M = M_solar * 1.989e30
    if rin is None:
        rin = isco_radius(a, M)
    if rout is None:
        rout = 100 * rin
    r = np.linspace(rin, rout, steps)

    # Sıcaklık profili (basit α-disk formülü)
    T = ((3 * G * M * mdot) / (8 * np.pi * sigma * r**3)
         * (1 - np.sqrt(rin / r)))**0.25

    wavelengths = np.linspace(1e-9, 3e-6, 500)

    def spectrum(r_idx):
        r_i = r[r_idx]
        T_i = T[r_idx]
        z   = gravitational_redshift(r_i, M, a)
        dpl = doppler_factor(r_i, M, a, inclination=inclination)
        tau = optical_depth(wavelengths, r_i, tau0, ref_radius=rin)
        wl_obs = wavelengths / z / dpl
        I = planck(wl_obs, T_i) * np.exp(-tau)
        return I

    # Toplam ışıma gücü
    L = eta * mdot * c**2

    if animate:
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(wavelengths.min()*1e9, wavelengths.max()*1e9)
        ax.set_ylim(0, spectrum(int(steps/2)).max()*1.1)
        ax.set_xlabel("λ (nm)")
        ax.set_ylabel("I (W·m⁻³·sr⁻¹)")
        ax.set_title("Disk Spektrumu Zamanla Değişimi")

        def init():
            line.set_data([], [])
            return (line,)

        def update(frame):
            I = spectrum(frame)
            line.set_data(wavelengths*1e9, I)
            ax.set_title(f"r = {int(r[frame]/1e3)} km, t = {int(time_s+frame)} s")
            return (line,)

        animation.FuncAnimation(
            fig, update, frames=steps, init_func=init,
            blit=True, interval=50
        )
        plt.show()
    else:
        for idx in (int(steps*0.2), int(steps*0.5), int(steps*0.8)):
            I = spectrum(idx)
            plt.plot(wavelengths*1e9, I,
                     label=f"r={int(r[idx]/1e3)} km, T={int(T[idx])} K")
        plt.xlabel("λ (nm)")
        plt.ylabel("I (W·m⁻³·sr⁻¹)")
        plt.title("Akresyon Disk Spektrumu")
        plt.legend()
        plt.grid(True)
        plt.show()

    logger.info(f"Toplam radyasyon gücü L: {L:.2e} W")
    return {"r": r, "T": T, "L": L, "wavelengths": wavelengths}


# --- CLI entegrasyonu ---
if __name__ == "__main__":
    from quasistar.__main__ import cli
    cli()
# QuasiStarDiskSimulator.py

import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
from matplotlib import cm, animation

# Evrensel sabitler
G     = 6.674e-11        # m^3 kg^-1 s^-2
c     = 3.0e8            # m/s
sigma = 5.67e-8          # W m^-2 K^-4
h     = 6.626e-34        # J s
k     = 1.381e-23        # J/K
b     = 2.8977719e-3     # m·K (Wien sabiti)

# ------------------------- #
# --- Fiziksel Fonksiyonlar #
# ------------------------- #

def planck(wl, T):
    exp_arg = h * c / (wl * k * T)
    return (2 * h * c**2 / wl**5) / (np.exp(exp_arg) - 1)

def wien_peak_lambda(T):
    return b / T

def isco_radius(M):
    """İnnermost Stable Circular Orbit (Schwarzschild) [m]."""
    return 6 * G * M / c**2

def gravitational_redshift(r, M):
    rs = 2 * G * M / c**2
    return np.sqrt(1 - rs / r)

def doppler_factor(r, rin):
    """Basitleştirilmiş Doppler etkisi (yaklaşık)."""
    v = np.sqrt(G * M / r)
    beta = v / c
    return np.sqrt((1 + beta) / (1 - beta))

def optical_depth(wl, r, tau0=0.01):
    """Dalga boyu ve yarıçap bağımlı optik derinlik."""
    return tau0 * (wl / 1e-9)**(-2) * (r / rin)**(-1)

# ---------------------------- #
# --- Akresyon Simülasyonu --- #
# ---------------------------- #

def simulate_accretion_disk(
    M_solar, mdot, rin=None, rout=None,
    eta=0.1, tau0=0.01, steps=500,
    time_s=0, animate=False
):
    M    = M_solar * 1.989e30
    if rin is None: rin = isco_radius(M)
    if rout is None: rout = 100 * rin
    r    = np.linspace(rin, rout, steps)

    T = ((3 * G * M * mdot) / (8 * np.pi * sigma * r**3)
         * (1 - np.sqrt(rin / r)))**0.25

    wavelengths = np.linspace(1e-9, 3e-6, 500)

    def spectrum(r_idx):
        r_i = r[r_idx]
        T_i = T[r_idx]
        z   = gravitational_redshift(r_i, M)
        dpl = doppler_factor(r_i, rin)
        tau = optical_depth(wavelengths, r_i, tau0)
        wl_obs = wavelengths / z / dpl
        I = planck(wl_obs, T_i) * np.exp(-tau)
        return I

    # Toplam ışıma gücü
    L = eta * mdot * c**2

    if animate:
        fig, ax = plt.subplots(figsize=(8,5))
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

        anim = animation.FuncAnimation(
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

    print(f"Toplam radyasyon gücü L: {L:.2e} W")
    return {"r": r, "T": T, "L": L, "wavelengths": wavelengths}
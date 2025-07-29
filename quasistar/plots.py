# plots.py

import os
from typing import Dict, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .logging_config import get_logger

logger = get_logger(__name__)


def animate_disk(
    field: np.ndarray,
    sim_id: str,
    output_dir: str = "results",
    save: bool = True,
    cmap: str = "inferno",
    interval: int = 50,
    fps: int = 20
) -> animation.FuncAnimation:
    """
    Create an animation of the disk radial profile (density or temperature).

    Parameters
    ----------
    field : np.ndarray
        1D array (static profile) or 2D array (n_frames x n_bins).
    sim_id : str
        Unique identifier used in output filenames.
    output_dir : str, default "results"
        Directory for saving outputs.
    save : bool, default True
        If True, save animation as MP4.
    cmap : str, default "inferno"
        Colormap for line (color gradient over time).
    interval : int, default 50
        Delay between frames (in milliseconds).
    fps : int, default 20
        Frames per second for saved video.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    """
    # 1) Girdi validasyonu
    if not isinstance(field, np.ndarray) or field.ndim not in (1, 2):
        logger.error(f"animate_disk: `field` must be 1D or 2D numpy array, got {field.shape}")
        raise ValueError("`field` must be a 1D or 2D numpy array")

    # 2) Veri hazırlığı
    data = field if field.ndim == 2 else field[np.newaxis, :]
    n_frames, n_bins = data.shape

    # 3) Figür & eksen ayarları
    fig, ax = plt.subplots(figsize=(6, 4))
    # Zamanla renk değişimi için colormap index
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_frames))
    line, = ax.plot(data[0], color=colors[0])
    ax.set(
        title="Disk Profile Animation",
        xlabel="Radial Bin",
        ylabel="Value"
    )

    # 4) Kare güncelleme fonksiyonu
    def update(frame: int):
        line.set_ydata(data[frame])
        line.set_color(colors[frame])
        return (line,)

    # 5) Animasyon objesi
    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval,
        blit=True
    )
    logger.info(f"animate_disk: Created animation with {n_frames} frames")

    # 6) Kaydetme
    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{sim_id}_disk_animation.mp4")
        ani.save(path, fps=fps, dpi=200, codec="h264")
        logger.info(f"animate_disk: Animation saved to {path}")

    return ani


def plot_spectrum(
    spectrum: Dict[str, np.ndarray],
    sim_id: str,
    output_dir: str = "results",
    save: bool = True,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot the radiation spectrum (frequency vs. flux).

    Parameters
    ----------
    spectrum : dict
        Must contain keys "frequency" and "flux" with numpy arrays.
    sim_id : str
        Unique identifier used in output filenames.
    output_dir : str, default "results"
        Directory for saving outputs.
    save : bool, default True
        If True, save figure as PNG.
    log_scale : bool, default True
        If True, set both axes to log scale.
    figsize : tuple, default (8, 6)
        Figure dimensions.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    freq = spectrum.get("frequency")
    flux = spectrum.get("flux")

    if freq is None or flux is None:
        logger.error("plot_spectrum: spectrum dict must contain 'frequency' and 'flux'")
        raise KeyError("Spectrum dict missing 'frequency' or 'flux'")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, flux, label="Spectrum", color="crimson")

    ax.set(
        title=f"QuasiStarSim Spectrum ({sim_id})",
        xlabel="Frequency [Hz]",
        ylabel="Flux [erg/s/cm²/Hz]"
    )
    ax.grid(True, linestyle="--", alpha=0.5)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend(loc="best")
    logger.info("plot_spectrum: Spectrum plotted")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{sim_id}_spectrum.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"plot_spectrum: Figure saved to {path}")

    return fig
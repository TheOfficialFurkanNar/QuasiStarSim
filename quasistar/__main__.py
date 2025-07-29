# __main__.py

import os
import sys
import json
import pickle
import click
import yaml
import numpy as np

from config_logging import get_logger
from io_utils import save_results, load_results
from plots import animate_disk, plot_spectrum

logger = get_logger(__name__)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--log-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="YAML ile özel logging yapılandırması."
)
def cli(log_config):
    """
    PioneerAI Komut Satırı Arayüzü.

    Alt komutlar:
      animate   Disk profili animasyonu oluşturur
      spectrum  Işınım spektrumu çizer
      save      Sonuçları farklı formatlarda kaydeder
      load      Kaydedilmiş sonuçları yükler ve özetler
    """
    # Eğer kullanıcı özel bir YAML config sağlamışsa onu yükle
    if log_config:
        try:
            with open(log_config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            import logging.config
            logging.config.dictConfig(cfg)
            logger.info(f"Using custom logging config: {log_config}")
        except Exception as e:
            logger.error("Failed to load provided log-config, falling back to defaults.", exc_info=e)


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--sim-id",     required=True, help="Simülasyon kimliği, çıktı dosyası isminde kullanılır.")
@click.option("--output-dir", default="results", help="Çıktı klasörü.")
@click.option("--cmap",       default="inferno", help="Matplotlib renk haritası.")
@click.option("--interval",   default=50,       help="Kareler arası milisaniye.")
@click.option("--fps",        default=20,       help="Animasyon fps değeri.")
def animate(input, sim_id, output_dir, cmap, interval, fps):
    """
    Disk profili animasyonu oluşturur.

    INPUT olarak .npz ya da .npy uzantılı dosya bekler,
    içinde 'field' anahtarı veya tek boyutlu array olmalı.
    """
    logger.info(f"Loading field data from {input}")
    data = np.load(input, allow_pickle=True)
    field = data["field"] if "field" in data else data[:]
    animate_disk(
        field=field,
        sim_id=sim_id,
        output_dir=output_dir,
        save=True,
        cmap=cmap,
        interval=interval,
        fps=fps
    )


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--sim-id",     required=True, help="Simülasyon kimliği, çıktı dosyası isminde kullanılır.")
@click.option("--output-dir", default="results", help="Çıktı klasörü.")
@click.option(
    "--log-scale/--no-log-scale",
    default=True,
    help="Frekans ve flux eksenlerini logaritmik yapar."
)
@click.option(
    "--figsize",
    nargs=2,
    type=float,
    default=(8.0, 6.0),
    help="Grafik boyutu: genişlik yükseklik."
)
def spectrum(input, sim_id, output_dir, log_scale, figsize):
    """
    Işınım spektrumu çizer.

    INPUT olarak .npz ya da .npy dosya bekler,
    içinde 'frequency' ve 'flux' array’leri olmalı.
    """
    logger.info(f"Loading spectrum data from {input}")
    data = np.load(input, allow_pickle=True)
    spec = {
        "frequency": data["frequency"],
        "flux":      data["flux"]
    }
    plot_spectrum(
        spectrum=spec,
        sim_id=sim_id,
        output_dir=output_dir,
        save=True,
        log_scale=log_scale,
        figsize=tuple(figsize)
    )


@cli.command()
@click.argument("input-file", type=click.Path(exists=True, dir_okay=False))
@click.option("--sim-id",     required=True, help="Simülasyon kimliği.")
@click.option("--output-dir", default="results", help="Çıktı klasörü.")
@click.option(
    "--fmt",
    default="json",
    type=click.Choice(["json", "yaml", "npz", "pickle", "hdf5"], case_sensitive=False),
    help="Kaydetme formatı."
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Varolan dosyanın üzerine yazılsın mı?"
)
@click.option(
    "--version",
    default=None,
    help="İsteğe bağlı versiyon etiketi (default: UTC timestamp)."
)
def save(input_file, sim_id, output_dir, fmt, overwrite, version):
    """
    Sonuçları JSON/YAML/NPZ/Pickle/HDF5 formatında kaydeder.

    INPUT-FILE; JSON, YAML, NPZ, Pickle veya HDF5 olabilir.
    """
    ext = os.path.splitext(input_file)[1].lower().lstrip(".")
    # Girdi tipi neyse önce onu yükle
    if ext in ("json", "yaml"):
        loader = yaml.safe_load if ext == "yaml" else json.load
        with open(input_file, "r", encoding="utf-8") as f:
            raw = loader(f)
        data = {k: np.array(v) if isinstance(v, list) else v for k, v in raw.items()}

    elif ext == "npz":
        arr = np.load(input_file, allow_pickle=True)
        data = {k: arr[k] for k in arr.files}

    elif ext == "pickle":
        with open(input_file, "rb") as f:
            data = pickle.load(f)

    elif ext in ("hdf5", "h5"):
        import h5py  # optional dependency
        with h5py.File(input_file, "r") as hf:
            data = {k: hf[k][()] for k in hf.keys()}

    else:
        logger.error(f"Unsupported input ext: {ext}")
        sys.exit(1)

    save_results(
        data=data,
        sim_id=sim_id,
        output_dir=output_dir,
        fmt=fmt,
        overwrite=overwrite,
        version=version
    )


@cli.command()
@click.option("--sim-id",     required=True, help="Simülasyon kimliği.")
@click.option("--output-dir", default="results", help="Sonuçların bulunduğu klasör.")
@click.option(
    "--fmt",
    default=None,
    type=click.Choice(["json", "yaml", "npz", "pickle", "hdf5"], case_sensitive=False),
    help="Zorunlu format (default: klasördeki ilk eşleşme)."
)
def load(sim_id, output_dir, fmt):
    """
    Daha önce kaydedilmiş sonuçları yükler ve özetler.
    """
    data = load_results(sim_id=sim_id, output_dir=output_dir, fmt=fmt)
    click.echo(f"\nLoaded '{sim_id}' results from {output_dir}:\n")
    for key, val in data.items():
        shape = getattr(val, "shape", "-")
        click.echo(f"  • {key}: type={type(val).__name__}, shape={shape}")
    click.echo("")


if __name__ == "__main__":
    cli()
# simulation.py

import os
from datetime import datetime
from typing import Dict, Any

from tqdm import tqdm
from quasistar.density import calculate_density_profile
from temperature import solve_temperature_distribution
from spectrum import generate_spectrum
from quasistar.io_utils import save_results
from quasistar.plots import plot_spectrum, animate_disk
from .logging_config import get_logger

VALID_PARAMS = {
    "mass": float,
    "inner_radius": float,
    "outer_radius": float,
    "alpha": float,
    "num_radial_bins": int,
    "inclination": float,
    "output_dir": str,
    "n_jobs": int
}


def validate_params(params: Dict[str, Any]) -> None:
    missing = [k for k in VALID_PARAMS if k not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}")

    for key, expected in VALID_PARAMS.items():
        if not isinstance(params[key], expected):
            raise TypeError(f"Parameter '{key}' must be {expected.__name__}, got {type(params[key]).__name__}")

    if params["n_jobs"] < 1:
        raise ValueError("n_jobs must be >= 1")

    if params["inner_radius"] <= 0 or params["outer_radius"] <= params["inner_radius"]:
        raise ValueError("Require 0 < inner_radius < outer_radius")

    if not (0 < params["alpha"] <= 1):
        raise ValueError("alpha must be in (0, 1]")

    if not (0 <= params["inclination"] <= 90):
        raise ValueError("inclination must be between 0° and 90°")


def run_simulation(
    params: Dict[str, Any],
    save: bool = True,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    High-level simulation runner:
      1. validate_params
      2. calculate_density_profile (parallel + tqdm)
      3. solve_temperature_distribution (parallel + tqdm)
      4. generate_spectrum
      5. save & visualize (optional)
    """
    logger = get_logger("simulation")
    logger.info("Starting simulation")
    validate_params(params)

    # Create unique ID & timestamp
    timestamp = datetime.now().isoformat()
    sim_id = "sim_" + timestamp.replace(":", "").replace("-", "").split(".")[0]

    results: Dict[str, Any] = {}
    steps = [
        ("Density", lambda: calculate_density_profile(params)),
        ("Temperature", lambda: solve_temperature_distribution(results["Density"], params)),
        ("Spectrum", lambda: generate_spectrum(results["Temperature"], params))
    ]

    with tqdm(total=len(steps), desc="Simulation steps") as bar:
        for name, func in steps:
            results[name] = func()
            logger.info(f"{name} computed")
            bar.update(1)

    results.update({
        "simulation_id": sim_id,
        "timestamp": timestamp,
        "params": params
    })

    if save:
        os.makedirs(params["output_dir"], exist_ok=True)
        out_path = os.path.join(params["output_dir"], f"{sim_id}_out.json")
        save_results(results, file_path=out_path)
        logger.info(f"Results saved to {out_path}")

    if visualize:
        plot_spectrum(results["Spectrum"], sim_id=sim_id)
        animate_disk(results["Temperature"], sim_id=sim_id)
        logger.info("Visualization done")

    logger.info("Simulation completed successfully")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run QuasiStarSim disk simulation")
    parser.add_argument("--mass", type=float, required=True, help="Black hole mass (solar masses)")
    parser.add_argument("--inner_radius", type=float, required=True, help="Inner disk radius (Rg)")
    parser.add_argument("--outer_radius", type=float, required=True, help="Outer disk radius (Rg)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Viscosity α (0 < α ≤ 1)")
    parser.add_argument("--num_radial_bins", type=int, default=100, help="Radial bins")
    parser.add_argument("--inclination", type=float, default=30.0, help="Inclination (°)")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--no_save", action="store_true", help="Skip saving results")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization")

    args = parser.parse_args()
    params = vars(args)
    save_flag = not args.no_save
    viz_flag = not args.no_visualize

    run_simulation(params, save=save_flag, visualize=viz_flag)
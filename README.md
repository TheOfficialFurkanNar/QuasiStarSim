# Project Overview
 QuasiStarSim is a Python-based simulation framework for modeling post-main-sequence stellar phenomena, including:
- Accretion disk dynamics
- Quasi-star formation
- Gravitational collapse leading to black hole creation
The code is modular and supports customized input parameters, real-time plotting, and reproducible scientific output. Ideal for researchers, educators, and preliminary mission simulations.
- # Installation 🚀
- To get started, clone the repository and install the required packages:
- git clone https://github.com/TheOfficialFurkanNar/QuasiStarSim.git
cd QuasiStarSim
pip install -r requirements.txt
- # Usage 🛠️
- Run the main simulation script with default parameters:
- python simulate_quasistar.py
- Or customize your simulation via the CLI:
- python simulate_quasistar.py --mass 300 --rotation 0.5 --output results.csv
- Run the accretion disk simulator (an example) python -m quasistar run-disk --mass 4.3e6 --mdot 1e16 --eta 0.1 --steps 200 --animate
- # Output 📦
- Simulation produces:
- Stellar evolution plots (.png)
- Parameter logs (.csv)
- Final state summaries (.txt)
All outputs are stored in the results/ directory by default.
- # Applications 🎯
- Astrophysics Research
- Educational demos
- Pre-mission modeling for ESA or JAXA simulations
- Preliminary data generation for neural training in AGI models
- # Acknowledgements 🙏
- Built using:
- numpy, scipy, pandas, matplotlib, seaborn
Inspired by frontier work in stellar evolution and post-main-sequence dynamics.
- Don't forget to check requirements.txt for dependencies.

# MIT License 📜
This project is licensed under the terms of the MIT license. See the LICENSE file for more legal details.




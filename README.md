# Membrane-Anchored Mobile Tethers

This repository contains code, simulation outputs, and figure notebooks for:

**Membrane-anchored mobile tethers modulate condensate wetting, localization, and migration**  
*PRX Life* (2026). DOI: https://doi.org/10.1103/kxpb-9srd  
Preprint: https://www.biorxiv.org/content/10.1101/2024.12.04.626804v3

## Repository Layout

- `src/`: simulation scripts (JAX/NumPy/SciPy)
- `data/`: precomputed simulation outputs used for plotting
- `figures/`: Jupyter notebooks and exported figure panels
  - `fig1.ipynb`, `fig2.ipynb`, `fig3.ipynb`
  - output subfolders: `fig1/`, `fig2/`, `fig3/`

## Environment

Recommended Python: `3.10+` (this repo was edited in a Python 3.12 environment).

Core Python packages used:

- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `jax` and `jaxlib`
- `jupyterlab` (or `notebook`)

Example setup (CPU JAX):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib tqdm jupyterlab "jax[cpu]"
```

Note: The simulation scripts in `src/` can be used to reproduce all the simulation data presented in the paper. 
This repository currently contains precomputed simulation data for plotting Fig. 1-2. The raw data for Fig. 3 is too large to be included here, but it can be reproduced by running `curved_tubule_one_side_v1_run.py` with the appropriate parameters.

## Citation

If you use this repository, please cite the associated article:

Qiwei Yu et al., *Membrane-anchored mobile tethers modulate condensate wetting, localization, and migration*, *PRX Life* (2026), https://doi.org/10.1103/kxpb-9srd

Preprint: https://www.biorxiv.org/content/10.1101/2024.12.04.626804v3

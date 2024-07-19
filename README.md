# waveflow
A boundary-conditioned normalizing flows for electronic structures, and more!

Authors: Luca Thiede and Chong Sun [Email](sunchong137@gmail.com)

## Getting started
Create a new conda environment (default name `waveflow`) by
```bash
conda env create -f environment.yml
```
Then activate the conda environment by
```bash
conda activate waveflow
```
### Installing
In the same path containing `README.md`, type
```bash
pip install -e .
```

### Running waveflow
We provide two examples in the `example/` directory, where `run_benchmark.py` showcases the normalizing flows and `run_vqmc.py` examplifies square-normalizing flows applied to a one-dimensional hellium-like system.

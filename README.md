# waveflow
A boundary-conditioned normalizing flows for electronic structures, and more!

Authors: Luca Thiede, Chong Sun

## Getting started
We recommend using the GPU version. However, you can still run the code on CPU. Copy the 
`environment_<processor>.yml` to `environment.yml` and create a conda environment by
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
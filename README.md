# README
[![DOI](https://zenodo.org/badge/491751180.svg)](https://zenodo.org/badge/latestdoi/491751180)

This is the repository for the paper "Variational learning algorithms for quantum query complexity". 


Currently, this repo contains the following files:

1. `/VarQQA` folder : the main file for the paper, contains the implementation of the variational learning algorithms for quantum query complexity.
2.  `/sdp` folder: contains the SDP solver used in the paper.
3.  `/data` folder: contains the quantum query algorithm found the by VarQQA in this paper.It contains three subfolders 
    1.  `/data/grover`: contains the data for Grover search
    2. `/data/hamming_weight_modulo`: contains the data for the Hamming Weight Modulo  
    3.  `/data/EXACT`: contains the data for the $\mathrm{EXACT}_{k,l}^n$  function

To reproduce the ressult, create a new python environment with the following command:
```bash
conda create -y -n cuda118
conda install -y -n cuda118 -c conda-forge pytorch ipython matplotlib scipy tqdm cvxpy
conda activate cuda118
```
Run the corresponding python file in the `/VarQQA` for the VarQQA algorithm.

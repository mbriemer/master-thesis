This repository contains the code of my Master's thesis *Estimation with GANs*, which is based on the paper *A structural approach to adversarial estimation* ([Kaji, Manresa and Pouliot 2023](https://www.econometricsociety.org/publications/econometrica/2023/11/01/An-Adversarial-Approach-to-Structural-Estimation), [arXiv](https://arxiv.org/abs/2007.06169)) and uses the approximation of the Wasserstein distance avaiable in [`geomloss`](http://www.kernel-operations.io/geomloss/). The thesis was written under the supervision of Joachim Freyberger in 2024.

The latex and graphics are in `latex/`, with the PDF of the actual thesis [here](https://github.com/mbriemer/master-thesis/blob/main/latex/Thesis.pdf).

The code for the simulations is in `Code/`.
I programmed simulations both in the scientific Python stack and using pytorch. The code can be found in `Code/sp/` and `Code/torch/`, respectively.
`Code/hpc/` contains scripts for interacting with [an HPC cluster of the University of Bonn](https://www.hpc.uni-bonn.de/en/systems/marvin), usage of which I gratefully acknowledge.
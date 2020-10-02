# Content

This repository contains the implementation of a generative model, employing semi-supervised learning and physical constraints to learn the input-output map implicitly defined by a linear elliptic partial differential equation. The encoder and decoder are based on convolutional neural networks, and a physical coarse-grained model is embedded within the computational graph of PyTorch. The model is trained using stochastic variational inference to optimize the evidence lower bound, which incorporates physical information in addition to labeled and unlabeled data.

# Dependencies


The implementation makes use of PyTorch [PyTorch](https://pytorch.org/) for machine learning and automatic differentiation, while [FEniCS](https://fenicsproject.org/) is employed for the FOM, CGM and the physical constraints.


* pytorch 1.1.0
* fenics 2018.1.0


# Examples

A simple example to build and run a model is provided in example.ipynb

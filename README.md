# MpgGAN
GAN-based multi-parameter generation of stochastic rock discontinuities

This repository contains the source code associated with the manuscript:

**"GAN-Based Multi-Parameter Generation of Stochastic Rock Discontinuities for Key Block Identification"**

The code implements a Generative Adversarial Network (GAN) for generating stochastic rock discontinuity parameters from input data.

## 1. Overview

The workflow consists of two main steps:

1. Data preprocessing and normalization.
2. GAN training and stochastic discontinuity parameter generation.

Four discontinuity parameters are considered:

- Dip direction
- Dip angle
- Spacing
- Trace length

The input parameters are normalized to the range [0, 1] before GAN training.

## 2. Repository Structure

```text
MpgGAN/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── example_input.csv
│   └── preprocessed_data.csv
│
├── results/
│
├── preprocess.py
└── train_gan.py

# TP-TCM

Implementation of TP-TCM for tongue-pulse synergistic prescription recommendation.

## Overview

TP-TCM is a multi-source prescription recommendation framework that integrates
symptom, tongue, pulse, syndrome, and herb information for Traditional Chinese
Medicine (TCM) prescription recommendation.

This repository provides the implementation used for the main modeling and
downstream prediction procedures, including multi-source contrastive learning,
syndrome prediction, label-aware herb recommendation, training, and evaluation.

## Requirements

The implementation is based on Python and PyTorch.

Main dependencies include:

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- SciPy
- matplotlib

The provided STP subset is intended only for code execution and implementation verification. Due to its limited size and incomplete coverage of the original data distribution, results obtained on this subset are not expected to reproduce the performance reported in the paper.


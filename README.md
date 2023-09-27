Overview
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

![](https://raw.githubusercontent.com/skaliy/skaliy.github.io/master/assets/fastmonai_v1.png)

![CI](https://github.com/MMIV-ML/fastMONAI/workflows/CI/badge.svg)
[![Docs](https://github.com/MMIV-ML/fastMONAI/actions/workflows/deploy.yaml/badge.svg)](https://fastmonai.no)
[![PyPI](https://img.shields.io/pypi/v/fastMONAI?color=blue&label=PyPI%20version&logo=python&logoColor=white.png)](https://pypi.org/project/fastMONAI)

A low-code Python-based open source deep learning library built on top of [fastai](https://github.com/fastai/fastai), [MONAI](https://monai.io/), [TorchIO](https://torchio.readthedocs.io/), and [Imagedata](https://imagedata.readthedocs.io/).

fastMONAI simplifies the use of state-of-the-art deep learning
techniques in 3D medical image analysis for solving classification,
regression, and segmentation tasks. fastMONAI provides the users with
functionalities to step through data loading, preprocessing, training,
and result interpretations.

<b>Note:</b> This documentation is also available as interactive
notebooks.

# Installing

## From [PyPI](https://pypi.org/project/fastMONAI/)

`pip install fastMONAI`

## From [GitHub](https://github.com/MMIV-ML/fastMONAI)

If you want to install an editable version of fastMONAI run:

- `git clone https://github.com/MMIV-ML/fastMONAI`
- `pip install -e 'fastMONAI[dev]'`

# Getting started

The best way to get started using fastMONAI is to read our [paper](https://www.sciencedirect.com/science/article/pii/S2665963823001203) and dive into our beginner-friendly [video tutorial](https://fastmonai.no/tutorial_beginner_video). For a deeper understanding and hands-on experience, our comprehensive instructional notebooks will walk you through model training for various tasks like classification, regression, and segmentation. See the docs at https://fastmonai.no for more information.

| Notebook                                                                                                                                                                                                                                     | 1-Click Notebook                                                                                                                                                                                   |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [10a_tutorial_classification.ipynb](https://nbviewer.org/github/MMIV-ML/fastMONAI/blob/master/nbs/10a_tutorial_classification.ipynb) <br>shows how to construct a binary classification model based on MRI data.                             | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/10a_tutorial_classification.ipynb)          |
| [10b_tutorial_regression.ipynb](https://nbviewer.org/github/MMIV-ML/fastMONAI/blob/master/nbs/10b_tutorial_regression.ipynb) <br>shows how to construct a model to predict the age of a subject from MRI scans (“brain age”).                | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/10b_tutorial_regression.ipynb)              |
| [10c_tutorial_binary_segmentation.ipynb](https://nbviewer.org/github/MMIV-ML/fastMONAI/blob/master/nbs/10c_tutorial_binary_segmentation.ipynb) <br>shows how to do binary segmentation (extract the left atrium from monomodal cardiac MRI). | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/10c_tutorial_binary_segmentation.ipynb)     |
| [10d_tutorial_multiclass_segmentation.ipynb](https://nbviewer.org/github/MMIV-ML/fastMONAI/blob/master/nbs/10d_tutorial_multiclass_segmentation.ipynb) <br>shows how to perform segmentation from multimodal MRI (brain tumor segmentation). | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/10d_tutorial_multiclass_segmentation.ipynb) |

# How to contribute

See
[CONTRIBUTING.md](https://github.com/MMIV-ML/fastMONAI/blob/master/CONTRIBUTING.md)

# Citing fastMONAI

If you are using fastMONAI in your research, please use the following citation:

```
@article{KALIYUGARASAN2023100583,
title = {fastMONAI: A low-code deep learning library for medical image analysis},
journal = {Software Impacts},
pages = {100583},
year = {2023},
issn = {2665-9638},
doi = {https://doi.org/10.1016/j.simpa.2023.100583},
url = {https://www.sciencedirect.com/science/article/pii/S2665963823001203},
author = {Satheshkumar Kaliyugarasan and Alexander S. Lundervold},
keywords = {Deep learning, Medical imaging, Radiology},
abstract = {We introduce fastMONAI, an open-source Python-based deep learning library for 3D medical imaging. Drawing upon the strengths of fastai, MONAI, and TorchIO, fastMONAI simplifies the use of advanced techniques for tasks like classification, regression, and segmentation. The library's design addresses domain-specific demands while promoting best practices, facilitating efficient model development. It offers newcomers an easier entry into the field while keeping the option to make advanced, lower-level customizations if needed. This paper describes the library's design, impact, limitations, and plans for future work.}
}
```
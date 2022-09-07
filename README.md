Overview
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

![](https://raw.githubusercontent.com/skaliy/skaliy.github.io/master/assets/fastmonai_v1.png)

![CI](https://github.com/MMIV-ML/fastMONAI/workflows/CI/badge.svg)
[![Docs](https://github.com/MMIV-ML/fastMONAI/actions/workflows/deploy.yaml/badge.svg)](https://fastmonai.no)
[![PyPI](https://img.shields.io/pypi/v/fastMONAI?label=PyPI%20version&logo=python&logoColor=white.png)](https://pypi.org/project/fastMONAI)

A low-code Python-based open source deep learning library built on top
of [fastai](https://github.com/fastai/fastai),
[MONAI](https://monai.io/), and
[TorchIO](https://torchio.readthedocs.io/).

fastMONAI simplifies the use of state-of-the-art deep learning
techniques in 3D medical image analysis for solving classification,
regression, and segmentation tasks. fastMONAI provides the users with
functionalities to step through data loading, preprocessing, training,
and result interpretations.

<b>Note:</b> This documentation is also available as interactive
notebooks.

# Installing

## From PyPI

`pip install fastMONAI`

## From Github

If you want to install an editable version of fastMONAI run:

- `git clone https://github.com/MMIV-ML/fastMONAI`
- `pip install -e 'fastMONAI[dev]'`

# Getting started

The best way to get started using fastMONAI is to read the
[paper](https://github.com/MMIV-ML/fastMONAI/tree/master/paper) and look
at the step-by-step tutorial-like notebooks to learn how to train your
own models on different tasks (e.g., classification, regression,
segmentation). See the docs at https://fastmonai.no for more
information.

| Notebook                                                                                                                                                                                                                                     | 1-Click Notebook                                                                                                                                                                                   |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [09a_tutorial_classification.ipynb](https://nbviewer.org/github/MMIV-ML/fastMONAI/blob/master/nbs/09a_tutorial_classification.ipynb) <br>shows how to construct a binary classification model based on MRI data.                             | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/09a_tutorial_classification.ipynb)          |
| [09b_tutorial_regression.ipynb](https://nbviewer.org/github/MMIV-ML/fastMONAI/blob/master/nbs/09b_tutorial_regression.ipynb) <br>shows how to construct a model to predict the age of a subject from MRI scans (“brain age”).                | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/09b_tutorial_regression.ipynb)              |
| [09c_tutorial_binary_segmentation.ipynb](https://nbviewer.org/github/MMIV-ML/fastMONAI/blob/master/nbs/09c_tutorial_binary_segmentation.ipynb) <br>shows how to do binary segmentation (extract the left atrium from monomodal cardiac MRI). | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/09c_tutorial_binary_segmentation.ipynb)     |
| [09d_tutorial_multiclass_segmentation.ipynb](https://nbviewer.org/github/MMIV-ML/fastMONAI/blob/master/nbs/09d_tutorial_multiclass_segmentation.ipynb) <br>shows how to perform segmentation from multimodal MRI (brain tumor segmentation). | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/09d_tutorial_multiclass_segmentation.ipynb) |

# How to contribute

See
[CONTRIBUTING.md](https://github.com/MMIV-ML/fastMONAI/blob/master/CONTRIBUTING.md)

# Citing fastMONAI

    @article{kaliyugarasan2022fastMONAI,
      title={fastMONAI: a low-code deep learning library for medical image analysis},
      author={Kaliyugarasan, Satheshkumar and Lundervold, Alexander Selvikv{\aa}g},
      year={2022}
    }

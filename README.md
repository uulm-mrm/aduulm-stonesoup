# Self-Assessment and Monitoring Module for Tracking Algorithms: Implementation in the Stone Soup Framework

This repository contains the implementation of self-assessment extensions for the Stone Soup framework. 
These extensions are part of our ongoing research on performance monitoring in tracking systems.

For more technical details, please refer to our publication [Self-Assessment and Monitoring Module for Tracking Algorithms in the Stone Soup Framework](https://doi.org/10.1109/MFI67357.2025.11259162) with the corresponding [slides](https://www.events-project.eu/wp-content/uploads/2026/02/2025_mfi_self-assessment-module_griebel_wodtko_eu_export.pdf). 

This publication has won the **Second Place Best Paper Award** at 2025 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI).

## 🔍 Overview

Our proposal introduces a Self-Assessment (SA) module, referred to as Self-Assessor, into the Stone Soup framework.
The module enables tracking algorithms to monitor and evaluate their own performance in real-time, facilitating more reliable decision-making in autonomous systems.

<img src="aduulm_scripts/images/concept-selfassessor-stonesoup.png" width="600">

## 📄 Citation
If you find this repository useful in your research, please consider citing our work.

```
@INPROCEEDINGS{griebel2025aduulmstonesoup,
  author={Griebel, Thomas and Wodtko, Thomas and Buchholz, Michael and Dietmayer, Klaus},
  booktitle={2025 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI)}, 
  title={Self-Assessment and Monitoring Module for Tracking Algorithms in the Stone Soup Framework}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/MFI67357.2025.11259162}
}
```

The following publications are included in the self-assessment framework:

* [Kalman Filter Meets Subjective Logic: A Self-Assessing Kalman Filter Using Subjective Logic](https://doi.org/10.23919/FUSION45008.2020.9190520)
* [Self-Assessment for Single-Object Tracking in Clutter Using Subjective Logic](https://doi.org/10.23919/FUSION49751.2022.9841294)
* [Online Performance Assessment of Multi-Sensor Kalman Filters Based on Subjective Logic](https://doi.org/10.23919/FUSION52260.2023.10224188)

## 🛠️ Installation & Development Setup

To start developing with our self-assessment extensions, please use Python 3.12 and clone the appropriate branch:

```
git clone "https://github.com/uulm-mrm/aduulm-stonesoup.git"
cd Stone-Soup
python -m pip install -e ".[dev,aduulm]"
```

Make sure to check out our self-assessment extensions branch:
[selfassessment_extensions](https://github.com/uulm-mrm/aduulm-stonesoup/tree/selfassessment_extensions)

## 📘 Tutorials & Usage

If you want to experiment with the Self-Assessor, tutorials can be found here:

👉 [Self-Assessor Tutorials](https://github.com/uulm-mrm/aduulm-stonesoup/tree/selfassessment_extensions/aduulm_scripts/tutorials)

These tutorials allow you to:

 - Disturb and manipulate ground truth trajectories
 - Disturb and manipulate measurements
 - Obtain self-assessment results to detect disturbances

The currently available self-assessment approaches consist of:

- Linear and nonlinear Kalman filter self-assessor
  - `01_KalmanFilterTutorialWithSelfAssessment.py`
  - `02_ExtendedKalmanFilterTutorialWithSelfAssessment.py`  
  - `03_UnscentedKalmanFilterTutorialWithSelfAssessment.py`  

- Multi-sensor Kalman filter self-assessor 
  - `01_MultiSensorKalmanFilterTutorialWithSelfAssessment.py`  

- Self-assessor for (multi-sensor) single-object tracking in clutter using nearest neighbour association
  - `05_DataAssociation-ClutterWithSelfAssessment.py`  
  - `05_MultiSensorDataAssociation-ClutterWithSelfAssessment.py`  

- Self-assessor for (multi-sensor) single-object tracking in clutter using probabilistic data association
  - `07_PDATutorialWithSelfAssessment.py`  
  - `07_MultiSensorPDATutorialWithSelfAssessment.py`

### 🔧 Disturbance in Transition Model
<img src="aduulm_scripts/images/disturbance-transition-model.png" width="600">

### 🔧 Disturbance in Measurement Model
<img src="aduulm_scripts/images/disturbance-measurement-model.png" width="600">

### 📉 Self-Assessment: Kalman Self-Assessor and Single-Time Step NIS
<img src="aduulm_scripts/images/selfassessor_kalman_and_nis_single.png" width="600">

### 📉 Self-Assessment: Kalman Self-Assessor and Time-Averaged NIS
<img src="aduulm_scripts/images/selfassessor_kalman_and_nis_averaged.png" width="600">

✅ Try out the tutorials and start experimenting with self-assessment for your own tracking setups.

**--- Here begins the original Stone Soup README ---**

The following section contains the unmodified README from the original Stone Soup project.

<h1><img valign="middle" alt="Stone Soup Logo" src="https://raw.githubusercontent.com/dstl/Stone-Soup/main/docs/source/_static/stone_soup_logo.svg" height="100"> Stone Soup</h1>

[![PyPI](https://img.shields.io/pypi/v/stonesoup?style=flat)](https://pypi.org/project/stonesoup)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/stonesoup.svg)](https://anaconda.org/conda-forge/stonesoup)
[![CircleCI branch](https://img.shields.io/circleci/project/github/dstl/Stone-Soup/main.svg?label=tests&style=flat)](https://circleci.com/gh/dstl/Stone-Soup)
[![Codecov](https://img.shields.io/codecov/c/github/dstl/Stone-Soup.svg)](https://codecov.io/gh/dstl/Stone-Soup)
[![Read the Docs](https://img.shields.io/readthedocs/stonesoup.svg?style=flat)](https://stonesoup.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://img.shields.io/gitter/room/dstl/Stone-Soup.svg?color=informational&style=flat)](https://gitter.im/dstl/Stone-Soup?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.4663993-informational)](https://doi.org/10.5281/zenodo.4663993)

## Background
Stone Soup is a software project to provide the target tracking and state estimation
community with a framework for the development and testing of tracking and state
estimation algorithms.

An article is [available](https://www.gov.uk/government/news/dstl-shares-new-open-source-framework-initiative) that details the background to the project, and contains links to sample data.

Please see the
[Stone Soup documentation](https://stonesoup.readthedocs.org/) for more
information.

Please see the [tutorials](https://stonesoup.readthedocs.io/en/latest/auto_tutorials/index.html),
[examples](https://stonesoup.readthedocs.io/en/latest/auto_examples/index.html),
and [demonstrations](https://stonesoup.readthedocs.io/en/latest/auto_demos/index.html),
which you can also try out on Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dstl/Stone-Soup/main?filepath=notebooks)

## License
Stone Soup is released under MIT License. Please see [License](LICENSE) for details.

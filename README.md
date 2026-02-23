# RETROICOR AROMA Comparison

This repository contains the analysis pipeline for project 3013068.03 at the Donders Center for Cognitive Neuroimaging, evaluating data-driven and peripheral-driven physiological noise correction in fMRI.

---

## Setup
* **`setup.py`** Creates the required data structure for the analysis scripts. After specifying the path to the downloaded dataset, it validates data availability and provides feedback for subsequent execution.

---

## HR_approach
* **`HR_01_analysis.py`** Handles the processing and cleaning of heart rate data using the HERA tool. It calculates and plots the percentage of heart rate peaks rejected per subject to assess the quality of potential RETROICOR regressors.

---

## GLM_approach



* **`GLM_01_confound_creation.py`** Creates separate regressor files for RETROICOR (10 cardiac phase, 10 respiratory phase, 4 interaction terms) and ICA-AROMA.
* **`GLM_02_run.py`** Implements a standard GLM in `nilearn` using a combination of RETROICOR, AROMA, and aCompCor. Outputs thresholded images (Uncorrected, FDR, and FWE corrected).
* **`GLM_03_plotting.py`** Generates glass-brain plots showing variance explained (total and unique) for RETROICOR, AROMA, and aCompCor.
* **`GLM_04_melodic_component_glm.py`** Assesses the overlap of melodic components with the unique variance distribution of RETROICOR to identify potential AROMA misclassifications.
* **`GLM_05_aroma_misclassifications.py`** Evaluates if adding specific Melodic components back into the design matrix improves denoising performance.
* **`GLM_06_dice_calculation.py`** Calculates and plots a Dice index for thresholded F-maps to visualize spatial improvements from Melodic component additions.
* **`GLM_07_zchange_calculation.py`** The primary script for manuscript calculations, using a z-map change approach rather than the Dice coefficient to assess improvement quality.

---

## TSNR_approach



* **`TSNR_01_calculation.py`** Calculates tSNR maps for the different tested noise-cleaning procedures.
* **`TSNR_02_calculation.py`** Handles mask creation for subsequent tSNR analysis, focusing on Gray Matter and the Locus Coeruleus.
* **`TSNR_03_stats.py`** Calculates and tests tSNR improvements across multiple regions of interest: Whole Brain, Cortex Gray Matter, Brainstem, and Locus Coeruleus.
* **`TSNR_04_plotting.py`** Generates corresponding bar graphs for the main and unique cleaning effects.
* **`TSNR_05_stats_correlation.py`** Calculates correlations between unique RETROICOR improvements and ICA-AROMA (Manuscript Figure 6).
* **`TSNR_06_mean_plotting.py`** Computes mean tSNR maps for the various noise correction methods and combinations (Manuscript Figure 1).

---

## Citation

> Krentz, M., Tutunji, R., Kogias, N., Mahadevan, H. M., Reppmann, Z. C., Krause, F., & Hermans, E. J. (2023). **Physiological Noise Correction in Brainstem Imaging: An Empirical Evaluation of fMRI Data-Driven and Peripheral Physiological Recording-Driven Methods.** *bioRxiv*. doi: [10.1101/2023.02.22.529506](https://doi.org/10.1101/2023.02.22.529506)

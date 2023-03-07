# RETROICOR AROMA Comparison

This repository contains all used analysis steps of project 3013068.03 at the Donders Center for Cognitive Neuroimaging.
Assuming familiarity with the general project, the folder/script structure is as follows:

**Setup**

*setup.py*

To run the analysis scripts a corresponding data structure is created. Additionally, after path specification of the downloaded data set, 
availability of all data for subsequent script execution will be checked and fed back.

**HR_approach**

*HR_01_analysis.py*

The processing and cleaning of heart rate data was done using an inhouse tool (HERA, https://github.com/can-lab/hera).
To assess the quality of potential RETROICOR regressors and evaluate the amount of interpolation this script calculates and plots the amount of 
heart rate peaks rejected due to low data quality on a per subject basis.

**GLM_approach**

*GLM_01_confound_creation.py*

This script creates seperate available .txt files for all necessary RETROICOR regressors as well as all used AROMA regressors.
Including: 10 cardiac phase regressors, 10 respiratory phase regressors, 4 interation terms.
    
*GLM_02_run.py*

This script implements the calculation of a standard glm in nilearn. Using a combination of RETROICOR model, AROMA regressors as well as aCompCor regressors
as variables of interest. 

Additionally, multiple thresholded images are created: 
* Uncorrected, 
* FDR-corrected, 
* FWE-corrected.

*GLM_03_plotting.py*

This script creates a variety of glass-brain plots showing per subject as well as overall:
* The variance explained by RETROICOR
* The variance explained by AROMA
* The variance explained by aCompCor
* The unqiue variance explained by aCompCor in the presence of AROMA
* The unqiue variance explained by RETROICOR in the presence of AROMA
* The unqiue variance explained by RETROICOR in the presence of AROMA and aCompCor

*GLM_04_melodic_component_glm.py*

This script is used to assess the overlay of melodic components used for AROMA with previously identified unique variance distribution of RETROICOR. Potentially, this allow for identifcation
of AROMA misclassifications. Consequently the following steps are taken on a per subject basis:
   * Binarise F-Map for unique variance explained by RETROICOR
   * Compute Melodic spatial maps by running single-regressor GLMs
   * Assess Overlap of melodic results with binarised RETROICOR maps
   
*GLM_05_aroma_misclassifications.py*

This scripts assesses whether Melodic spatial maps with a high RETROICOR overlap - calculated in the previous script -
removes the unique variance explained by RETROICOR as a consequence of improving AROMA output. To do so, new GLMs are created
as run in *GLM_02_run.py* with the addition identified Melodic components to the design matrix.

*GLM_06_dice_calculation.py*

To assess the quality of the potential improvement as a consequence of added melodic components a dice index is calculated and plotted
for a binarised thresholded F-map as created by *GLM_02_run.py* and *GLM_05_aroma_misclassifications.py*. This is done to visualise the potential improvement
of Melodic component additions. Note: The binarising of supra-threshold F-values does not maintain the strength of the potential effect. Furthermore, this approach was not described in the manuscript resulting from these scripts.

*GLM_07_zchange_calculation.py*

Similar to *GLM_06_dice_calculation.py*, this script aims to assess the quality of the potential improvement as a consequence of added melodic components. However, this script ultimately used for manuscript calulations, uses an approach of comparing z-map change rather than dice coeffiecient. 


**TSNR_approach**

*TSNR_01_calculation.py*

This script implements the calculation of TSNR for the different tested noise-cleaning procedures.

*TSNR_02_calculation.py*

This script implements the mask creation for the following TSNR analysis for grey matter as well as locus coeruleus.

*TSNR_03_stats.py*

This scripts calculates and tests the TSNR improvement as a consequence of the different cleaning procedures for a variety of regions of interest:
* Whole Brain
* Cortex Gray Matter
* Brainstem
* Locus Coeruleus

*TSNR_04_plotting.py*

This script creates corresponding bar-graphs for calculations made in *TSNR_03_stats.py*.
* Collected main effets of cleaning
* Collected unique cleaning effects

*TSNR_05_stats_correlation*

This script calculates correlations between unique RETROICOR improvements and ICA-AROMA (Manuscript Figure 6).

*TSNR_06_mean_plotting*

This script calculates mean TSNR maps for the different noise correction methods and combinations (Manuscript Figure 1).

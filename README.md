# RETROICOR AROMA Comparison

This repository contains all used analysis steps of project 3013068.03 at the Donders Center for Cognitive Neuroimaging.
Assuming familiarity with the general project the folder/script structure is as follows:

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

This script implements the calculation of a standard glm in nilearn. Using a combination of RETROICOR model and AROMA regressors
as variables of interest. 

*GLM_03_plotting.py*

This script creates a variety of glass-brain plots showing per subject as well as overall.

*GLM_04_melodic_component_glm.py*

This script is used to assess the overlay of melodic components used for AROMA with previously identified unique variance distribution of RETROICOR. This allows for identifcation of AROMA misclassifications. Consequently the following steps are taken on a per subject basis:
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
of Melodic component additions. Note: The binarising of supra-threshold F-values does not maintain the strength of the potential effect.

**TSNR_approach**

*TSNR_01_calculation.py*

This script implements the calculation of TSNR for the different tested noise-cleaning procedures.

Creating different TSNR map for:
   * Uncleaned data
   * aggrAROMA cleaned data
   * RETROICOR cleaned data
   * aggrAROMA AND RETROICOR cleaned data

Additionally maps are created visualising the unique contributions of one method OVER another.
   * Unique TSNR improvement of aggrAROMA (TSNR of aggrAROMA+RETRO - TSNR of RETRO)
   * Unique TSNR improvement of RETROICOR (TSNR of aggrAROMA+RETRO - TSNR of aggrAROMA)
   * Difference in TSNR improvement between RETROICOR and aggrAROMA (TSNR of aggrAROMA - TSNR of RETRO)
   * TSNR improvement of uncleaned data for aggrAROMA (TSNR of aggrAROMA - TSNR of uncleaned data)
   * TSNR improvement of uncleaned data for RETROICOR (TSNR of RETROICOR - TSNR of uncleaned data)

*TSNR_02_stats.py*

This scripts calculates and tests the TSNR improvement as a consequence of the different cleaning procedures for a variety of regions of interest:
* Whole Brain
* Cortex Gray Matter
* Brainstem
* Locus Coeruleus

*TSNR_03_plotting.py*

This script creates corresponding bar-graphs for calculations made in *TSNR_02_stats.py*.
* Main effect of cleaning per ROI
* Unqiue cleaning effect per ROI
* Collected main effets of cleaning
* Collected unique cleaning effects


# Disparities in Police (NYPD) Deployment Patterns
Code to reproduce results in "Detecting disparities in police deployments using dashcam data" (2023) by Matt Franchi, J.D. Zamfirescu-Pereira, Wendy Ju, and Emma Pierson.

## Reproducing results 
1. **Virtual environment setup.** Our code is run within a conda environment, with all analysis performed on a Linux Ubuntu HPC resource. This environment can be reproduced by running `conda env create --prefix YOUR_PATH_HERE -- file fpp_policing_env.yml`. Once the environment has been set up successfully, activate it before running any code by running the command `conda activate YOUR_PATH_HERE`. 

2. **Downloading data.** Reach out for more info about accessing the original dashcam image data. However, we will not be releasing the original dataset due to size constraints (the data clocks in at over 10TB) and privacy considerations. *We instead release aggregated data in the form of bootstrapped disparity results.*

3. **Generating results for paper**. Figures and results can be reproduced by running `bootstrapped_plots.ipynb`. See below for details. 

## Model Training - Compute Resources 
To train our YOLOv7 object classifier, we utilized the following hardware: 
- 4x RTX A6000 GPU
- 256GB RAM 

This allowed us to use a batch size of 40. 

# Disparities in Police (NYPD) Deployment Patterns
Code to reproduce results in "Detecting disparities in police deployments using dashcam data" (2023) by Matt Franchi, J.D. Zamfirescu-Pereira, Wendy Ju, and Emma Pierson.

## Reproducing results 
1. **Virtual environment setup.** Our code is run within a conda environment, with all analysis performed on a Linux Ubuntu HPC resource. This environment can be reproduced by running `conda env create --prefix YOUR_PATH_HERE -- file fpp_policing_env.yml`. Once the environment has been set up successfully, activate it before running any code by running the command `conda activate YOUR_PATH_HERE`. 

2. **Downloading data.** Reach out for more info about accessing the original dashcam image data. However, we will not be releasing the original dataset due to size constraints (the data clocks in at over 10TB) and privacy considerations. *We instead release aggregated data in the form of bootstrapped disparity results.*

3. **Generating results for paper**. Figures and results can be reproduced by running `bootstrapped_plots.ipynb`. See below for details. 


## Public datasets we used as additional metadata for each dashcam image. 
### [NYPD Crime Data (*NYPD Complaint Data Historic*)](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i)
The NYPD Complaint Data Historic dataset is a collection of records related to complaints received by the New York City Police Department (NYPD) from civilians. This dataset includes information about the type of complaint, the location where the incident occurred, and the date and time of the complaint.

The data in this dataset covers a period of several years, from 2006 to the present day, and is updated regularly. It contains a wealth of information that can be used to study crime trends, identify areas where crime is more prevalent, and to monitor the effectiveness of the NYPD's response to complaints.

Some of the specific types of complaints included in this dataset are related to incidents such as assault, robbery, burglary, and noise complaints. Each record in the dataset contains detailed information about the complaint, including the complaint type, the location where the incident occurred, and the status of the complaint (whether it is still under investigation or has been resolved).

We filter the dataset to only include crimes from our dataset's period of coverage (March-November 2020), and also only analyze felony-level crimes. 

### [Median Household Income @ CBG Level (*2020 ACS 5-year Estimates*)](https://data.census.gov/table?q=B19013&g=050XX00US36005$1500000,36047$1500000,36061$1500000,36081$1500000,36085$1500000&tid=ACSDT5Y2021.B19013)
The census table B19013 is a dataset that provides information about the income levels of people living in the block groups of New York City. Block groups are small geographic areas used by the U.S. Census Bureau to divide larger census tracts into smaller, more manageable units.

This dataset is derived from the American Community Survey (ACS), a yearly survey conducted by the U.S. Census Bureau that collects data on various demographic and socioeconomic characteristics of households across the country. The B19013 table specifically provides data on median household income, which is the middle point of all household incomes in a given block group.

The dataset provides a wealth of information on income levels across New York City, which can be used by researchers, policymakers, and community organizations to better understand economic disparities and plan programs to address them. For example, the data could be used to identify neighborhoods with low median incomes, which may be in need of affordable housing or job training programs.

It is important to note that while median household income is a useful measure of overall economic well-being, it does not capture the full range of income levels in a given block group. Additionally, the data is subject to sampling error, as it is based on survey responses from a sample of households rather than a complete count of all households in a given area. However, despite these limitations, the census table B19013 remains a valuable resource for anyone interested in understanding the economic landscape of New York City.


## Model Training - Compute Resources 
To train our YOLOv7 object classifier, we utilized the following hardware: 
- 4x RTX A6000 GPU
- 256GB RAM 

This allowed us to use a batch size of 40. 

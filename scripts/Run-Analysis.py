#!/usr/bin/env python
# coding: utf-8

# # Estimating Exposure to Police from Dashcam Data 
# Matt Franchi, Jan 2023 
# 
# This notebook contains all work needed to generate paper materials for the FAccT 2023 conference. 

# ## 0. Module Imports 

# In[2]:


from pprint import pprint
import pandas as pd 
import geopandas as gpd
import numpy as np 
from tqdm import tqdm
from zoneinfo import ZoneInfo
import statsmodels.api as sm
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from tqdm.auto import tqdm
tqdm.pandas()
import seaborn as sns 
from os import get_terminal_size, terminal_size
import json

# ## (Optional) Enable LaTeX font rendering 

# In[3]:


# Requires local LaTeX installation 
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rcParams['text.usetex'] = True
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage[T1]{fontenc}')


# ## Global Constants 
try:
    TERMINAL_SIZE = get_terminal_size()
    def print_whole_screen_line(): 
        print('-' * TERMINAL_SIZE.columns)
except Exception as e: 
    print(e) 
    def print_whole_screen_line(): 
        print('-' * 80)
    




# ### I/O Paths

# In[4]:


ANL_DATASET_PATH = "../output/yesterday/analysis_dataset.csv"
FIRST_CHUNK_PATH = "../output/1603771200000.csv"
VALSET_PATH = "../valset.csv"
TESTSET_PATH = "../testset.csv"
PAPER_GIT_REPO_PATH = "../plots_reverified_1215_20bootstraps"

# make figures, tables subdir in PAPER_GIT_REPO_PATH 
import os
os.makedirs(f"{PAPER_GIT_REPO_PATH}/figures", exist_ok=True)
os.makedirs(f"{PAPER_GIT_REPO_PATH}/tables", exist_ok=True)


# ### Geographic 

# In[5]:


WGS = 'EPSG:4326'
PROJ_CRS = 'EPSG:2263'
NYC_COUNTY_CODES = ['005', '047', '061', '081', '085']


# ### Analysis Parameters 
COLS_TO_DEDUPLICATE_ON = ['lat', 'lng', 'timestamp'] # columns to use to check for duplicates
MIN_DATE_FOR_DEMOGRAPHIC_ANALYSIS = datetime.datetime(2020, 10, 5, 0, 0, 0, tzinfo=ZoneInfo('US/Eastern')) # don't use data before this data to analyze disparities / demographics
POSITIVE_CLASSIFICATION_THRESHOLD = 0.770508 # threshold to define a positive prediction
LOCATION_COL_TO_GROUP_ON = 'GEOID20' # This should be the name of the column we're analyzing location grouping at - e.g., corresponding to Census Block Group or Census tract. CHECKED
TOTAL_POPULATION_COL = 'Estimate_Total' # needs to match whether using Census tract or Block group. [Answer: CBG]
WHITE_POPULATION_COL = 'Estimate_Total_Not_Hispanic_or_Latino_White_alone'
BLACK_POPULATION_COL = 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone'
ASIAN_POPULATION_COL = 'Estimate_Total_Not_Hispanic_or_Latino_Asian_alone'
HISPANIC_POPULATION_COL = 'Estimate_Total_Hispanic_or_Latino'
POPULATION_COUNT_COLS = [WHITE_POPULATION_COL, BLACK_POPULATION_COL, ASIAN_POPULATION_COL, HISPANIC_POPULATION_COL, TOTAL_POPULATION_COL]
TIME_AND_DATE_COL = 'time_and_date_of_image'
DEMOGRAPHIC_COLS = ['density_cbg', # things we want to look at correlations with. Demographic cols may not be best name. 
                    'distance_from_nearest_crime_6hr',
                    'distance_from_nearest_police_station',
                    'median_household_income']
PREDICTION_COLS = ['above_threshold', 'calibrated_prediction', 'prediction_adjusted_for_police_station_distance'] # columns with police car predictions. We define these
MIN_POPULATION_IN_AREA = 500
BOROUGH_COL = 'boroname'
NEIGHBORHOOD_COL = 'ntaname'
N_BOOTSTRAPS = 20
ZONE_THRESHOLD = 0.5


# ## 1. Dataset Verification

# function to load dataset from CSV on disk, with three options: (1) in chunks, (2) load first chunk, (3) load at once with pyarrow 
def load_analysis_dataset(load_in_chunks=False, load_first_chunk=False, use_pyarrow=False):
    if load_in_chunks:
        print("Loading data in chunks...")
        d = pd.concat([chunk for chunk in tqdm(pd.read_csv(ANL_DATASET_PATH, chunksize=100000), total=221, desc='Loading data')])
    elif load_first_chunk:
        print("Loading first chunk...")
        d = pd.concat([chunk for chunk in tqdm(pd.read_csv(FIRST_CHUNK_PATH, chunksize=100000), total=5, desc='Loading data')])
    elif use_pyarrow:
        print("Loading data with pyarrow...")
        d = pd.read_csv(ANL_DATASET_PATH, engine='pyarrow')
    else:
        print("Loading data...")
        d = pd.read_csv(ANL_DATASET_PATH)
    return d


d = load_analysis_dataset(load_first_chunk=False, use_pyarrow=True)
print_whole_screen_line()





d.head()


# ### Preprocessing



# function to convert specified datetime column to datetime type and convert to EST timezone
def convert_datetime_column_to_est(d, datetime_col):
    d[datetime_col] = pd.to_datetime(d[datetime_col])
    d[datetime_col] = d[datetime_col].dt.tz_convert('US/Eastern')
    return d

d = convert_datetime_column_to_est(d, TIME_AND_DATE_COL)


# function to inspect columns of dataframe 
def inspect_columns(d):
    print("Columns in d: ")
    pprint(list(d.columns.values), width=120, compact=True)

inspect_columns(d)
print_whole_screen_line()

# 

# function to remove duplicates from dataset based on specific columns 
def remove_duplicates(d, cols_to_deduplicate_on):
    duplicate_idxs = d.duplicated(subset=cols_to_deduplicate_on)
    print("warning: %i duplicates identified using %s, fraction %2.6f of rows; dropping rows" % (duplicate_idxs.sum(), cols_to_deduplicate_on, duplicate_idxs.mean()))
    d = d.loc[~duplicate_idxs].copy()
    return d

d = remove_duplicates(d, COLS_TO_DEDUPLICATE_ON)
print_whole_screen_line()

cbg_zone_data = pd.read_csv('/share/pierson/nexar_data/5_other_datasets/cbgs_zone_data.csv')
assert (1.*(cbg_zone_data['C'] > ZONE_THRESHOLD) + 1.*(cbg_zone_data['M'] > ZONE_THRESHOLD) + 1.*(cbg_zone_data['R'] > ZONE_THRESHOLD)).max() == 1
cbg_zone_dict = {}
for zone_val in ['C', 'M', 'R']:
    zones = cbg_zone_data.loc[cbg_zone_data[zone_val] >= ZONE_THRESHOLD]
    print("%i CBGs classified as %s" % (len(zones), zone_val))
    cbg_zone_dict.update(dict(zip(zones['GEOID20'].values, [zone_val for _ in range(len(zones))])))
print(len(cbg_zone_dict))
d['zone'] = d['GEOID20'].map(lambda x:cbg_zone_dict[x] if x in cbg_zone_dict else None)
print("zone classification of images")
print(d['zone'].value_counts(dropna=False))

print_whole_screen_line()


def household_income_map(x):
    if x == '-' or x == '':
        return None
    elif x == '250,000+':
        return 250000
    elif x == '2,500-':
        return 2500
    return float(x)

# function to add median household income column to dataset, with custom map
def add_median_household_income_column(d, map):
    d['median_household_income'] = d['median_household_income'].map(map)
    return d

d = add_median_household_income_column(d, household_income_map)

# function to add race population fraction columns to dataset
def add_race_frac_columns(d, population_cols): 
    for col in population_cols:
        d[col+'_frac'] = d[col] / d[TOTAL_POPULATION_COL]
        assert d[col+'_frac'].dropna().max() <= 1
        assert d[col+'_frac'].dropna().min() >= 0
        DEMOGRAPHIC_COLS.append(col+'_frac')
    return d

d = add_race_frac_columns(d, POPULATION_COUNT_COLS)




d['GeoID'] = d['GeoID'].astype(str)



# function to add date column to dataset 
def add_date_column(d, datetime_col):
    d['date'] = d[datetime_col].dt.date
    return d

d = add_date_column(d, TIME_AND_DATE_COL)

# function to print unique locations by date 
def print_unique_locations_by_date(d, location_col_to_group_on):
    locations_by_date = d.groupby('date')[location_col_to_group_on].nunique()
    print('unique locations by', locations_by_date)

print_unique_locations_by_date(d, LOCATION_COL_TO_GROUP_ON)
print_whole_screen_line()

# function to filter dataset for dates with full coverage 
def filter_for_dates_with_full_coverage(d, min_date_for_full_coverage):
    len_before_filter = len(d.index)
    print("In demographic analysis, filtering for locations after %s because more geographically representative" % MIN_DATE_FOR_DEMOGRAPHIC_ANALYSIS)
    d = d.loc[d['time_and_date_of_image'] >= min_date_for_full_coverage].copy()
    print("%i/%i rows remaining" % (len(d), len_before_filter))
    return d

d = filter_for_dates_with_full_coverage(d, MIN_DATE_FOR_DEMOGRAPHIC_ANALYSIS)
print_whole_screen_line()



# 

# function to fill in NA data in specified columns 
def fill_na_data(d, cols_to_fill_na):
    for col in cols_to_fill_na:
        if col in d.columns:
            d[col].fillna(0, inplace=True)
    return d


COLS_TO_FILL_NA = ['conf', 'distance_from_nearest_police_station', 'distance_from_nearest_crime_1hr', 'distance_from_nearest_crime_3hr', 'distance_from_nearest_crime_6hr', 'density_cbg', 'Estimate_Total', 'Estimate_Total_Not_Hispanic_or_Latino_White_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Asian_alone', 'Estimate_Total_Hispanic_or_Latino', 'Estimate_Total_Not_Hispanic_or_Latino']
d = fill_na_data(d, COLS_TO_FILL_NA)



# ### Sanity Checks

# 

# function to check that all lat/long coordinates are in range
def check_lat_long_in_range(d, lng_bounds, lat_bounds):
    lng_in_range = ((d.lng > lng_bounds[0]) & (d.lng < lng_bounds[1]))
    print(f"{sum(lng_in_range)} / {len(d.index)} have in range longitudes.")
    lat_in_range = ((d.lat > lat_bounds[0]) & (d.lat < lat_bounds[1]))
    print(f"{sum(lat_in_range)} / {len(d.index)} have in range latitudes.")
    assert lng_in_range.all()
    assert lat_in_range.all()

# Check that all lng/lat coordinates are in range 
LNG_BOUNDS = (-78,-73)
LAT_BOUNDS = (40, 45)

check_lat_long_in_range(d, LNG_BOUNDS, LAT_BOUNDS)
print_whole_screen_line()

# 

# function to print out columns with more than specified percent missing
def print_columns_with_more_than_threshold_missing(d, na_threshold):
    print(f"Dataset columns with > {na_threshold} proportion of missing images.")
    pprint(d.loc[:, d.isnull().mean() > na_threshold].isnull().mean())


# Missing data -- set threshold, print out columns with more than this percent missing 
NA_THRESHOLD = 0.025

print_columns_with_more_than_threshold_missing(d, NA_THRESHOLD)
print_whole_screen_line()



# 


# reformat all variables below into one dictionary 
# key: column name
# value: tuple of (lower bound, upper bound)
bounds = {
    'conf': (0,1),
    'distance_from_nearest_police_station': (0,50000),
    'distance_from_nearest_crime_1hr': (0,500000),
    'distance_from_nearest_crime_3hr': (0,500000),
    'distance_from_nearest_crime_6hr': (0,500000),
    'density_cbg': (0,10000000),
    'Estimate_Total': (0,10000000),
    'Estimate_Total_Not_Hispanic_or_Latino_White_alone': (0,10000000),
    'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone': (0,10000000),
    'Estimate_Total_Not_Hispanic_or_Latino_Asian_alone': (0,10000000),
    'Estimate_Total_Hispanic_or_Latino': (0,10000000),
    'Estimate_Total_Not_Hispanic_or_Latino': (0,10000000),
    'time_and_date_of_image': (datetime.datetime(2020,3,1,0,0,0,tzinfo=ZoneInfo('US/Eastern')), datetime.datetime(2020,11,16,0,0,0,tzinfo=ZoneInfo('US/Eastern'))),
    'hour': (0,23),
    'day_of_week': (0,6),
    'month': (1,12),
    'day_of_month': (1,31),
    'weekend': (0,1),
    'nighttime': (0,1),
}

# function to check that all values in specified columns are in range
def check_values_in_range(d, bounds):
    for col, (lower, upper) in bounds.items():
        in_range = ((d[col] >= lower) & (d[col] <= upper))
        print(f"{sum(in_range)} / {len(d.index)} have in range {col}.")
        assert in_range.all()


check_values_in_range(d, bounds)
print_whole_screen_line()



# 


core_anl_vars = ['distance_from_nearest_police_station','distance_from_nearest_crime_1hr','distance_from_nearest_crime_3hr','distance_from_nearest_crime_6hr',
                 'Estimate_Total_Not_Hispanic_or_Latino_White_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone','ntaname','time_and_date_of_image','hour','month','nighttime','day_of_month','day_of_week',
                'density_cbg','median_household_income','boroct2020']


# function to describe specific columns in dataset 
def describe_columns(d, cols):
    return d[cols].describe().apply(lambda s: s.apply('{0:.2f}'.format))

describe_columns(d, core_anl_vars)
print_whole_screen_line()

def describe_d(d): 
    return d.describe().apply(lambda s: s.apply('{0:.2f}'.format)).to_csv(f"{PAPER_GIT_REPO_PATH}/tables/describe_d.csv")

describe_d(d)


# ## 2. Loading in Validation & Test Sets, External Datasets

# ### Validation, Test Sets 

# 


v = pd.read_csv(VALSET_PATH)
t = pd.read_csv(TESTSET_PATH)


# 


vgdf = gpd.GeoDataFrame(v, geometry=gpd.points_from_xy(v.lng, v.lat), crs=WGS)
vgdf = vgdf.to_crs(PROJ_CRS)


# 


tgdf = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t.lng, t.lat), crs=WGS)
tgdf = tgdf.to_crs(PROJ_CRS)


# ### NYC Neighborhood Tabulation Areas (NTAs) 

# 


nyc_ntas = gpd.read_file("/share/pierson/nexar_data/5_other_datasets/nynta2020_22c")
nyc_ntas = nyc_ntas.to_crs(PROJ_CRS)


# 


nyc_ntas.plot()


# ### NYC Census Block Groups (CBGs) 

# 


ny_cbgs = gpd.read_file('/share/pierson/nexar_data/5_other_datasets/tl_2020_36_all/tl_2020_36_bg20.shp')
ny_cbgs = ny_cbgs.to_crs(WGS)

nyc_cbgs = ny_cbgs[ny_cbgs.COUNTYFP20.isin(NYC_COUNTY_CODES)]
nyc_cbgs.reset_index(inplace=True)
nyc_cbgs = nyc_cbgs.to_crs(PROJ_CRS)
nyc_cbgs.GEOID20 = pd.to_numeric(nyc_cbgs.GEOID20)
nyc_cbgs.plot()


# ### NYC Zoning Data 

# 


# Zoning Tests 
nyc_zoning = gpd.read_file("/share/pierson/nexar_data/5_other_datasets/nycgiszoningfeatures_202212shp")
nyc_zoning = nyc_zoning.to_crs('EPSG:2263')
def residential(z): 
    if 'R' in z:
        return True
    else:
        return False
    
def commercial(z): 
    if 'C' in z: 
        return True 
    else: 
        return False 

def manufacturing(z): 
    if 'M' in z:
        return True
    else: 
        return False 

def high_level_zoning(z): 
    if 'R' in z: 
        return 'R'
    elif 'C' in z:
        return 'C'
    elif 'M' in z:
        return 'M'
    
nyc_zoning['high_level_zone'] = nyc_zoning.ZONEDIST.map(lambda z: high_level_zoning(z))


nyc_zoning


# ### NYPD Precinct Locations 

# 


precincts = pd.read_csv("/share/pierson/nexar_data/5_other_datasets/nypd_precinct_locs.csv")
precincts_gdf = gpd.GeoDataFrame(precincts, geometry=gpd.points_from_xy(precincts.lng, precincts.lat), crs=WGS)
precincts_gdf = precincts_gdf.to_crs(PROJ_CRS)


# ### NYC Borough Boundaries (NYBB) 

# 


nybb = gpd.read_file(gpd.datasets.get_path('nybb'))
nybb = nybb.to_crs(PROJ_CRS)


# ## 3. NYPD Deployment Analysis 

# ### Computing Probability Measures with Validation Set 

# 


def calibrate_probabilities_using_valset(v, d_to_add_prediction_columns_to):
    """
    Annotate a dataframe, d_to_add_prediction_columns_to, with three prediction columns
    derived from the val set v.
    
    1. A simple binary variable with whether conf > POSITIVE_CLASSIFICATION_THRESHOLD
    2. A probabilistic prediction from val set: if above threshold, Pr(ground truth positive | above threshold in val set)
    and if below threshold, Pr(ground truth negative | below threshold in val set)
    3. A probability adjusted for police station distance. Not sure if this is a good thing to use, and should definitely check it is calibrated on test set if we do.
    """
    
    # 1. annotate with simple binary score
    assert v['Model_predicted_score'].isnull().sum() == 0
    v['classified_positive'] = v['Model_predicted_score'] > POSITIVE_CLASSIFICATION_THRESHOLD
    d_to_add_prediction_columns_to['above_threshold'] = (d_to_add_prediction_columns_to['conf'] > POSITIVE_CLASSIFICATION_THRESHOLD) * 1.

    # print the number of images > threshold 
    print("Number of images above threshold: %i" % (d_to_add_prediction_columns_to['above_threshold'].sum()))
    
    # 2. compute probabilities given above/below threshold from val set
    p_positive_given_classified_positive = v.loc[v['classified_positive'] == True, 'ground_truth'].mean()
    p_positive_given_classified_negative = v.loc[v['classified_positive'] == False, 'ground_truth'].mean()
    print("Fraction of val set classified positive: %2.3f (%i rows)" % 
          (v['classified_positive'].mean(), v['classified_positive'].sum()))
    print("Pr(true positive | classified positive): %2.3f" % p_positive_given_classified_positive)
    print("Pr(true positive | classified negative): %2.3f" % p_positive_given_classified_negative)
    d_to_add_prediction_columns_to['calibrated_prediction'] = d_to_add_prediction_columns_to['above_threshold'].map(lambda x:p_positive_given_classified_positive if x == 1 else p_positive_given_classified_negative) 
    
    # 3. compute adjusted probability given police station distance. Not sure if this is necessary or wise, but adding just in case. 
    police_station_distance_model = sm.Logit.from_formula('ground_truth ~ Model_predicted_score + distance_from_nearest_police_station', data=v).fit()
    print(police_station_distance_model.summary())
    d_to_add_prediction_columns_to['Model_predicted_score'] = 0 # compute police-distance adjusted probability on d_to_add_prediction_columns_to. 
    d_to_add_prediction_columns_to.loc[~pd.isnull(d_to_add_prediction_columns_to['conf']), 'Model_predicted_score'] = d_to_add_prediction_columns_to['conf'].loc[~pd.isnull(d_to_add_prediction_columns_to['conf'])]
    assert d_to_add_prediction_columns_to['Model_predicted_score'].isnull().sum() == 0
    d_to_add_prediction_columns_to['prediction_adjusted_for_police_station_distance'] = police_station_distance_model.predict(d_to_add_prediction_columns_to).values
    del d_to_add_prediction_columns_to['Model_predicted_score']
    
    added_cols = ['above_threshold', 'calibrated_prediction', 'prediction_adjusted_for_police_station_distance']
    assert pd.isnull(d_to_add_prediction_columns_to[added_cols]).values.sum() == 0
    assert (d_to_add_prediction_columns_to[added_cols].values < 0).sum() == 0
    assert (d_to_add_prediction_columns_to[added_cols].values > 1).sum() == 0
    for col in added_cols:
        print("Mean value of prediction column %s: %2.3f; std %2.3f; > 0 %2.3f" % (
            col,
            d_to_add_prediction_columns_to[col].mean(), 
            d_to_add_prediction_columns_to[col].std(), 
            (d_to_add_prediction_columns_to[col] > 0).mean()))
    
    return d_to_add_prediction_columns_to
    
d_for_demo_analysis = calibrate_probabilities_using_valset(v=v, d_to_add_prediction_columns_to=d)
print_whole_screen_line()


# PUT DGDF HERE 
# GeoDataFrame for d 
d_for_demo_analysis = gpd.GeoDataFrame(d_for_demo_analysis, geometry=gpd.points_from_xy(d_for_demo_analysis.lng, d_for_demo_analysis.lat), crs=WGS)
d_for_demo_analysis = d_for_demo_analysis.to_crs(PROJ_CRS)



# ### Geographic Aggregation 

# 


pop_by_nta = d_for_demo_analysis.groupby(['ntaname','NAME'])[POPULATION_COUNT_COLS + 
                                                              DEMOGRAPHIC_COLS + 
                                                             PREDICTION_COLS].agg('first')[["Estimate_Total_Not_Hispanic_or_Latino_White_alone","Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone"]].groupby(level='ntaname').agg('sum')
pop_by_nta

est_by_nta = d_for_demo_analysis.groupby(['ntaname'])[PREDICTION_COLS].agg('mean')
est_by_nta

nta_grouped_d = pop_by_nta.join(est_by_nta)
nta_grouped_d = (nta_grouped_d, )


# 


# group by Census area. 
grouped_d = d_for_demo_analysis.groupby(LOCATION_COL_TO_GROUP_ON)[POPULATION_COUNT_COLS + 
                                                              DEMOGRAPHIC_COLS + 
                                                              PREDICTION_COLS].agg('mean')
for col in POPULATION_COUNT_COLS + DEMOGRAPHIC_COLS: # check consistent values by location for demographics. Should only be one value of population count per Census area, for example. 
    if col in ['distance_from_nearest_crime_6hr', 'distance_from_nearest_police_station']:
        continue
    #assert d_for_demo_analysis.groupby(LOCATION_COL_TO_GROUP_ON)[col].nunique().map(lambda x:x in [0, 1]).all()

print("%i unique Census areas using column %s" % (len(grouped_d), LOCATION_COL_TO_GROUP_ON))
print("Population statistics by area")
print(grouped_d[TOTAL_POPULATION_COL].describe([0.01, 0.05, 0.1, 0.5, 0.9, 0.99]))
print("excluding census areas with population < %i keeps fraction %2.3f of population" % 
      (MIN_POPULATION_IN_AREA, 
       grouped_d.loc[grouped_d[TOTAL_POPULATION_COL] >= MIN_POPULATION_IN_AREA, TOTAL_POPULATION_COL].sum()/grouped_d[TOTAL_POPULATION_COL].sum()))
for col in POPULATION_COUNT_COLS: # sanity check that total counts look right. 
    print("summed values of %s: %i" % (col, grouped_d[col].sum()))
    
print_whole_screen_line()


# ### Disparities Estimator 
def weighted_disparities_estimator(df, census_area_col, weighting_cols, total_population_col, estimate_col, check_consistent_vals_by_group):
    """
    Given a census dataframe
    group by census_area_col and compute the mean value of estimate_col in each census area
    then return weighted means across Census areas, 
    weighting each Census area by all the columns in weighting_cols and total_population_col. 
    We use this for computations of race-specific results. 
    Emma reviewed. 
    """
    grouped_d = df.groupby(census_area_col)[weighting_cols + [total_population_col, estimate_col]].mean().reset_index()
    if check_consistent_vals_by_group: # sanity check to make sure that values are consistent
        consistency_df = df.groupby(census_area_col)[weighting_cols + [total_population_col]].nunique()
        assert (consistency_df.values == 1).all()
        assert df[weighting_cols + [total_population_col, estimate_col]].isnull().values.sum() == 0
    results = {}
    for col in weighting_cols + [total_population_col]:
        results['%s_weighted_mean' % col] = (grouped_d[estimate_col] * grouped_d[col]).sum()/grouped_d[col].sum()
    for col in weighting_cols:
        results['%s_relative_to_average' % col] = results['%s_weighted_mean' % col]/results['%s_weighted_mean' % total_population_col]
    return results

def weighted_disparities_estimator_two_level_grouping(df, census_area_col, high_level_group_col, total_population_col, estimate_col, check_consistent_vals_by_group):
    """
    This function is similar to that above, but is (hopefully) a faster way to compute 
    two-level groupings (e.g., we want to compute borough-specific numbers, and weight by Census tract population within borough). 
    high_level_group col specifies the column we want to compute disparities over (e.g. borough). 
    All other columns are as explained above. 
    Verified that this gives identical results to function above for borough, zone, etc. 
    """
    if check_consistent_vals_by_group: # sanity check to make sure that values are consistent
        consistency_df = df.groupby(census_area_col)[high_level_group_col].nunique()
        assert ((consistency_df.values == 1) | (consistency_df.values == 0)).all()
    results = {}
    # first compute overall mean. 
    overall_mean_grouping = df.groupby(census_area_col)[[total_population_col, estimate_col]].mean().reset_index()
    results['%s_weighted_mean' % total_population_col] = (overall_mean_grouping[estimate_col] * overall_mean_grouping[total_population_col]).sum()/overall_mean_grouping[total_population_col].sum()
    high_level_grouping = df.groupby(high_level_group_col)
    all_names = []
    for name, group_df in high_level_grouping:
        if group_df[total_population_col].sum() == 0:
            print("Skipping %s because total population is 0" % name)
            continue
        all_names.append(name)
        second_level_grouping = group_df.groupby(census_area_col)[[total_population_col, estimate_col]].mean().reset_index()
        results['%s_weighted_mean' % name] = (second_level_grouping[estimate_col] * second_level_grouping[total_population_col]).sum()/second_level_grouping[total_population_col].sum()
    for name in all_names:
        results['%s_relative_to_average' % name] = results['%s_weighted_mean' % name]/results['%s_weighted_mean' % total_population_col]
    return results

def bootstrap_function_errorbars(df, fxn_to_apply, fxn_kwargs, n_bootstraps=100, filename=None):
    # compute the point estimate fxn_to_apply(df) on the original data
    # and then do bootstrap iterates. Emma reviewed. 
    bootstrap_statistics = []
    point_estimate = fxn_to_apply(df, check_consistent_vals_by_group=True, **fxn_kwargs)
    for bootstrap in tqdm(range(n_bootstraps)):
            bootstrap_df = df.sample(frac=1, replace=True)
            bootstrap_statistics.append(fxn_to_apply(bootstrap_df, check_consistent_vals_by_group=False, **fxn_kwargs))
    if filename is not None:
        with open(f"{PAPER_GIT_REPO_PATH}/{filename}", 'w') as f:
            json.dump({'point_estimate':point_estimate, 'bootstrap_statistics':bootstrap_statistics}, f)
    return point_estimate, bootstrap_statistics

# print out a table. Emma reviewed. 
def create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics):
    bootstrap_statistics = pd.DataFrame(bootstrap_statistics)
    bootstrap_results_table = []
    for k in bootstrap_point_estimate.keys():
        lower_CI = np.percentile(bootstrap_statistics[k], 2.5)
        upper_CI = np.percentile(bootstrap_statistics[k], 97.5)
        bootstrap_results_table.append({'quantity':k, 
                                        'estimate':bootstrap_point_estimate[k], 
                                        #'bootstrap std':bootstrap_statistics[k].std(), 
                                        #'2.5% percentile':lower_CI, 
                                        #'97.5% percentile':upper_CI, 
                                        'percentile CI':'%2.3f (%2.3f, %2.3f)' % (bootstrap_point_estimate[k], 
                                                                                         lower_CI, 
                                                                                         upper_CI), 
                                       '1.96 sd CI':'%2.3f +/- %2.3f' % (bootstrap_point_estimate[k], 1.96 * bootstrap_statistics[k].std())})
    bootstrap_results_table = pd.DataFrame(bootstrap_results_table)
    return (bootstrap_results_table.loc[bootstrap_results_table['quantity'].map(lambda x:'relative_to_average' in x), 
                                       ['quantity', 'estimate', 'percentile CI', '1.96 sd CI']].sort_values(by='estimate')[::-1])


# race table (not conditioning on Zone). Emma reviewed. 
# compute point estimate and errorbars
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'weighting_cols':[WHITE_POPULATION_COL, BLACK_POPULATION_COL, HISPANIC_POPULATION_COL, ASIAN_POPULATION_COL], 
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='race_bootstraps.json')
print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics).to_string())

print_whole_screen_line()

# race table CONDITIONING ON ZONE. Emma reviewed. 
# compute point estimate and errorbars
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(
    df=d_for_demo_analysis.loc[d_for_demo_analysis['zone'] == 'R'], 
                             fxn_to_apply=weighted_disparities_estimator, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'weighting_cols':[WHITE_POPULATION_COL, BLACK_POPULATION_COL, HISPANIC_POPULATION_COL, ASIAN_POPULATION_COL], 
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='race_residential_zones_only_bootstraps.json')
print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics).to_string())

print_whole_screen_line()

# continuous variables: divide into quartile. Emma reviewed. 

# density_cbg, median_household_income

for col in ['median_household_income', 'density_cbg']:
    percentile_cutoffs = [25, 50, 75]
    print("Fraction of missing values for %s: %2.6f" % (col, d_for_demo_analysis[col].isnull().mean()))
    d_for_col = d_for_demo_analysis.dropna(subset=[col]).copy()
    cutoff_vals = np.percentile(d_for_col[col], percentile_cutoffs)
    cutoff_vals = [-np.inf] + list(cutoff_vals) + [np.inf]
    print('cutoffs for %s' % col, cutoff_vals)
    d_for_col['%s_quartile' % col] = None

    for i in range(len(cutoff_vals) - 1):
        quartile_idxs = d_for_col[col].map(lambda x:(x >= cutoff_vals[i]) & (x < cutoff_vals[i + 1]))
        d_for_col.loc[quartile_idxs, '%s_quartile' % col] = '%s_quartile_%i' % (col, i + 1)
        print('number of rows in %s: %i' % ('%s_quartile' % col, quartile_idxs.sum()))
    bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_col, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':'%s_quartile' % col,
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='%s_bootstraps.json' % col)

    print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics).to_string())

    print_whole_screen_line()
    
# zone table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':'zone',
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='zone_bootstraps.json')

print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics).to_string())

print_whole_screen_line()

# boro table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':'boroname',
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='boro_bootstraps.json')

print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics).to_string())

print_whole_screen_line()

# neighborhood table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':NEIGHBORHOOD_COL,
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='neighborhood_bootstraps.json')

print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics).to_string())


for prediction_col in PREDICTION_COLS:
    print("Using prediction col", prediction_col)
    estimates = {}
    for demo_col in POPULATION_COUNT_COLS:
        if demo_col == TOTAL_POPULATION_COL:
            continue
        # compute weighted mean as described in Census tract. 
        grouped_mean = (grouped_d[prediction_col] * grouped_d[demo_col]).sum()/grouped_d[demo_col].sum()
       
        print(demo_col, grouped_mean)
        estimates[demo_col] = grouped_mean
    print("Ratio of Black estimate to white estimate: %2.3f" % (estimates[BLACK_POPULATION_COL]/estimates[WHITE_POPULATION_COL]))

print_whole_screen_line()

# ## 4. Analysis Plots 

# ### Correlations Between All Measures 

# Pearson Correlation Coefficient
pearson_corr = grouped_d.loc[grouped_d[TOTAL_POPULATION_COL] > MIN_POPULATION_IN_AREA, DEMOGRAPHIC_COLS].corr(method='pearson')
mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
heatmap = sns.heatmap(pearson_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Pearson Correlations', fontdict={'fontsize':18}, pad=16);
plt.savefig(f"{PAPER_GIT_REPO_PATH}/pearson_correlations.png", bbox_inches='tight')

# 


# Spearman Correlation Coefficient
spearman_corr = grouped_d.loc[grouped_d[TOTAL_POPULATION_COL] > MIN_POPULATION_IN_AREA, DEMOGRAPHIC_COLS].corr(method='spearman')
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
heatmap = sns.heatmap(spearman_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Spearman Correlations', fontdict={'fontsize':18}, pad=16);
plt.savefig(f"{PAPER_GIT_REPO_PATH}/spearman_correlations.png", bbox_inches='tight')

# ### Breakdown by Neighborhood 


for col in PREDICTION_COLS:
    print("\n\nneighborhoods with highest mean values of %s" % col)
    print(d_for_demo_analysis
          .groupby(NEIGHBORHOOD_COL)[col]
          .mean()
          .reset_index()
          .sort_values()[::-1])
nta_breakdown = d_for_demo_analysis.groupby(NEIGHBORHOOD_COL)['calibrated_prediction'].agg(['mean', 'size']).reset_index().sort_values(by='mean')[::-1]
print_whole_screen_line()


# ## Map of Pr(police) by Census area (either neighborhood or Census tract). Can show this next to maps of density and potentially other variables - we decided to make this by neighborhood for a number of reasons. 

# ### Plot Data Loaded from Bootstraps 

# 


import json
nta_data = json.load(open(f"{PAPER_GIT_REPO_PATH}/neighborhood_bootstraps.json"))


# 


nta_rta_means = pd.DataFrame.from_dict(nta_data['point_estimate'], orient='index')
nta_rta_means = nta_rta_means[nta_rta_means.index.map(lambda x: 'relative_to_average' in x)]
nta_rta_means.index = nta_rta_means.index.str.replace('_relative_to_average','')
nta_rta_means.columns = ['Pr_police_rta']
nta_rta_means


# 


nyc_ntas = nyc_ntas.merge(nta_rta_means, left_on='NTAName', right_index=True, how='left')


# 


bins = [0, 0.25, 0.5, 0.75, 1, 2, 3,  4, 5]
#labels = ['( 0, 0.5 ]', '( 0.5, 1 ]', '( 1, 2 ]', '> 2']
nyc_ntas['Pr_police_rta'].fillna(0, inplace=True)
nyc_ntas['pr_quantile'] = pd.cut(nyc_ntas['Pr_police_rta'], bins)



fig, ax = plt.subplots(figsize=(24,24))
colormap = 'bwr'
nybb.plot(ax=ax, color='gainsboro', edgecolor='grey')
nyc_ntas.plot(column='pr_quantile', ax=ax, cmap=colormap, legend=False)#, legend_kwds={'loc': 'upper left', 'title': 'Police deployment\n(relative to city average)', 'ncols':2, 'fontsize':50, 'markerscale':3, 'title_fontsize':60, 'alignment':'center'})

n = 16 # how many lines to draw or number of discrete color levels

cmap = plt.get_cmap(colormap)

norm = matplotlib.colors.Normalize(vmin=0, vmax=5)
stretched_bounds = np.interp(np.linspace(0, 1, 257), np.linspace(0, 1, 17), [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
# normalize stretched bound values
norm = matplotlib.colors.BoundaryNorm(stretched_bounds, ncolors=256)
scalarmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])
cb = plt.colorbar(scalarmap, ticks=np.arange(0,n), ax=ax, orientation='horizontal', ticklocation='bottom', pad=0, anchor=(0.2,6), shrink=0.35)
cb.ax.xaxis.set_label_position('top')
cb.set_label(label='Police deployment\n(relative to city average)',size=40,weight='bold', labelpad=20)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1, 2, 3,  4, 5], labels=[r'${0\scriptstyle\times}$', r'${0.25\scriptstyle\times}$', r'${0.5\scriptstyle\times}$', r'${0.75\scriptstyle\times}$', r'${1\scriptstyle\times}$', r'${2\scriptstyle\times}$', r'${3\scriptstyle\times}$', r'${4\scriptstyle\times}$', r'${5\scriptstyle\times}$'],size=25)


plt.axis('off')
plt.tight_layout()
plt.savefig(f'{PAPER_GIT_REPO_PATH}/figures/Pr_police_rta_ntas.pdf')


# ### VARIANT: Un-bootstrapped Data 
fig, ax = plt.subplots(figsize=(12,12))

try: 
    d_for_demo_analysis.drop(['index_right'],axis=1,inplace=True)

except Exception as e:
    print(e)
    pass 


try: 
    d_for_demo_analysis.drop(['index'],axis=1,inplace=True)

except Exception as e:
    print(e)
     

try: 
    d_for_demo_analysis.drop(['index_left'],axis=1,inplace=True)

except Exception as e:
    print(e)


dgdf_for_demo_by_ntas = gpd.sjoin(nyc_ntas, d_for_demo_analysis, how='left', predicate='contains').groupby('NTAName').agg('mean','size')

nyc_ntas_proj_demo = nyc_ntas.merge(dgdf_for_demo_by_ntas,left_on='NTAName',right_on='NTAName')
nyc_ntas_proj_demo.calibrated_prediction.fillna(0, inplace=True)
nyc_ntas_proj_demo.calibrated_prediction_decile = pd.qcut(nyc_ntas_proj_demo.calibrated_prediction, 5, duplicates='drop')
print(nyc_ntas_proj_demo.calibrated_prediction.describe())

nybb.plot(ax=ax, edgecolor='grey', color='w')
nyc_ntas_proj_demo.plot(column=nyc_ntas_proj_demo.calibrated_prediction_decile, ax=ax, cmap='cividis', legend=True, legend_kwds={'title':'Calibrated Probability of Police Exposure'})

plt.axis('off')
plt.tight_layout()
plt.savefig(f"{PAPER_GIT_REPO_PATH}/figures/Pr_police_by_nta_point_estimates.pdf", dpi=450)


# ### VARIANT2: More granular, exposure by CBG 
dgdf_for_demo_by_cbgs = gpd.sjoin(nyc_cbgs, d_for_demo_analysis, how='left', predicate='contains')
groupby_cols = ['index']

# aggregate via mean, but only for numeric columns 
agg_function = {
    "number": 'mean', 
    "object": lambda col: col.mode()
    }
agg_dct = {k: v for i in [{col: agg for col in dgdf_for_demo_by_cbgs.select_dtypes(tp).columns.difference(groupby_cols)} for tp, agg in agg_function.items()] for k, v in i.items()}

dgdf_for_demo_by_cbgs = dgdf_for_demo_by_cbgs.groupby(groupby_cols).agg(**{k: (k, v) for k, v in agg_dct.items()})

# 


fig, ax = plt.subplots(figsize=(12,12))

nyc_cbgs_proj_demo = nyc_cbgs.merge(dgdf_for_demo_by_cbgs,left_on='GEOID20',right_on='GEOID20_left')
nyc_cbgs_proj_demo.calibrated_prediction.fillna(0, inplace=True)
nyc_cbgs_proj_demo.calibrated_prediction_decile = pd.qcut(nyc_cbgs_proj_demo.calibrated_prediction, 18, duplicates='drop')
print(nyc_cbgs_proj_demo.calibrated_prediction_decile)
nyc_cbgs_proj_demo.plot(column=nyc_cbgs_proj_demo.calibrated_prediction_decile, ax=ax, cmap='cividis', legend=True)#,legend_kwds={"location": "bottom", "shrink": 0.5, "pad": 0, 'label': 'Calibrated Probability of Police Exposure'})

plt.axis('off')

plt.savefig(f'{PAPER_GIT_REPO_PATH}/figures/PR_police_by_cbg_point_estimates.pdf', dpi=450)

# ## 5. Model Development & Evaluation Plots
# Metrics 
pandr_vt = pd.DataFrame()
p = []
r = []
rows = ['Validation Set', 'Test Set']
for s in [v, t]: 


    tp = (s['Model_predicted_score'] >= POSITIVE_CLASSIFICATION_THRESHOLD) & (s['ground_truth'] == 1)
    fp = (s['Model_predicted_score'] >= POSITIVE_CLASSIFICATION_THRESHOLD) & (s['ground_truth'] == 0)

    fn = (s['Model_predicted_score'] < POSITIVE_CLASSIFICATION_THRESHOLD) & (s['ground_truth'] == 1)
    tn = (s['Model_predicted_score'] < POSITIVE_CLASSIFICATION_THRESHOLD) & (s['ground_truth'] == 0)
    
    p.append( tp.sum() / (tp.sum() + fp.sum()))
    r.append( tp.sum() / (tp.sum() + fn.sum()))
    
    

pandr_vt['Precision'] = p
pandr_vt['Recall'] = r
pandr_vt.index = rows 

pandr_vt.to_latex(f"{PAPER_GIT_REPO_PATH}/tables/pandr_vt.tex", float_format="%.2f")


# 


from sklearn import metrics
perf_stats = pd.DataFrame() 
p = []
r = []
auc = []
ap = []

rows = ['Validation Set', 'Test Set']

for s in [v, t]: 
    
    tp = (s['Model_predicted_score'] >= POSITIVE_CLASSIFICATION_THRESHOLD) & (s['ground_truth'] == 1)
    fp = (s['Model_predicted_score'] >= POSITIVE_CLASSIFICATION_THRESHOLD) & (s['ground_truth'] == 0)

    fn = (s['Model_predicted_score'] < POSITIVE_CLASSIFICATION_THRESHOLD) & (s['ground_truth'] == 1)
    tn = (s['Model_predicted_score'] < POSITIVE_CLASSIFICATION_THRESHOLD) & (s['ground_truth'] == 0)
    
    p.append( tp.sum() / (tp.sum() + fp.sum()))
    r.append( tp.sum() / (tp.sum() + fn.sum()))
    
    auc.append(metrics.roc_auc_score(y_true=s['ground_truth'], y_score=s['Model_predicted_score']))
    ap.append(metrics.average_precision_score(y_true=s['ground_truth'], y_score=s['Model_predicted_score']))

              
perf_stats['Precision'] = p
perf_stats['Recall'] = r
perf_stats['AUC'] = auc 
perf_stats['AP'] = ap
perf_stats.index = rows 
              
perf_stats.to_latex(f"{PAPER_GIT_REPO_PATH}/tables/performance_vt.tex", float_format="%.2f")


# ### Combined AUC / AUPRC Plot 

# 


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

fig, (axauc, axprc) = plt.subplots(1, 2, figsize=(14,6))
#plt.style.use('seaborn-v0_8-paper')
RocCurveDisplay.from_predictions(
    t.ground_truth,
    t.Model_predicted_score,
    name="Police Vehicle",
    color="green", ax=axauc
)
axauc.plot([0, 1], [0, 1], "k--")
axauc.axis("square")
axauc.set_xlabel("False Positive Rate", fontsize=20)
axauc.set_ylabel("True Positive Rate", fontsize=20)

#plt.title("ROC Curve")
axauc.legend(prop={'size': 20})
#plt.show()

axauc.set_xlim(0,1)
axauc.set_ylim(0,1)

axauc.tick_params(axis='both', which='major', labelsize=14)
axauc.tick_params(axis='both', which='minor', labelsize=14)

#plt.style.use('seaborn-v0_8-paper')
PrecisionRecallDisplay.from_predictions(
    t.ground_truth,
    t.Model_predicted_score,
    name="Police Vehicle",
    color="purple", ax=axprc
)

axauc.set_title("AUC", fontdict = {'fontsize': 24})

axprc.axis("square")
axprc.set_xlabel("Recall", fontsize=20)
axprc.set_ylabel("Precision", fontsize=20)
#plt.title("ROC Curve")
axprc.legend(prop={'size': 20})
#plt.show()

axprc.set_xlim(0,1)
axprc.set_ylim(0,1)

axprc.tick_params(axis='both', which='major', labelsize=14)
axprc.tick_params(axis='both', which='minor', labelsize=14)

axprc.set_title("AUPRC", fontdict = {'fontsize': 24})

plt.savefig(f"{PAPER_GIT_REPO_PATH}/figures/test_auc_auprc.pdf")


# ### Tuning: Additional Feature on Calibration Plot for In/Out of Manhattan

# 


tgdf_with_boro = gpd.sjoin(tgdf, nybb)


# 


t["Manhattan"] = tgdf_with_boro['BoroName'] == "Manhattan"


# 


t["Manhattan"].describe()


# ### Calibration Plots 

# 


if 'ground_truth' not in t.columns:
    t['ground_truth'] = 0
if 'Model_predicted_score' not in t.columns:
    t['Model_predicted_score'] = 0
else:
    # simple correction model for distance from nearest police station. 
    police_station_model = sm.Logit.from_formula('ground_truth ~ distance_from_nearest_police_station + Model_predicted_score', data=t).fit()
    print(police_station_model.summary())
    t['police_distance_adjusted_score'] = police_station_model.predict(t)

t.loc[t['median_household_income'] == '250,000+', 'median_household_income']  = 250000
t.loc[t['median_household_income'] == '2,500-', 'median_household_income']  = 2500

t.loc[t['median_household_income'] == '-', 'median_household_income'] = None
t['median_household_income'] = t['median_household_income'].astype(float)
t.describe()
print_whole_screen_line()


t['classified_positive'] = t.Model_predicted_score > 0.86 # what is our threshold for a positive classification
print('p(Police | classified_positive): %2.3f (precision)' % t.loc[t['classified_positive'] == 1, 'ground_truth'].mean())
print('p(Police | classified_negative): %2.3f (forget what this metric is called)' % t.loc[t['classified_positive'] == 0, 'ground_truth'].mean())
print('p(classified_positive | police): %2.3f (recall)' % t.loc[t['ground_truth'] == 1, 'classified_positive'].mean())
print_whole_screen_line()


results=[]
vars_to_plot = ['phase','weekend','daytime', 'percent_white', 'percent_black', 'percent_asian', 'percent_hispanic', 'median_household_income', 'pplpersqmi', 'distance_from_nearest_crime_1hr', 'distance_from_nearest_crime_3hr', 'distance_from_nearest_crime_6hr', 'distance_from_nearest_police_station', 'Manhattan']
vars_to_plot_names = ['Phase 1', 'Weekend', 'Daytime', 'Percent White > Median', 'Percent Black > Median', 'Percent Asian > Median', 
                      'Percent Hispanic > Median', 'Median Household Income > Median', 'Population Density > Median', 'Distance From Nearest Crime [1hr] > Median', 'Distance From Nearest Crime [3hr] > Median', 'Distance From Nearest Crime [6hr] > Median', 'Distance From Nearest Police Station > Median', 'Manhattan']
for x in vars_to_plot:
    df_to_plot = t.dropna(subset=x).copy()
    if df_to_plot[x].median() == (1 | 0): 
        df_to_plot["above_median"] = df_to_plot[x] == 1
    else:
        df_to_plot['above_median'] = df_to_plot[x] > df_to_plot[x].median()
    
    for above_median in [True, False]:
        for classified_positive in [True, False]:
            idxs = (df_to_plot['above_median'] == above_median) & (df_to_plot['classified_positive'] == classified_positive)
            mu = df_to_plot.loc[idxs, 'ground_truth'].mean() # probability of a police car given whether you're above the median and whether you're classified positive. 
            err = 1.96 * np.sqrt(mu * (1 - mu) / idxs.sum()) # confidence interval: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
            results.append({'x':x, 
                            'above_median':above_median, 
                            'classified_positive':classified_positive,
                            'mean':mu, 
                            'error':err})
results = pd.DataFrame(results)


fig, (axp, axn) = plt.subplots(1,2, figsize=(10,4), sharey='row')



# positive classifications plot
axp.errorbar(y=range(len(vars_to_plot)),
             x=results.loc[(results['classified_positive'] == True) & (results['above_median'] == True), 'mean'], 
             xerr=results.loc[(results['classified_positive'] == True) & (results['above_median'] == True), 'error'], 
             label='True', 
             fmt='.', 
             markersize=10)
axp.errorbar(y=[a + 0.1 for a in range(len(vars_to_plot))],
             x=results.loc[(results['classified_positive'] == True) & (results['above_median'] == False), 'mean'], 
             xerr=results.loc[(results['classified_positive'] == True) & (results['above_median'] == False), 'error'], 
             label='False', 
             fmt='.', 
             markersize=10)
#axp.legend(loc='center left')
axp.set_yticks(range(len(vars_to_plot)), vars_to_plot_names, fontsize=12)
axp.set_xlabel("Probability Image Truly has a Police Car", fontsize=12)
axp.set_title("Images Classified Positive", fontsize=12)

# negative classifications plot 
axn.errorbar(y=range(len(vars_to_plot)),
             x=results.loc[(results['classified_positive'] == False) & (results['above_median'] == True), 'mean'], 
             xerr=results.loc[(results['classified_positive'] == False) & (results['above_median'] == True), 'error'], 
             label='True', 
             fmt='.', 
             markersize=10)
axn.errorbar(y=[a + 0.1 for a in range(len(vars_to_plot))],
             x=results.loc[(results['classified_positive'] == False) & (results['above_median'] == False), 'mean'], 
             xerr=results.loc[(results['classified_positive'] == False) & (results['above_median'] == False), 'error'], 
             label='False', 
             fmt='.', 
             markersize=10)


axn.legend(loc='center right')
#axn.set_yticks(range(len(vars_to_plot)), '')
axn.set_xlabel("Probability Image Truly has a Police Car", fontsize=12)
axn.set_title("Images Classified Negative", fontsize=12)


plt.tight_layout()

plt.savefig(f"{PAPER_GIT_REPO_PATH}/figures/calplots.pdf")


# ### AUC/ AUPRC By Subgroups Table 

# 


from sklearn import metrics

# AUC/AUPRC by subgroup as well. 
auc_auprc_results = []
for x in vars_to_plot:
    df_to_plot = t.dropna(subset=x).copy()

    if df_to_plot[x].median() == (1 | 0): 
        df_to_plot["above_median"] = df_to_plot[x] == 1
    else:
        df_to_plot['above_median'] = df_to_plot[x] > df_to_plot[x].median()
    
    for above_median in [True, False]:
        
        auc = metrics.roc_auc_score(y_true=df_to_plot.loc[df_to_plot['above_median'] == above_median,'ground_truth'], 
                                    y_score=df_to_plot.loc[df_to_plot['above_median'] == above_median,'Model_predicted_score'])

        average_precision = metrics.average_precision_score(
            y_true=df_to_plot.loc[df_to_plot['above_median'] == above_median,'ground_truth'], 
            y_score=df_to_plot.loc[df_to_plot['above_median'] == above_median,'Model_predicted_score'])
        auc_auprc_results.append({'x':x, 
                                  'above_median':above_median, 
                                  'auc':auc, 
                                  'auprc':average_precision, 
                                  'n':(df_to_plot['above_median'] == above_median).sum(), 
                                  'n_pos':df_to_plot.loc[df_to_plot['above_median'] == above_median,'ground_truth'].sum()})

#pd.DataFrame(auc_auprc_results).to_csv(f"valset_{os.path.splitext(FILENAME)[0][-1]}_auc_by_subgroup.csv",index=False)

auc_auprc_results_table = pd.DataFrame(auc_auprc_results)

rename = {'x': 'Subgroup', 'above_median': 'Above Median?', 'auc': 'AUC', 'auprc': 'Average Precision'}

auc_auprc_results_table.rename(columns=rename, inplace=True)
auc_auprc_results_table.drop(['n','n_pos'], axis=1, inplace=True)

auc_auprc_results_table = auc_auprc_results_table.groupby(['Subgroup', 'Above Median?'], sort=False).sum().unstack(level=1)

subgroups_print = {'day_of_month': 'Day of Month', 'daytime': 'Daytime', 'distance_from_nearest_crime_1hr': 'Distance From Nearest Crime [1hr] > Median', 'distance_from_nearest_crime_3hr': 'Distance From Nearest Crime [3hr] > Median', 'distance_from_nearest_crime_6hr': 'Distance From Nearest Crime [6hr] > Median', 'distance_from_nearest_police_station': 'Distance From Nearest Police Station > Median', 'median_household_income': 'Median Household Income > Median', 'month': 'Month', 'percent_black': 'Percent Black > Median', 'percent_white': 'Percent White > Median', 'percent_hispanic': 'Percent Hispanic > Median', 'percent_asian': 'Percent Asian > Median', 'phase': 'Phase 1', 'pplpersqmi': 'Population Density > Median', 'weekend': 'Weekend', 'Manhattan': 'Manhattan'}
vars_to_plot_names = ['Phase 1', 'Weekend', 'Daytime', 'Percent White > Median', 'Percent Black > Median', 'Percent Asian > Median', 
                      'Percent Hispanic > Median', 'Median Household Income > Median', 'Population Density > Median', 'Distance From Nearest Crime [1hr] > Median', 'Distance From Nearest Crime [3hr] > Median', 'Distance From Nearest Crime [6hr] > Median', 'Distance From Nearest Police Station > Median']
auc_auprc_results_table.index = auc_auprc_results_table.index.map(subgroups_print)
auc_aurpc_results_table = auc_auprc_results_table.iloc[::-1]
auc_auprc_results_table.to_latex(f'{PAPER_GIT_REPO_PATH}/tables/auc_auprc_results_by_subgroup.tex', float_format="%.2f") 

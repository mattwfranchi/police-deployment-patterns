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
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import pearsonr, spearmanr
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from scipy.stats import bootstrap
from tqdm.auto import tqdm
tqdm.pandas()
import seaborn as sns 
import json 

# ## (Optional) Enable LaTeX font rendering 

# In[3]:


# Requires local LaTeX installation 
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rcParams['text.usetex'] = True
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage[T1]#!/usr/bin/env python {fontenc}')
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
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import pearsonr, spearmanr
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from scipy.stats import bootstrap
from tqdm.auto import tqdm
tqdm.pandas()
import seaborn as sns 
import json 

# ## (Optional) Enable LaTeX font rendering 

# In[3]:


# Requires local LaTeX installation 
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rcParams['text.usetex'] = True
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage[T1]{fontenc}')


# ## Global Constants 

# ### I/O Paths

# In[4]:


ANL_DATASET_PATH = "../../nypd-deployment-patterns-old/output/analysis_dataset.csv"
FIRST_CHUNK_PATH = "../../nypd-deployment-patterns-old/output/1603771200000.csv"
VALSET_PATH = "../valset.csv"
TESTSET_PATH = "../testset.csv"
PAPER_GIT_REPO_PATH = "../plots"


# ### Geographic 

# In[5]:


WGS = 'EPSG:4326'
PROJ_CRS = 'EPSG:2263'
NYC_COUNTY_CODES = ['005', '047', '061', '081', '085']


# ### Analysis Parameters 

# In[6]:

N_BOOTSTRAPS = 100
BOOTSTRAP_OUTPUT_DIR='../output/bootstraps'
ZONE_THRESHOLD = 0.5 # threshold for zone classification
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
                    'black_frac',
                    'white_frac', 
                    'distance_from_nearest_crime_6hr',
                    'distance_from_nearest_police_station',
                    'median_household_income']
PREDICTION_COLS = ['above_threshold', 'calibrated_prediction', 'prediction_adjusted_for_police_station_distance'] # columns with police car predictions. We define these
MIN_POPULATION_IN_AREA = 500
BOROUGH_COL = 'boroname'
NEIGHBORHOOD_COL = 'ntaname'


# ## 1. Dataset Verification 

# In[7]:

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


d = load_analysis_dataset(use_pyarrow=True)



# In[8]:


d.head()


# ### Preprocessing

# In[9]:

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


# In[10]:

# function to remove duplicates from dataset bsaed on specific columns 
def remove_duplicates(d, cols_to_deduplicate_on):
    duplicate_idxs = d.duplicated(subset=cols_to_deduplicate_on)
    print("warning: %i duplicates identified using %s, fraction %2.6f of rows; dropping rows" % (duplicate_idxs.sum(), cols_to_deduplicate_on, duplicate_idxs.mean()))
    d = d.loc[~duplicate_idxs].copy()
    return d

d = remove_duplicates(d, COLS_TO_DEDUPLICATE_ON)

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

# function to filter dataset for dates with full coverage 
def filter_for_dates_with_full_coverage(d, min_date_for_full_coverage):
    len_before_filter = len(d.index)
    print("In demographic analysis, filtering for locations after %s because more geographically representative" % MIN_DATE_FOR_DEMOGRAPHIC_ANALYSIS)
    d = d.loc[d['time_and_date_of_image'] >= min_date_for_full_coverage].copy()
    print("%i/%i rows remaining" % (len(d), len_before_filter))
    return d

d = filter_for_dates_with_full_coverage(d, MIN_DATE_FOR_DEMOGRAPHIC_ANALYSIS)



# In[11]:

# function to fill in NA data in specified columns 
def fill_na_data(d, cols_to_fill_na):
    for col in cols_to_fill_na:
        d[col].fillna(0, inplace=True)
    return d


COLS_TO_FILL_NA = ['conf', 'distance_from_nearest_police_station', 'distance_from_nearest_crime_1hr', 'distance_from_nearest_crime_3hr', 'distance_from_nearest_crime_6hr', 'density_cbg', 'Estimate_Total', 'Estimate_Total_Not_Hispanic_or_Latino_White_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Asian_alone', 'Estimate_Total_Hispanic_or_Latino']
d = fill_na_data(d, COLS_TO_FILL_NA)


cbg_zone_data = pd.read_csv('../../nypd-deployment-patterns-old/external_datasets/cbgs_zone_data.csv')
# function to fill in zone data for each image 
def fill_in_zone_data(d, zone_data):
    
    assert (1.*(cbg_zone_data['C'] > ZONE_THRESHOLD) + 1.*(cbg_zone_data['M'] > ZONE_THRESHOLD) + 1.*(cbg_zone_data['R'] > ZONE_THRESHOLD)).max() == 1
    cbg_zone_dict = {}
    for zone_val in ['C', 'M', 'R']:
        zones = cbg_zone_data.loc[cbg_zone_data[zone_val] >= ZONE_THRESHOLD]
        print("%i CBGs classified as %s" % (len(zones), zone_val))
        cbg_zone_dict.update(dict(zip(zones['GEOID20'].values, [zone_val for _ in range(len(zones))])))
    print(len(cbg_zone_dict))
    d['zone'] = d['GEOID20'].map(lambda x:cbg_zone_dict[x] if x in cbg_zone_dict else None)

    return d

d = fill_in_zone_data(d, cbg_zone_data)

# ### Sanity Checks

# In[12]:

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

# In[13]:

# function to print out columns with more than specified percent missing
def print_columns_with_more_than_threshold_missing(d, na_threshold):
    print(f"Dataset columns with > {na_threshold} proportion of missing images.")
    pprint(d.loc[:, d.isnull().mean() > na_threshold].isnull().mean())


# Missing data -- set threshold, print out columns with more than this percent missing 
NA_THRESHOLD = 0.025

print_columns_with_more_than_threshold_missing(d, NA_THRESHOLD)



# In[14]:


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
    'time_and_date_of_image': (datetime.datetime(2020,3,1,0,0,0,tzinfo=ZoneInfo('US/Eastern')), datetime.datetime(2020,11,16,0,0,0, tzinfo=ZoneInfo('US/Eastern'))),
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
        #assert in_range.all()


check_values_in_range(d, bounds)



# In[15]:


core_anl_vars = ['distance_from_nearest_police_station','distance_from_nearest_crime_1hr','distance_from_nearest_crime_3hr','distance_from_nearest_crime_6hr',
                 'Estimate_Total_Not_Hispanic_or_Latino_White_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone','ntaname','time_and_date_of_image','hour','month','nighttime','day_of_month','day_of_week',
                'density_cbg','median_household_income','boroct2020']


# function to describe specific columns in dataset 
def describe_columns(d, cols):
    return d[cols].describe(datetime_is_numeric=True).apply(lambda s: s.apply('{0:.2f}'.format))

describe_columns(d, core_anl_vars)


# ## 2. Loading in Validation & Test Sets, External Datasets

# ### Validation, Test Sets 

# In[16]:


v = pd.read_csv(VALSET_PATH)
t = pd.read_csv(TESTSET_PATH)


# In[17]:


vgdf = gpd.GeoDataFrame(v, geometry=gpd.points_from_xy(v.lng, v.lat), crs=WGS)
vgdf = vgdf.to_crs(PROJ_CRS)


# In[18]:


tgdf = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t.lng, t.lat), crs=WGS)
tgdf = tgdf.to_crs(PROJ_CRS)


# ### NYC Neighborhood Tabulation Areas (NTAs) 

# In[19]:


nyc_ntas = gpd.read_file("../../nypd-deployment-patterns-old/external_datasets/nynta2020_22c")
nyc_ntas = nyc_ntas.to_crs(PROJ_CRS)


# In[20]:


nyc_ntas.plot()


# ### NYC Census Block Groups (CBGs) 

# In[21]:


ny_cbgs = gpd.read_file('../../nypd-deployment-patterns-old/external_datasets/tl_2020_36_all/tl_2020_36_bg20.shp')
ny_cbgs = ny_cbgs.to_crs(WGS)

nyc_cbgs = ny_cbgs[ny_cbgs.COUNTYFP20.isin(NYC_COUNTY_CODES)]
nyc_cbgs.reset_index(inplace=True)
nyc_cbgs = nyc_cbgs.to_crs(PROJ_CRS)
nyc_cbgs.GEOID20 = pd.to_numeric(nyc_cbgs.GEOID20)
nyc_cbgs.plot()


# ### NYC Zoning Data 

# In[22]:


# Zoning Tests 
nyc_zoning = gpd.read_file("../../nypd-deployment-patterns-old/external_datasets/nycgiszoningfeatures_202212shp")
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

print(nyc_zoning.columns)
nyc_zoning['high_level_zone'] = nyc_zoning['ZONEDIST'].map(lambda z: high_level_zoning(z))


nyc_zoning


# ### NYPD Precinct Locations 

# In[23]:


precincts = pd.read_csv("../../nypd-deployment-patterns-old/external_datasets/nypd_precinct_locs.csv")
precincts_gdf = gpd.GeoDataFrame(precincts, geometry=gpd.points_from_xy(precincts.lng, precincts.lat), crs=WGS)
precincts_gdf = precincts_gdf.to_crs(PROJ_CRS)


# ### NYC Borough Boundaries (NYBB) 

# In[24]:


nybb = gpd.read_file(gpd.datasets.get_path('nybb'))
nybb = nybb.to_crs(PROJ_CRS)


# ### NYC Arrests Data 

# In[25]:


nyc_arrests = pd.read_csv("../../nypd-deployment-patterns-old/external_datasets/NYPD_Arrests_Data__Historic_.csv")


# In[26]:


nyc_arrests = gpd.GeoDataFrame(nyc_arrests, geometry=gpd.points_from_xy(nyc_arrests.Longitude, nyc_arrests.Latitude), crs=WGS)
nyc_arrests = nyc_arrests.to_crs(PROJ_CRS)


# In[27]:


arrests_by_nta = gpd.sjoin(nyc_arrests,nyc_ntas).groupby('NTAName').agg('size').to_frame('num_arrests')


# In[28]:


nyc_ntas = nyc_ntas.merge(arrests_by_nta, left_on='NTAName', right_on='NTAName')


# ### NYC Community Districts 

# In[83]:










# ## 3. NYPD Deployment Analysis 

# ### Computing Probability Measures with Validation Set 

# In[29]:


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


# PUT DGDF HERE 
# GeoDataFrame for d 
d_for_demo_analysis = gpd.GeoDataFrame(d_for_demo_analysis, geometry=gpd.points_from_xy(d_for_demo_analysis.lng, d_for_demo_analysis.lat), crs=WGS)
d_for_demo_analysis = d_for_demo_analysis.to_crs(PROJ_CRS)


# In[30]:
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
        with open(filename, 'w') as f:
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


# density_cbg, median_household_income (residential zones only)
for col in ['median_household_income', 'density_cbg']: 
    percentile_cutoffs = [25, 50, 75]
    print("Fraction of missing values for %s: %2.6f" % (col, d_for_demo_analysis[col].isnull().mean())) 
    dfdd_rzones = d_for_demo_analysis.loc[d_for_demo_analysis['zone'] == 'R']
    d_for_col = dfdd_rzones.dropna(subset=[col]).copy()
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
                             filename='%s_residential_only_bootstraps.json' % col)

    print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics))        

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

    print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics))

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
create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)

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
create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)






# zone table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':'zone',
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='zone_bootstraps.json')

create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)


# boro table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':'boroname',
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='boro_bootstraps.json')

create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)


# neighborhood table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':NEIGHBORHOOD_COL,
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='neighborhood_bootstraps.json')

create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)


# ## Global Constants 

# ### I/O Paths

# In[4]:


ANL_DATASET_PATH = "../../nypd-deployment-patterns-old/output/analysis_dataset.csv"
FIRST_CHUNK_PATH = "../../nypd-deployment-patterns-old/output/1603771200000.csv"
VALSET_PATH = "../valset.csv"
TESTSET_PATH = "../testset.csv"
PAPER_GIT_REPO_PATH = "../plots"


# ### Geographic 

# In[5]:


WGS = 'EPSG:4326'
PROJ_CRS = 'EPSG:2263'
NYC_COUNTY_CODES = ['005', '047', '061', '081', '085']


# ### Analysis Parameters 

# In[6]:

N_BOOTSTRAPS = 10
BOOTSTRAP_OUTPUT_DIR='../output/bootstraps'
ZONE_THRESHOLD = 0.5 # threshold for zone classification
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
                    'black_frac',
                    'white_frac', 
                    'distance_from_nearest_crime_6hr',
                    'distance_from_nearest_police_station',
                    'median_household_income']
PREDICTION_COLS = ['above_threshold', 'calibrated_prediction', 'prediction_adjusted_for_police_station_distance'] # columns with police car predictions. We define these
MIN_POPULATION_IN_AREA = 500
BOROUGH_COL = 'boroname'
NEIGHBORHOOD_COL = 'ntaname'


# ## 1. Dataset Verification 

# In[7]:

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


d = load_analysis_dataset(use_pyarrow=True)



# In[8]:


d.head()


# ### Preprocessing

# In[9]:

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


# In[10]:

# function to remove duplicates from dataset bsaed on specific columns 
def remove_duplicates(d, cols_to_deduplicate_on):
    duplicate_idxs = d.duplicated(subset=cols_to_deduplicate_on)
    print("warning: %i duplicates identified using %s, fraction %2.6f of rows; dropping rows" % (duplicate_idxs.sum(), cols_to_deduplicate_on, duplicate_idxs.mean()))
    d = d.loc[~duplicate_idxs].copy()
    return d

d = remove_duplicates(d, COLS_TO_DEDUPLICATE_ON)

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

# function to filter dataset for dates with full coverage 
def filter_for_dates_with_full_coverage(d, min_date_for_full_coverage):
    len_before_filter = len(d.index)
    print("In demographic analysis, filtering for locations after %s because more geographically representative" % MIN_DATE_FOR_DEMOGRAPHIC_ANALYSIS)
    d = d.loc[d['time_and_date_of_image'] >= min_date_for_full_coverage].copy()
    print("%i/%i rows remaining" % (len(d), len_before_filter))
    return d

d = filter_for_dates_with_full_coverage(d, MIN_DATE_FOR_DEMOGRAPHIC_ANALYSIS)



# In[11]:

# function to fill in NA data in specified columns 
def fill_na_data(d, cols_to_fill_na):
    for col in cols_to_fill_na:
        d[col].fillna(0, inplace=True)
    return d


COLS_TO_FILL_NA = ['conf', 'distance_from_nearest_police_station', 'distance_from_nearest_crime_1hr', 'distance_from_nearest_crime_3hr', 'distance_from_nearest_crime_6hr', 'density_cbg', 'Estimate_Total', 'Estimate_Total_Not_Hispanic_or_Latino_White_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Asian_alone', 'Estimate_Total_Hispanic_or_Latino']
d = fill_na_data(d, COLS_TO_FILL_NA)


cbg_zone_data = pd.read_csv('../../nypd-deployment-patterns-old/external_datasets/cbgs_zone_data.csv')
# function to fill in zone data for each image 
def fill_in_zone_data(d, zone_data):
    
    assert (1.*(cbg_zone_data['C'] > ZONE_THRESHOLD) + 1.*(cbg_zone_data['M'] > ZONE_THRESHOLD) + 1.*(cbg_zone_data['R'] > ZONE_THRESHOLD)).max() == 1
    cbg_zone_dict = {}
    for zone_val in ['C', 'M', 'R']:
        zones = cbg_zone_data.loc[cbg_zone_data[zone_val] >= ZONE_THRESHOLD]
        print("%i CBGs classified as %s" % (len(zones), zone_val))
        cbg_zone_dict.update(dict(zip(zones['GEOID20'].values, [zone_val for _ in range(len(zones))])))
    print(len(cbg_zone_dict))
    d['zone'] = d['GEOID20'].map(lambda x:cbg_zone_dict[x] if x in cbg_zone_dict else None)

    return d

d = fill_in_zone_data(d, cbg_zone_data)

# ### Sanity Checks

# In[12]:

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

# In[13]:

# function to print out columns with more than specified percent missing
def print_columns_with_more_than_threshold_missing(d, na_threshold):
    print(f"Dataset columns with > {na_threshold} proportion of missing images.")
    pprint(d.loc[:, d.isnull().mean() > na_threshold].isnull().mean())


# Missing data -- set threshold, print out columns with more than this percent missing 
NA_THRESHOLD = 0.025

print_columns_with_more_than_threshold_missing(d, NA_THRESHOLD)



# In[14]:


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
    'time_and_date_of_image': (datetime.datetime(2020,3,1,0,0,0,tzinfo=ZoneInfo('US/Eastern')), datetime.datetime(2020,11,16,0,0,0, tzinfo=ZoneInfo('US/Eastern'))),
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
        #assert in_range.all()


check_values_in_range(d, bounds)



# In[15]:


core_anl_vars = ['distance_from_nearest_police_station','distance_from_nearest_crime_1hr','distance_from_nearest_crime_3hr','distance_from_nearest_crime_6hr',
                 'Estimate_Total_Not_Hispanic_or_Latino_White_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone','ntaname','time_and_date_of_image','hour','month','nighttime','day_of_month','day_of_week',
                'density_cbg','median_household_income','boroct2020']


# function to describe specific columns in dataset 
def describe_columns(d, cols):
    return d[cols].describe(datetime_is_numeric=True).apply(lambda s: s.apply('{0:.2f}'.format))

describe_columns(d, core_anl_vars)


# ## 2. Loading in Validation & Test Sets, External Datasets

# ### Validation, Test Sets 

# In[16]:


v = pd.read_csv(VALSET_PATH)
t = pd.read_csv(TESTSET_PATH)


# In[17]:


vgdf = gpd.GeoDataFrame(v, geometry=gpd.points_from_xy(v.lng, v.lat), crs=WGS)
vgdf = vgdf.to_crs(PROJ_CRS)


# In[18]:


tgdf = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t.lng, t.lat), crs=WGS)
tgdf = tgdf.to_crs(PROJ_CRS)


# ### NYC Neighborhood Tabulation Areas (NTAs) 

# In[19]:


nyc_ntas = gpd.read_file("../../nypd-deployment-patterns-old/external_datasets/nynta2020_22c")
nyc_ntas = nyc_ntas.to_crs(PROJ_CRS)


# In[20]:


nyc_ntas.plot()


# ### NYC Census Block Groups (CBGs) 

# In[21]:


ny_cbgs = gpd.read_file('../../nypd-deployment-patterns-old/external_datasets/tl_2020_36_all/tl_2020_36_bg20.shp')
ny_cbgs = ny_cbgs.to_crs(WGS)

nyc_cbgs = ny_cbgs[ny_cbgs.COUNTYFP20.isin(NYC_COUNTY_CODES)]
nyc_cbgs.reset_index(inplace=True)
nyc_cbgs = nyc_cbgs.to_crs(PROJ_CRS)
nyc_cbgs.GEOID20 = pd.to_numeric(nyc_cbgs.GEOID20)
nyc_cbgs.plot()


# ### NYC Zoning Data 

# In[22]:


# Zoning Tests 
nyc_zoning = gpd.read_file("../../nypd-deployment-patterns-old/external_datasets/nycgiszoningfeatures_202212shp")
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

print(nyc_zoning.columns)
nyc_zoning['high_level_zone'] = nyc_zoning['ZONEDIST'].map(lambda z: high_level_zoning(z))


nyc_zoning


# ### NYPD Precinct Locations 

# In[23]:


precincts = pd.read_csv("../../nypd-deployment-patterns-old/external_datasets/nypd_precinct_locs.csv")
precincts_gdf = gpd.GeoDataFrame(precincts, geometry=gpd.points_from_xy(precincts.lng, precincts.lat), crs=WGS)
precincts_gdf = precincts_gdf.to_crs(PROJ_CRS)


# ### NYC Borough Boundaries (NYBB) 

# In[24]:


nybb = gpd.read_file(gpd.datasets.get_path('nybb'))
nybb = nybb.to_crs(PROJ_CRS)


# ### NYC Arrests Data 

# In[25]:


nyc_arrests = pd.read_csv("../../nypd-deployment-patterns-old/external_datasets/NYPD_Arrests_Data__Historic_.csv")


# In[26]:


nyc_arrests = gpd.GeoDataFrame(nyc_arrests, geometry=gpd.points_from_xy(nyc_arrests.Longitude, nyc_arrests.Latitude), crs=WGS)
nyc_arrests = nyc_arrests.to_crs(PROJ_CRS)


# In[27]:


arrests_by_nta = gpd.sjoin(nyc_arrests,nyc_ntas).groupby('NTAName').agg('size').to_frame('num_arrests')


# In[28]:


nyc_ntas = nyc_ntas.merge(arrests_by_nta, left_on='NTAName', right_on='NTAName')


# ### NYC Community Districts 

# In[83]:










# ## 3. NYPD Deployment Analysis 

# ### Computing Probability Measures with Validation Set 

# In[29]:


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


# PUT DGDF HERE 
# GeoDataFrame for d 
d_for_demo_analysis = gpd.GeoDataFrame(d_for_demo_analysis, geometry=gpd.points_from_xy(d_for_demo_analysis.lng, d_for_demo_analysis.lat), crs=WGS)
d_for_demo_analysis = d_for_demo_analysis.to_crs(PROJ_CRS)


# In[30]:
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
        with open(filename, 'w') as f:
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


# density_cbg, median_household_income (residential zones only)
for col in ['median_household_income', 'density_cbg']: 
    percentile_cutoffs = [25, 50, 75]
    print("Fraction of missing values for %s: %2.6f" % (col, d_for_demo_analysis[col].isnull().mean())) 
    dfdd_rzones = d_for_demo_analysis.loc[d_for_demo_analysis['zone'] == 'R']
    d_for_col = dfdd_rzones.dropna(subset=[col]).copy()
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
                             filename='%s_residential_only_bootstraps.json' % col)

    print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics))        

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

    print(create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics))

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
create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)

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
create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)






# zone table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':'zone',
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='zone_bootstraps.json')

create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)


# boro table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':'boroname',
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='boro_bootstraps.json')

create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)


# neighborhood table. Emma reviewed. 
bootstrap_point_estimate, bootstrap_statistics = bootstrap_function_errorbars(df=d_for_demo_analysis, 
                             fxn_to_apply=weighted_disparities_estimator_two_level_grouping, 
                             fxn_kwargs={'census_area_col':LOCATION_COL_TO_GROUP_ON, 
                                         'high_level_group_col':NEIGHBORHOOD_COL,
                                         'total_population_col':TOTAL_POPULATION_COL, 
                                         'estimate_col':'calibrated_prediction'}, 
                             n_bootstraps=N_BOOTSTRAPS, 
                             filename='neighborhood_bootstraps.json')

create_table_from_bootstrap_results(bootstrap_point_estimate, bootstrap_statistics)
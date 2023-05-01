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
from os import get_terminal_size


# ## (Optional) Enable LaTeX font rendering 

# In[3]:


# Requires local LaTeX installation 
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rcParams['text.usetex'] = True
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage[T1]{fontenc}')


# ## Global Constants 

TERMINAL_SIZE = get_terminal_size()
def print_whole_screen_line(): 
    print('-' * TERMINAL_SIZE.columns)

# ### I/O Paths

# In[4]:


ANL_DATASET_PATH = "../output/analysis_dataset.csv"
FIRST_CHUNK_PATH = "../output/1603771200000.csv"
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


d = load_analysis_dataset(load_first_chunk=True)
print_whole_screen_line()


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
print_whole_screen_line()

# In[10]:

# function to remove duplicates from dataset bsaed on specific columns 
def remove_duplicates(d, cols_to_deduplicate_on):
    duplicate_idxs = d.duplicated(subset=cols_to_deduplicate_on)
    print("warning: %i duplicates identified using %s, fraction %2.6f of rows; dropping rows" % (duplicate_idxs.sum(), cols_to_deduplicate_on, duplicate_idxs.mean()))
    d = d.loc[~duplicate_idxs].copy()
    return d

d = remove_duplicates(d, COLS_TO_DEDUPLICATE_ON)
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



# In[11]:

# function to fill in NA data in specified columns 
def fill_na_data(d, cols_to_fill_na):
    for col in cols_to_fill_na:
        d[col].fillna(0, inplace=True)
    return d


COLS_TO_FILL_NA = ['conf', 'distance_from_nearest_police_station', 'distance_from_nearest_crime_1hr', 'distance_from_nearest_crime_3hr', 'distance_from_nearest_crime_6hr', 'density_cbg', 'Estimate_Total', 'Estimate_Total_Not_Hispanic_or_Latino_White_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Asian_alone', 'Estimate_Total_Hispanic_or_Latino', 'Estimate_Total_Not_Hispanic_or_Latino']
d = fill_na_data(d, COLS_TO_FILL_NA)



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
print_whole_screen_line()

# In[13]:

# function to print out columns with more than specified percent missing
def print_columns_with_more_than_threshold_missing(d, na_threshold):
    print(f"Dataset columns with > {na_threshold} proportion of missing images.")
    pprint(d.loc[:, d.isnull().mean() > na_threshold].isnull().mean())


# Missing data -- set threshold, print out columns with more than this percent missing 
NA_THRESHOLD = 0.025

print_columns_with_more_than_threshold_missing(d, NA_THRESHOLD)
print_whole_screen_line()



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



# In[15]:


core_anl_vars = ['distance_from_nearest_police_station','distance_from_nearest_crime_1hr','distance_from_nearest_crime_3hr','distance_from_nearest_crime_6hr',
                 'Estimate_Total_Not_Hispanic_or_Latino_White_alone', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone','ntaname','time_and_date_of_image','hour','month','nighttime','day_of_month','day_of_week',
                'density_cbg','median_household_income','boroct2020']


# function to describe specific columns in dataset 
def describe_columns(d, cols):
    return d[cols].describe(datetime_is_numeric=True).apply(lambda s: s.apply('{0:.2f}'.format))

describe_columns(d, core_anl_vars)
print_whole_screen_line()


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


nyc_ntas = gpd.read_file("/share/pierson/nexar_data/5_other_datasets/nynta2020_22c")
nyc_ntas = nyc_ntas.to_crs(PROJ_CRS)


# In[20]:


nyc_ntas.plot()


# ### NYC Census Block Groups (CBGs) 

# In[21]:


ny_cbgs = gpd.read_file('/share/pierson/nexar_data/5_other_datasets/tl_2020_36_all/tl_2020_36_bg20.shp')
ny_cbgs = ny_cbgs.to_crs(WGS)

nyc_cbgs = ny_cbgs[ny_cbgs.COUNTYFP20.isin(NYC_COUNTY_CODES)]
nyc_cbgs.reset_index(inplace=True)
nyc_cbgs = nyc_cbgs.to_crs(PROJ_CRS)
nyc_cbgs.GEOID20 = pd.to_numeric(nyc_cbgs.GEOID20)
nyc_cbgs.plot()


# ### NYC Zoning Data 

# In[22]:


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

# In[23]:


precincts = pd.read_csv("/share/pierson/nexar_data/5_other_datasets/nypd_precinct_locs.csv")
precincts_gdf = gpd.GeoDataFrame(precincts, geometry=gpd.points_from_xy(precincts.lng, precincts.lat), crs=WGS)
precincts_gdf = precincts_gdf.to_crs(PROJ_CRS)


# ### NYC Borough Boundaries (NYBB) 

# In[24]:


nybb = gpd.read_file(gpd.datasets.get_path('nybb'))
nybb = nybb.to_crs(PROJ_CRS)


# ### NYC Arrests Data 

# In[25]:


nyc_arrests = pd.read_csv("/share/pierson/nexar_data/5_other_datasets/NYPD_Arrests_Data__Historic_.csv")


# In[26]:


nyc_arrests = gpd.GeoDataFrame(nyc_arrests, geometry=gpd.points_from_xy(nyc_arrests.Longitude, nyc_arrests.Latitude), crs=WGS)
nyc_arrests = nyc_arrests.to_crs(PROJ_CRS)


# In[27]:


arrests_by_nta = gpd.sjoin(nyc_arrests,nyc_ntas).groupby('NTAName').agg('size').to_frame('num_arrests')


# In[28]:


nyc_ntas = nyc_ntas.merge(arrests_by_nta, left_on='NTAName', right_on='NTAName')


# ### NYC Community Districts 

# In[83]:


nyc_cds = gpd.read_file("../external_datasets/nyc_cds.shp")
nyc_cds


# ### ACS PUMS Ancestry Dataset - NYC 2020 

# In[77]:


nyc_ancestry = pd.read_csv("../external_datasets/nyc_2020_pums_ancestry.csv", skiprows=4)


# In[81]:


nyc_ancestry


# In[ ]:





# In[156]:


import re
def convert_rows_to_cd(nyc_ancestry): 
    boro_codes = {
        "Manhattan": 1,
        "Bronx": 2,
        "Brooklyn": 3,
        "Queens": 4,
        "Staten Island": 5
    }
    
    mapping = {}
    
    for idx, row in nyc_ancestry.iterrows(): 

        pattern = r'^NYC-(\w+) Community District ((\d+)(\s*&\s*\d+)*)\s*--.*$'

        match = re.match(pattern, row["Selected Geographies"])
        if match:
            borough = match.group(1)
            district_numbers_str = match.group(2)
            district_numbers = re.findall(r'\d+', district_numbers_str)
            
            for n in district_numbers:
                cdnum = f"{boro_codes[borough]}{n.zfill(2)}"
                mapping[cdnum] = row["Selected Geographies"]
            
            
        else:
            pass
            
            
    return mapping


# In[157]:


nyc_cds_parsed = convert_rows_to_cd(nyc_ancestry)


# In[171]:


len(nyc_cds_parsed)


# In[186]:


nyc_cds_df = pd.Series(nyc_cds_parsed, name='Name').to_frame("Name")
nyc_cds_df['cdnum'] = nyc_cds_df.index.astype(float)


# In[187]:


nyc_ancestry_parsed = nyc_cds_df.merge(nyc_ancestry, how='left', left_on='Name', right_on='Selected Geographies')


# In[ ]:


nyc_ancestry_plottable = nyc_cds.merge(nyc_ancestry_parsed, how='left', left_on='boro_cd', right_on='cdnum')


# In[ ]:


nyc_ancestry_plottable.plot()


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
print_whole_screen_line()


# PUT DGDF HERE 
# GeoDataFrame for d 
d_for_demo_analysis = gpd.GeoDataFrame(d_for_demo_analysis, geometry=gpd.points_from_xy(d_for_demo_analysis.lng, d_for_demo_analysis.lat), crs=WGS)
d_for_demo_analysis = d_for_demo_analysis.to_crs(PROJ_CRS)



# ### Geographic Aggregation 

# In[31]:


pop_by_nta = d_for_demo_analysis.groupby(['ntaname','NAME'])[POPULATION_COUNT_COLS + 
                                                              DEMOGRAPHIC_COLS + 
                                                             PREDICTION_COLS].agg('first')[["Estimate_Total_Not_Hispanic_or_Latino_White_alone","Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone"]].groupby(level='ntaname').agg('sum')
pop_by_nta

est_by_nta = d_for_demo_analysis.groupby(['ntaname'])[PREDICTION_COLS].agg('mean')
est_by_nta

nta_grouped_d = pop_by_nta.join(est_by_nta)
nta_grouped_d = (nta_grouped_d, )


# In[32]:


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


# In[33]:


grouped_d.calibrated_prediction


# In[34]:


len(nyc_cbgs.index)


# ### Disparities Estimator 

# In[35]:


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

# In[36]:


# Pearson Correlation Coefficient
pearson_corr = grouped_d.loc[grouped_d[TOTAL_POPULATION_COL] > MIN_POPULATION_IN_AREA, DEMOGRAPHIC_COLS].corr(method='pearson')
mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
heatmap = sns.heatmap(pearson_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Pearson Correlations', fontdict={'fontsize':18}, pad=16);
plt.savefig(f"{PAPER_GIT_REPO_PATH}/pearson_correlations.png")

# In[37]:


# Spearman Correlation Coefficient
spearman_corr = grouped_d.loc[grouped_d[TOTAL_POPULATION_COL] > MIN_POPULATION_IN_AREA, DEMOGRAPHIC_COLS].corr(method='spearman')
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
heatmap = sns.heatmap(spearman_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Spearman Correlations', fontdict={'fontsize':18}, pad=16);
plt.savefig(f"{PAPER_GIT_REPO_PATH}/spearman_correlations.png")

# ### Breakdown by Neighborhood 

# In[38]:


for col in PREDICTION_COLS:
    print("\n\nneighborhoods with highest mean values of %s" % col)
    print(d_for_demo_analysis
          .groupby(NEIGHBORHOOD_COL)[col]
          .agg(['mean', 'size'])
          .reset_index()
          .sort_values(by='mean')[::-1])
nta_breakdown = d_for_demo_analysis.groupby(NEIGHBORHOOD_COL)['calibrated_prediction'].agg(['mean', 'size']).reset_index().sort_values(by='mean')[::-1]
print_whole_screen_line()


# ## Map of Pr(police) by Census area (either neighborhood or Census tract). Can show this next to maps of density and potentially other variables - we decided to make this by neighborhood for a number of reasons. 

# ### Plot Data Loaded from Bootstraps 

# In[39]:


import json
nta_data = json.load(open("/share/pierson/nexar_data/bootstraps_for_matt/neighborhood_bootstraps.json"))


# In[40]:


nta_rta_means = pd.DataFrame.from_dict(nta_data['point_estimate'], orient='index')
nta_rta_means = nta_rta_means[nta_rta_means.index.map(lambda x: 'relative_to_average' in x)]
nta_rta_means.index = nta_rta_means.index.str.replace('_relative_to_average','')
nta_rta_means.columns = ['Pr_police_rta']
nta_rta_means


# In[41]:


nyc_ntas = nyc_ntas.merge(nta_rta_means, left_on='NTAName', right_index=True, how='left')


# In[42]:


bins = [0, 0.25, 0.5, 0.75, 1, 2, 3,  4, 5]
#labels = ['( 0, 0.5 ]', '( 0.5, 1 ]', '( 1, 2 ]', '> 2']
nyc_ntas['Pr_police_rta'].fillna(0, inplace=True)
nyc_ntas['pr_quantile'] = pd.cut(nyc_ntas['Pr_police_rta'], bins)


# In[44]:


fig, ax = plt.subplots(figsize=(24,24))
colormap = 'bwr'
nybb.plot(ax=ax, color='gainsboro', edgecolor='grey')
nyc_ntas.plot(column='pr_quantile', ax=ax, cmap=colormap, legend=False)#, legend_kwds={'loc': 'upper left', 'title': 'Police deployment\n(relative to city average)', 'ncols':2, 'fontsize':50, 'markerscale':3, 'title_fontsize':60, 'alignment':'center'})

n = 16 # how many lines to draw or number of discrete color levels

cmap = plt.get_cmap(colormap)

norm = matplotlib.colors.Normalize(vmin=0, vmax=5)
stretched_bounds = np.interp(np.linspace(0, 1, 257), np.linspace(0, 1, 16), [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
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

# In[45]:


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
#plt.savefig(f"{PAPER_GIT_REPO_PATH}/figures/Pr_police_by_nta.jpg", dpi=450)


# ### VARIANT2: More granular, exposure by CBG 

# In[46]:


dgdf_for_demo_by_cbgs = gpd.sjoin(nyc_cbgs, d_for_demo_analysis, how='left', predicate='contains').groupby('index').agg('mean')


# In[47]:


fig, ax = plt.subplots(figsize=(12,12))

nyc_cbgs_proj_demo = nyc_cbgs.merge(dgdf_for_demo_by_cbgs,left_on='GEOID20',right_on='GEOID20_left')
nyc_cbgs_proj_demo.calibrated_prediction.fillna(0, inplace=True)
nyc_cbgs_proj_demo.calibrated_prediction_decile = pd.qcut(nyc_cbgs_proj_demo.calibrated_prediction, 18, duplicates='drop')
print(nyc_cbgs_proj_demo.calibrated_prediction_decile)
nyc_cbgs_proj_demo.plot(column=nyc_cbgs_proj_demo.calibrated_prediction_decile, ax=ax, cmap='cividis', legend=True)#,legend_kwds={"location": "bottom", "shrink": 0.5, "pad": 0, 'label': 'Calibrated Probability of Police Exposure'})

plt.axis('off')

plt.savefig(f'{PAPER_GIT_REPO_PATH}/PR_police_by_cbg.jpg', dpi=450)


# ### Table of neighborhoods with highest police levels  (include borough as a column as well assuming that each neighborhood is only in one borough). 

# In[48]:


print(nyc_cbgs_proj_demo['Estimate_Total'].sum())

nta_breakdown_top10 = nyc_ntas_proj_demo.sort_values(by='calibrated_prediction')[::-1][:10][["NTAName", "BoroName", "calibrated_prediction"]]

rename = {"NTAName": "Neighborhood", "BoroName": "Borough", "calibrated_prediction": "Calibrated Probability of Police Exposure"}

nta_breakdown_top10.rename(columns=rename, inplace=True)

nta_breakdown_top10.to_latex(f'{PAPER_GIT_REPO_PATH}/tables/nta_breakdown_top10.tex', index=False, float_format="%.2f")

nta_breakdown_top10


# ### Table of police levels by borough (currently this is showing big disparities for Manhattan)

# In[49]:


boro_breakdown = d_for_demo_analysis.groupby('boroname')['calibrated_prediction'].agg(['mean']).sort_values(by='mean')[::-1]

boro_populations = nyc_cbgs_proj_demo.groupby('borocode')['Estimate_Total'].agg('sum')

boro_populations.index = boro_breakdown.index


cols=['calibrated_prediction_mean', 'calibrated_prediction_size', 'calibrated_prediction_sum', 'total_population_mean', 'total_population_size', 'total_population_sum']

rename = {"mean": "Calibrated Probability of Police Exposure", "Index": "Borough"}
boro_breakdown = boro_breakdown.rename(columns=rename)
#boro_breakdown.columns=boro_breakdown.columns.droplevel(0) 
#boro_breakdown.columns = cols

boro_breakdown['Total Population'] = boro_populations
boro_breakdown = boro_breakdown.rename_axis('Borough')


boro_breakdown['Population Weighted Probability of Police'] = (boro_breakdown['Calibrated Probability of Police Exposure'] * boro_breakdown['Total Population']) / boro_breakdown['Total Population'].sum() 
boro_breakdown['Population Weighted Probability of Police, Relative to Mean'] = boro_breakdown['Population Weighted Probability of Police'] / boro_breakdown['Population Weighted Probability of Police'].mean()

boro_breakdown = boro_breakdown['Population Weighted Probability of Police, Relative to Mean']
boro_breakdown.to_latex(f'{PAPER_GIT_REPO_PATH}/tables/borough_breakdown.tex', index=True, float_format="%.2f")    

boro_breakdown


# ### Police levels by zone (residential vs commercial etc). 

# In[50]:


nyc_zoning_demo = gpd.overlay(nyc_cbgs_proj_demo, nyc_zoning, how='intersection')


# In[51]:


nyc_zoning_demo.plot()


# In[52]:


nyc_cbgs_proj_demo.geometry.area.describe()


# In[53]:


nyc_cbgs_proj_demo.groupby('GEOID20').agg('first')["shape_area"].describe()


# In[54]:


nyc_zoning_demo['subarea'] = nyc_zoning_demo.geometry.area
nyc_zoning_demo.subarea.describe()


# In[55]:


fig, ax = plt.subplots(figsize=(8,8))
nyc_cbgs_proj_demo.plot(ax=ax, color='blue', alpha=0.5)
nyc_zoning_demo.plot(ax=ax, color='red', alpha=0.5)
nyc_zoning_demo.groupby('GEOID20').agg('first').plot(ax=ax, color='green', alpha=0.8)


# In[56]:


nyc_zoning_demo.groupby('GEOID20').agg('first').geometry.area.describe()


# In[57]:


cbgs_by_zone_prop = nyc_zoning_demo.groupby(['GEOID20','high_level_zone'])["subarea"].agg('sum').unstack(level=1).fillna(0).div(nyc_cbgs_proj_demo.set_index('GEOID20').geometry.area, axis='rows') 
cbgs_by_zone_prop
residential_cbgs = cbgs_by_zone_prop[cbgs_by_zone_prop.R > 0.9]
commercial_cbgs = cbgs_by_zone_prop[cbgs_by_zone_prop.C > 0.9]
manufacturing_cbgs = cbgs_by_zone_prop[cbgs_by_zone_prop.M > 0.9]


print(commercial_cbgs)
print(manufacturing_cbgs)


# In[58]:


residential_grouped_d = grouped_d[grouped_d.index.isin(residential_cbgs.index)]
commercial_grouped_d = grouped_d[grouped_d.index.isin(commercial_cbgs.index)]
manufacturing_grouped_d = grouped_d[grouped_d.index.isin(manufacturing_cbgs.index)]


# In[59]:


print(d_for_demo_analysis.columns)
#dgdf_for_demo_analysis.drop('index_right', axis=1, inplace=True)
d_mapped_to_zones = gpd.sjoin(nyc_zoning, d_for_demo_analysis)


# In[60]:


pr_by_zone = d_mapped_to_zones.groupby('ZONEDIST').agg('mean','size')[['calibrated_prediction','density_cbg']].sort_values(by='calibrated_prediction')[::-1]
pr_by_zone.index.values
m = [x for x in pr_by_zone.index.values if 'M' in x]
c = [x for x in pr_by_zone.index.values if 'C' in x]
r = [x for x in pr_by_zone.index.values if 'R' in x]
# r also picks up 'PARK' and 'PLAYGROUND' which is a convenient catch in my mind 
print(m)
print(c)
print(r)

m_pr = pr_by_zone[pr_by_zone.index.isin(m)]
c_pr = pr_by_zone[pr_by_zone.index.isin(c)]
r_pr = pr_by_zone[pr_by_zone.index.isin(r)]



def zone_classifier(z): 
    if 'R' in z: 
        return 'R'
    elif 'C' in z:
        return 'C'
    elif 'M' in z:
        return 'M'

print(m_pr.agg('mean'), c_pr.agg('mean'), r_pr.agg('mean'))
print(c_pr.agg('mean') / r_pr.agg('mean'))

pr_by_zone['type'] = pr_by_zone.index.map(lambda x: zone_classifier(x))
pr_by_zone


# In[61]:




def population_weighting(metric_to_weight, weights):
    return (metric_to_weight * weights).sum() / weights.sum()


r_pr = population_weighting(residential_grouped_d.calibrated_prediction, residential_grouped_d['Estimate_Total'])
c_pr = population_weighting(commercial_grouped_d.calibrated_prediction, commercial_grouped_d['Estimate_Total'])
m_pr = population_weighting(manufacturing_grouped_d.calibrated_prediction, manufacturing_grouped_d['Estimate_Total'])


data = {'Residential': r_pr, 'Commercial': c_pr, 'Manufacturing': m_pr}
print(data)
pr_by_zone_table = pd.DataFrame.from_dict(data, orient='index')

pr_by_zone_table = pr_by_zone_table.rename_axis('Zoning Type')
pr_by_zone_table.columns = ['Population-Weighted Probability of Police Exposure']


#pr_by_zone_table.to_latex(f'{PAPER_GIT_REPO_PATH}/tables/pr_by_zone_type.tex', float_format="%.2f")

pr_by_zone_table




# ### Police levels by race 

# In[62]:


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


pr_by_race_table = pd.DataFrame.from_dict(estimates, orient='index')
pr_by_race_table['Weighted Probability of Police, Relative to Mean'] = pr_by_race_table.iloc[:,0] / pr_by_race_table.iloc[:,0].mean()

nice_names = {'Estimate_Total_Not_Hispanic_or_Latino_White_alone': 'White', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone': 'Black/African American', 'Estimate_Total_Not_Hispanic_or_Latino_Asian_alone': 'Asian', 'Estimate_Total_Hispanic_or_Latino': 'Hispanic / Some other race'}
pr_by_race_table.index = pr_by_race_table.index.map(lambda x: nice_names[x])
pr_by_race_table = pr_by_race_table['Weighted Probability of Police, Relative to Mean']



pr_by_race_table.to_latex(f'{PAPER_GIT_REPO_PATH}/tables/pr_by_race.tex', index=True, float_format="%.2f")    
pr_by_race_table
print_whole_screen_line()


# ### Police levels by race (only residential zones)

# In[63]:


for prediction_col in PREDICTION_COLS:
    print("Using prediction col", prediction_col)
    estimates = {}
    for demo_col in POPULATION_COUNT_COLS:
        if demo_col == TOTAL_POPULATION_COL:
            continue
        # compute weighted mean as described in Census tract. 
        grouped_mean = (residential_grouped_d[prediction_col] * residential_grouped_d[demo_col]).sum()/residential_grouped_d[demo_col].sum()
        print(demo_col, grouped_mean)
        estimates[demo_col] = grouped_mean
    #print("Ratio of Black estimate to white estimate: %2.3f" % (estimates[demo_col]/estimates[WHITE_POPULATION_COL]))


pr_by_race_rzones_table = pd.DataFrame.from_dict(estimates, orient='index')
pr_by_race_rzones_table['Weighted Probability of Police, Relative to Mean'] = pr_by_race_table.values
pr_by_race_rzones_table['Weighted Probability of Police, Relative to Mean [R Zoning Only]'] = pr_by_race_rzones_table.iloc[:,0] / pr_by_race_rzones_table.iloc[:,0].mean()

nice_names = {'Estimate_Total_Not_Hispanic_or_Latino_White_alone': 'White', 'Estimate_Total_Not_Hispanic_or_Latino_Black_or_African_American_alone': 'Black/African American', 'Estimate_Total_Not_Hispanic_or_Latino_Asian_alone': 'Asian', 'Estimate_Total_Hispanic_or_Latino': 'Hispanic / Some other race'}
pr_by_race_rzones_table.index = pr_by_race_rzones_table.index.map(lambda x: nice_names[x])
pr_by_race_rzones_table = pr_by_race_rzones_table.iloc[:,1:]



pr_by_race_rzones_table.to_latex(f'{PAPER_GIT_REPO_PATH}/tables/pr_by_race_residential.tex', index=True, float_format="%.2f")    


pr_by_race_rzones_table
print_whole_screen_line()


# ### Arrest Rates Plot

# In[64]:


nyc_arrests = pd.read_csv("/share/pierson/nexar_data/5_other_datasets/NYPD_Arrests_Data__Historic_.csv")


# In[65]:


nyc_arrests = gpd.GeoDataFrame(nyc_arrests, geometry=gpd.points_from_xy(nyc_arrests.Longitude, nyc_arrests.Latitude), crs=WGS)
nyc_arrests = nyc_arrests.to_crs(PROJ_CRS)


# In[66]:


arrests_by_nta = gpd.sjoin(nyc_arrests,nyc_ntas).groupby('NTAName').agg('size').to_frame('num_arrests')


# In[67]:


nyc_ntas = nyc_ntas.merge(arrests_by_nta, left_on='NTAName', right_on='NTAName')


# In[ ]:





# ## 5. Model Development & Evaluation Plots

# ### P/R on V/T Sets 

# In[68]:


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


# In[69]:


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

# In[70]:


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

# In[71]:


tgdf_with_boro = gpd.sjoin(tgdf, nybb)


# In[72]:


t["Manhattan"] = tgdf_with_boro['BoroName'] == "Manhattan"


# In[73]:


t["Manhattan"].describe()


# ### Calibration Plots 

# In[74]:


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

# In[75]:


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

auc_aurpc_results_table


# In[ ]:





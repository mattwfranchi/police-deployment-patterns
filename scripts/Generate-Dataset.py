#!/usr/bin/env python
# coding: utf-8

# # Analysis Dataset Generation 
# Used to produce results in *Detecting disparities in police deployments using dashcam data* (to appear in FAccT '23)

# ## Environment Setup
# 

# ### Prequisite Modules 

# In[1]:


# Built-in 
from glob import glob 
import os
import pandas as pd 
import numpy as np
import ast
import datetime
import zoneinfo
from multiprocessing import Pool

# Install with conda or pip 
import geopandas as gpd

from astral import LocationInfo
from astral.geocoder import database, lookup
from astral.sun import sun

from shapely.geometry import Point
from shapely.ops import nearest_points

from tqdm.notebook import tqdm



# ### Convenience / Helper Functions 

# In[2]:


def print_full(x):
    """
    print_full displays an entire pandas DataFrame x on the console. 

    :param x: a pandas DataFrame (or series?)
    """ 
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', len(x.columns))
    print(x)
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')


# ### Constants & I/O Variables

# In[3]:


# Path for all images in nexar dataset
ALL_IMAGES_PATH = "/share/pierson/nexar_data/raw_data/imgs"
# Root path for other datasets (census, crime, etc) 
DATASETS_ROOT = "/share/pierson/nexar_data/nypd-deployment-patterns/external_datasets"
# Path for all model inferences for images in nexar dataset 
PREDS_PATH = "/share/pierson/nexar_data/nexar_yolov7/"
# Path for output csvs 
OUTPUT_DIR = "/share/pierson/nexar_data/nypd-deployment-patterns/output"


# In[26]:


# EPSG code for WGS, standard coordinate frame for Earth 
WGS = 'EPSG:4326'
# EPSG code for projected NY-area local coordinates
PROJ_CRS = 'EPSG:2263'
# NYC county codes (Bronx, Kings, Queens, New York, Richmond [may be out of order])
NYC_COUNTY_CODES = ['005', '047', '061', '081', '085']
# conversion constant for square miles -> square feet
sqmi2sqft = 27878400
# num cpus for multiprocessing
NUM_CPUS = 12

# In[27]:


# Astral object for New York City
nyc = lookup("New York", database())
print((
    f"Information for {nyc.name}/{nyc.region}\n"
    f"Timezone: {nyc.timezone}\n"
    f"Latitude: {nyc.latitude:.02f}; Longitude: {nyc.longitude:.02f}\n"
))


# ### Classes 

# In[30]:


class CrimeData:
    """
    A class to interface with the NYC CrimeData dataset.

    ...

    Attributes
    ----------
    data : DataFrame
        the contained, parsed data from the dataset CSV


    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
        
    filter_by_datetime(start_datetime, end_datetime): 
        Filter self.data for rows within two datetimes. 
    
    filter_by_borough(borough): 
        Filter self.data for all rows contained within one (1) borough. 
    
    filter_by_crime_number(crime_number): 
        Filter for one (1) specific crime within the dataset. Useful for inspection / debugging. 
    
    filter_by_crime_desc(crime_desc): 
        Filter self.data by crime description (https://data.cityofnewyork.us/api/views/qgea-i56i/files/ee823139-888e-4ad0-badf-e18e2674a9cb?download=true&filename=NYPD_Complaint_Historic_DataDictionary.xlsx). 
    
    filter_by_crime_cat(crime_cat): 
        Filter self.data by crime category (https://data.cityofnewyork.us/api/views/qgea-i56i/files/ee823139-888e-4ad0-badf-e18e2674a9cb?download=true&filename=NYPD_Complaint_Historic_DataDictionary.xlsx)
        
    filter_by_age(age_group): 
        Filter self.data by a specific age group. 
    
    filter_by_race(race): 
        Filter self.data by a specific race. 
    
    return_coords(): 
        Prints and returns (Latitude, Longitude) coordinates of all rows in self.data.
    """
    def __init__(self, filename):
        self.data = pd.read_csv(filename)


        self.data['Lat_Lon'] = [ast.literal_eval(x) for x in self.data['Lat_Lon']]

        self.data["CMPLNT_FR_DTTM"] = pd.to_datetime(self.data["CMPLNT_FR_DT"] + " " + self.data["CMPLNT_FR_TM"]).dt.tz_localize('US/Eastern',nonexistent='shift_forward', ambiguous=True)

        self.data["CMPLNT_TO_DTTM"] = pd.to_datetime(self.data["CMPLNT_TO_DT"] + " " + self.data["CMPLNT_TO_TM"]).dt.tz_localize('US/Eastern', nonexistent='shift_forward', ambiguous=True)

        self.data.drop("CMPLNT_FR_DT", inplace=True, axis=1)
        self.data.drop("CMPLNT_FR_TM", inplace=True, axis=1)
        self.data.drop("CMPLNT_TO_DT", inplace=True, axis=1)
        self.data.drop("CMPLNT_TO_TM", inplace=True, axis=1)


    def filter_by_datetime(self, start_datetime, end_datetime):
        return self.data[(self.data.CMPLNT_FR_DTTM >= start_datetime) & (self.data.CMPLNT_TO_DTTM <= end_datetime)]

    def filter_by_borough(self, borough):
        return self.data[self.data.BORO_NM == borough]

    def filter_by_crime_number(self, crime_number):
        return self.data[self.data.CMPLNT_NUM == crime_number]

    def filter_by_crime_desc(self, crime_desc):
        return self.data[self.data.OFNS_DESC == crime_desc]

    def filter_by_crime_cat(self, crime_cat):
        return self.data[self.data.LAW_CAT_CD == crime_cat]

    def filter_by_age(self, age_group):
        return self.data[self.data.SUSP_AGE_GROUP == age_group]

    def filter_by_race(self, race):
        return self.data[self.data.SUSP_RACE == race]

    def return_coords(self):

        print(self.data["Lat_Lon"].tolist())
        return self.data["Lat_Lon"]


# ## Loading in External Data 

# ### Crime Data (felonies) 
# Filter crime data to only include crimes within range of dataset coverage (March-November 2020). Also filter to only include felonies. 

# In[31]:


reported_felonies = CrimeData(f"{DATASETS_ROOT}/NYPD_Complaint_Data_Historic.csv")\

# Start of dataset coverage: March 1 2020 (this is a little loose around the actual start date, just to ensure we don't miss anything)
START_DT = datetime.datetime(2020, 3, 1, 0, 0, 0, 0, zoneinfo.ZoneInfo('US/Eastern'))
# End of dataset coverage: December 1 2020 (also somewhat loose, same reasons)
END_DT = datetime.datetime(2020, 12, 1, 0, 0, 0, 0, zoneinfo.ZoneInfo('US/Eastern'))
reported_felonies.data = reported_felonies.filter_by_datetime(START_DT, END_DT)
reported_felonies.data.sort_values('CMPLNT_FR_DTTM',inplace=True)

felonies_gdf = gpd.GeoDataFrame(
    reported_felonies.data, 
    geometry= gpd.points_from_xy(
        reported_felonies.data.Longitude, 
        reported_felonies.data.Latitude,
        crs=WGS))
felonies_gdf = felonies_gdf.to_crs(PROJ_CRS)


# ### Load in police precinct location data. 

# In[8]:


precinct_locs = pd.read_csv(f"{DATASETS_ROOT}/nypd_precinct_locs.csv")
precincts_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
    precinct_locs["lng"],
    precinct_locs["lat"], 
    crs=WGS))
precincts_gdf = precincts_gdf.to_crs(PROJ_CRS)


# ### Load in NYC census tract boundary data. 

# In[9]:


nyc_tracts = gpd.read_file(f"{DATASETS_ROOT}/NYC-tracts/")
nyc_tracts_proj = nyc_tracts.to_crs(PROJ_CRS)
nyc_tracts_proj['geoid'] = pd.to_numeric(nyc_tracts_proj['geoid'])



# ### Load in NYC census data. 

# In[10]:


nyc_census = pd.read_excel(f"{DATASETS_ROOT}/nyc_decennialcensusdata_2010_2020_change.xlsx",sheet_name=1, header=3)
nyc_census_filtered = nyc_census[nyc_census["GeoType"] == "CT2020"]

nyc_census_demogs = nyc_census_filtered[["Pop_20","BCT2020","GeoID","Hsp_20P","WNH_20P","BNH_20P","ANH_20P","ONH_20P","NH2pl_20P"]]

nyc_census_demogs['GeoID'] = pd.to_numeric(nyc_census_demogs['GeoID'])


# Add demographic data to each tract through pandas DataFrame merge 

# In[32]:


nyc_tracts_w_demogs = nyc_tracts_proj.merge(nyc_census_demogs, left_on='geoid', right_on='GeoID', how="left")
nyc_tracts_w_demogs = nyc_tracts_w_demogs.to_crs(PROJ_CRS)
nyc_tracts_w_demogs["density_tract"] = (nyc_tracts_w_demogs["Pop_20"] / nyc_tracts_w_demogs["geometry"].area) * sqmi2sqft


# ### Load in State of New York census block group boundary data. 

# In[12]:


ny_cbgs = gpd.read_file(f"{DATASETS_ROOT}/tl_2020_36_all/tl_2020_36_bg20.shp")
ny_cbgs = ny_cbgs.to_crs(WGS)


nyc_cbgs = ny_cbgs[ny_cbgs.COUNTYFP20.isin(NYC_COUNTY_CODES)]
nyc_cbgs.reset_index(inplace=True)
nyc_cbgs = nyc_cbgs.to_crs(PROJ_CRS)

nyc_cbgs['GEOID20'] = pd.to_numeric(nyc_cbgs['GEOID20'])


# ### Load in ACS Data.

# In[13]:


nyc_acs = pd.read_csv(f"{DATASETS_ROOT}/nyc_cbgs_ethnicity_table.csv")


# Clean up ACS data

# In[14]:


nyc_acs.GEOID20 = nyc_acs.GEOID20.apply(lambda x: x[9:])
nyc_acs.GEOID20 = pd.to_numeric(nyc_acs.GEOID20)
nyc_acs.drop(0, inplace=True)
to_drop = [x for x in list(nyc_acs.columns) if 'annotation' in x.lower()]
nyc_acs.drop(to_drop, axis=1, inplace=True)

try:
    nyc_acs.drop(['nan_nan'], axis=1, inplace=True)
except: 
    pass 



nyc_cbgs = nyc_cbgs.merge(nyc_acs, how="left", on='GEOID20')


print(nyc_cbgs.head())
print(len(nyc_cbgs.index))


# ### Load in Median Household Income Data. 

# In[15]:


income = pd.read_csv(f"{DATASETS_ROOT}/ACSDT5Y2020.B19013_2023-01-16T120532/ACSDT5Y2020.B19013-Data.csv")


# Clean up MHI data, merge into NYC CBGs DataFrame 

# In[16]:


income.drop(0,axis=0, inplace=True)
income.drop(["B19013_001EA", "B19013_001M", "B19013_001MA","Unnamed: 6"], axis=1, inplace=True)
income.GEO_ID = income.GEO_ID.apply(lambda x: x[9:])
income.GEO_ID = pd.to_numeric(income.GEO_ID)
income.drop('NAME', axis=1, inplace=True)
nyc_cbgs = nyc_cbgs.merge(income, how="left", left_on="GEOID20", right_on="GEO_ID")

cols_to_rename = { 
    "B19013_001E": "median_household_income"
}
nyc_cbgs.rename(columns=cols_to_rename, inplace=True)
print(nyc_cbgs.columns)

assert ((income["B19013_001E"] == '-').sum()) == ((nyc_cbgs["median_household_income"] == '-').sum())


# ### Calculate CBG density

# In[17]:


nyc_cbgs["density_cbg"] = (nyc_cbgs["Estimate_Total"] / nyc_cbgs["geometry"].area) * sqmi2sqft
nyc_cbgs["density_cbg"].describe()


# ### Calculate CBG centroid

# In[18]:


nyc_cbgs['centroid'] = nyc_cbgs.geometry.centroid


# ### Merging census tract demographic data into CBG DataFrame

# In[19]:


from shapely import wkt
bounds = nyc_cbgs.geometry
print(nyc_cbgs.columns)
nyc_cbgs = nyc_cbgs.set_geometry(nyc_cbgs['centroid']).sjoin(nyc_tracts_w_demogs, how='left', lsuffix="cbg", rsuffix="ct", predicate='within')
nyc_cbgs.set_geometry(bounds,inplace=True)


# ### Final Cleanup

# In[20]:


MoE_columns = [x for x in list(nyc_cbgs.columns) if 'Margin of Error' in x]
est_columns = [x for x in list(nyc_cbgs.columns) if 'Estimate' in x]
other_numeric_cols = ['median_household_income','ALAND20','AWATER20']

for col in MoE_columns + est_columns: 
    nyc_cbgs[col] = nyc_cbgs[col].astype(str).str.replace('-','-1')
    nyc_cbgs[col] = nyc_cbgs[col].astype(float)


# ## Loading in internal data 

# In[21]:


md_filenames = glob("/share/pierson/nexar_data/raw_data/metadata_split_filtered/anl/*.csv")


# In[22]:


mds = [pd.read_csv(x, engine='pyarrow') for x in tqdm(md_filenames)]


# ## Analysis CSV Generation 

# In[23]:


def enriched_csv(md_tuple):
    
    # Unpack the enumerate() tuple 
    m_idx, md = md_tuple
    
    # Skip any empty dataframes (a couple days in original oct-nov metadata are empty, for some reason)
    if len(md.index) == 0: 
        return 
    
    # Output name of file being processed to console 
    print(md_filenames[m_idx])
    
    
    # Metadata -> image directory matching process 
    # Uses simple heuristics
    if "processed" in md_filenames[m_idx]:
        # Thursdays logic
        # Have to try all four folders, as naming is not informative enough to do any matching
        ALL_IMAGES_PATHS = ["/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_0", 
                            "/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_1",
                            "/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_2",
                            "/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_3"]
        PREDS_PATHS = glob("/share/pierson/nexar_data/nexar_yolov7/entire_dataset_inferences/dir_*/exp/labels")
        MD_PATHS = glob("/share/pierson/nexar_data/raw_data/metadata_split_filtered/anl/processed_*.csv")
        
    else:
        # Oct-Nov logic
        ALL_IMAGES_PATHS = [f"/share/pierson/nexar_data/raw_data/imgs/oct_15-nov-15/{os.path.splitext(os.path.basename(md_filenames[m_idx]))[0]}"]
        PREDS_PATHS = [f"/share/pierson/nexar_data/nexar_yolov7/entire_dataset_inferences/{os.path.splitext(os.path.basename(md_filenames[m_idx]))[0]}/exp/labels"]
        MD_PATHS = [f"/share/pierson/nexar_data/raw_data/metadata_split_filtered/anl/{os.path.basename(md_filenames[m_idx])}"]
        
    print(ALL_IMAGES_PATHS)
    
    # Load the data for all inferences
    all_preds = pd.DataFrame()
    for path in PREDS_PATHS:
        preds = {'image_ref': glob(f"{path}/*.txt")}
        all_preds = pd.concat([all_preds, pd.DataFrame(data=preds)], axis=0, ignore_index=True)
        
        
    yhat = []

    for idx, row in all_preds.iterrows():
        # read label y_hat
        full_label_path = row["image_ref"]
        if os.path.exists(full_label_path):
            d = pd.read_csv(full_label_path, 
                            sep=' ', 
                            names=['class_type', 'dummy1', 'dummy2', 'dummy3', 'dummy4', 'conf'])
            # sanity checks. 
            assert d['class_type'].map(lambda x:x in [0, 1]).all()
            assert d['conf'].max() <= 1
            assert d['conf'].min() >= 0
            d = d.loc[d['class_type'] == 1] # only interested in labels for police cars. 
            if len(d) == 0:
                yhat.append(0) # if no labels for police cars, yhat is 0. 
            else:
                yhat.append(d['conf'].max()) # otherwise take max confidence. 
        else:
            yhat.append(0)
        

    yhat = np.array(yhat)
    print(yhat.size, 'have detections')
    all_preds["conf"] = yhat 

    # Create 'base' column in both dataframes, for merge based on image base filename
    all_preds["base"] = all_preds["image_ref"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    md["base"] = md["image_ref"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    # set indexes to 'base' column for faster merge
    md.set_index(md["base"], inplace=True)
    all_preds.set_index(all_preds["base"], inplace=True)
    all_preds["has_prediction"] = 1
    
    # cleanup after setting indexes
    all_preds.drop(["base","image_ref"], axis=1, inplace=True)
    md.drop('base', axis=1, inplace=True)
    
    # process the intersection of the two dataframes
    intersection = pd.merge(md, all_preds, how='left', left_index=True, right_index=True)
    
    print(len(intersection.index)) 
    
    # make sure no rows were dropped
    assert len(md.index) == len(intersection.index)
    
    print(intersection.head())
    
    # at this point, good to update md 
    md = intersection
    
    
    
    # time_and_date
    md["time_and_date_of_image"] = pd.to_datetime(md["timestamp"], unit='ms', utc=True)
    md["time_and_date_of_image"] = md["time_and_date_of_image"].dt.tz_convert('US/Eastern')
    
    # extended temporal metrics
    md["hour"] = md["time_and_date_of_image"].apply(lambda x: x.hour)
    md["day_of_week"] = md["time_and_date_of_image"].apply(lambda x: x.weekday())
    md["day_of_month"] = md["time_and_date_of_image"].apply(lambda x: x.day)
    md["month"] = md["time_and_date_of_image"].apply(lambda x: x.month)
    md["day_of_week"].describe()
    md["day_of_month"].describe()
    md["weekend"] = md["day_of_week"] > 4
    md["weekend"] = md["weekend"].apply(lambda x: 0 if x is False else 1)
    md["phase"] = md["time_and_date_of_image"] > datetime.datetime(2020,9,1,0,0,0, tzinfo=zoneinfo.ZoneInfo("US/Eastern"))
    md["phase"] = md["phase"].apply(lambda x: 0 if x is False else 1)
    
    # daytime calculation
    batch = md.copy(deep=True)
    # Set index of batch to the timestamp column
    batch = batch.set_index('time_and_date_of_image')

    # Get list of all days occuring in subset 
    days = batch.index 
    days = pd.to_datetime(days).date
    days = np.unique(days)

    # Generate sunrise and sunset times for each day in list
    sunrises = np.empty(0,dtype=datetime.datetime)
    sunsets = np.empty(0,dtype=datetime.datetime)
    for day in days: 
        s = sun(nyc.observer, date=day, tzinfo=nyc.timezone)

        sunrises = np.append(sunrises, s["sunrise"])

        sunsets = np.append(sunsets, s["sunset"])



    # Compile sunrise, sunset, and dates into dataframe 
    sun_data = pd.DataFrame()
    sun_data["day"] = pd.to_datetime(days)
    sun_data["sunrise"] = sunrises 
    sun_data["sunset"] = sunsets 

    # Set index to date, convert sunset and sunrise to datetime
    sun_data = sun_data.set_index("day")
    sun_data["sunrise"] = pd.to_datetime(sun_data["sunrise"])
    sun_data["sunset"] = pd.to_datetime(sun_data["sunset"])

    # Generate day column for each batch 
    batch["day"] = pd.to_datetime(batch.index.date)
    #print(batch["day"])
    # Generate sunrise column for each batch 
    # Generate sunset column for each batch 
    batch = batch.merge(sun_data, left_on='day' ,right_index=True, how='left')
    batch.drop("day", axis=1, inplace=True)

    batch["nighttime"] = (batch.index < batch["sunrise"]) | (batch.index > batch["sunset"])

    daytime_imgs = batch[batch["nighttime"] == False]
    nighttime_imgs = batch[batch["nighttime"] == True]

    assert len(daytime_imgs.index) + len(nighttime_imgs.index) == len(batch.index)

    md["nighttime"] = batch["nighttime"].values
    md["nighttime"] = md["nighttime"].apply(lambda x: 0 if x is False else 1)
    
    # Nearest_Crime for 1hr, 3hr, and 6hr threshold
    md = gpd.GeoDataFrame(md, geometry=gpd.points_from_xy(md.lng, md.lat, crs="EPSG:4326"))
    md = md.to_crs("EPSG:2263")   
    
    point_of_nearest_crimes = []
    desc_of_nearest_crimes = []
    time_of_nearest_crimes = []

    metrics = {
        "nearest_crime_1hr": 1,
        "nearest_crime_3hr": 3,
        "nearest_crime_6hr": 6
    }

    for name, td in metrics.items(): 

        point_of_nearest_crimes = []
        desc_of_nearest_crimes = []
        time_of_nearest_crimes = []

        for idx_s, row in md.iterrows(): 
            felonies_soonafter = felonies_gdf[(felonies_gdf["CMPLNT_FR_DTTM"] >= row["time_and_date_of_image"]) & (felonies_gdf["CMPLNT_FR_DTTM"] <= row["time_and_date_of_image"] + datetime.timedelta(hours=td))]
            multipoint = felonies_soonafter.geometry.unary_union
            try:
                queried_geom, nearest_geom = nearest_points(row.geometry, multipoint)
                nearest = felonies_soonafter.geometry == nearest_points(row.geometry, multipoint)[1]
                nearest_crime = felonies_soonafter[nearest]
                nearest_crime = nearest_crime.iloc[0]

                point_of_nearest_crimes.append(nearest_crime.geometry)
                desc_of_nearest_crimes.append(nearest_crime.OFNS_DESC)
                time_of_nearest_crimes.append(nearest_crime.CMPLNT_FR_DTTM)
                
            except Exception as e: 
                #print(e)
                point_of_nearest_crimes.append(None)
                desc_of_nearest_crimes.append(None)
                time_of_nearest_crimes.append(None)


        print(len(point_of_nearest_crimes))
        md[f"point_of_{name}"] = point_of_nearest_crimes 
        md[f"desc_of_{name}"] = desc_of_nearest_crimes
        md[f"time_of_{name}"] = time_of_nearest_crimes


        points_of_nearest_crime = gpd.GeoSeries(md[f"point_of_{name}"], crs=PROJ_CRS)

        distances_from_nearest_crime = points_of_nearest_crime.distance(md.geometry)

        print(distances_from_nearest_crime.describe())

        md[f"distance_from_{name}"] = distances_from_nearest_crime


    
    # Nearest Police Station 
    nearest_station = gpd.sjoin_nearest(md, precincts_gdf, how="left", distance_col="distance")
    print(nearest_station["distance"].describe())
    md["distance_from_nearest_police_station"] = nearest_station["distance"]
    
    
    # Census
    md = gpd.sjoin(md, nyc_cbgs, how='left')
    
    #Filter
    #md.drop('geometry', axis=1, inplace=True)
    
    # Summary
    print(md.columns)
    try:
        md.drop('base.1', inplace=True, axis=1)
    except Exception as e: 
        print(e)

        
    md.to_csv(f"{OUTPUT_DIR}/{os.path.basename(md_filenames[m_idx])}", index=False)


# ### Distributed processing [this actually generates the dataset, assuming all prior steps have been run]

# In[24]:


pool = Pool(processes=NUM_CPUS)                         # Create a multiprocessing Pool
pool.map(enriched_csv, enumerate(mds))  # process data_inputs iterable with pool





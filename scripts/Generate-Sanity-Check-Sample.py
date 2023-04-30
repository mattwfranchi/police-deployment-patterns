#!/usr/bin/env python
# coding: utf-8

# # Generating Sanity Check Sample
# This notebook generates a dataset suitable for sanity checking our overall analysis dataset, which clocks in at 28GB. 

# In[21]:


import os 
from glob import glob 
import pandas as pd 
from PIL import Image
from pathlib import Path


# ### Helper Functions 

# In[28]:


def convert_md_path_to_img_path(path, image_ref, output_dir): 
    base = os.path.basename(image_ref)
    if 'processed' in path: 
        try: 
            img = Image.open("/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_0/"+base)
            img.save(f"{output_dir}/{base}")
            return "/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_0/"
        except FileNotFoundError as e: 
            pass 
        try: 
            img = Image.open("/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_1/"+base)
            img.save(f"{output_dir}/{base}")
            return "/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_1/"
        except FileNotFoundError as e: 
            pass 
        try: 
            img = Image.open("/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_2/"+base)
            img.save(f"{output_dir}/{base}")
            return "/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_2/"
        except FileNotFoundError as e: 
            pass 
        try: 
            img = Image.open("/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_3/"+base)
            img.save(f"{output_dir}/{base}")
            return "/share/pierson/nexar_data/raw_data/imgs/thursdays/dir_3/"
        except FileNotFoundError as e: 
            pass 
    else: 
        
        img = Image.open("/share/pierson/nexar_data/raw_data/imgs/oct_15-nov-15/"+Path(path).stem+'/'+base)
        img.save(f"{output_dir}/{base}")
        return "/share/pierson/nexar_data/raw_data/imgs/oct_15-nov-15/"+Path(path).stem+'/'+base


# ### Constants 

# In[32]:


# Aggregated analysis dataset path on disk (this dataset should be generated with the merge_csvs.sh script)
AGG_ANL_DATASET_PATH = "/share/pierson/nexar_data/nypd-deployment-patterns/output/analysis_dataset.csv"
# Output directory 
OUTPUT_DIR = "/share/pierson/nexar_data/nypd-deployment-patterns/output"
# Confidence threshold above which we denote y-hat as 1 (TRUE). Colloquially, we say this image has a police car if image.conf > CONF_THRESHOLD. 
CONF_THRESHOLD = 0.77

IMG_OUTPUT_DIR = "/share/pierson/nexar_data/nypd-deployment-patterns/output/sanity_check_imgs"


# In[38]:


d = pd.read_csv(AGG_ANL_DATASET_PATH, engine='pyarrow')


# In[5]:


d.describe()


# ### Recipe 

# Randomly sample 100 rows from the overall dataset. 50 rows should have police cars, and the other 50 should not. Also need to retrieve the 100 raw images. 

# In[39]:


sanity_check_d = pd.concat([d.loc[d.conf >= CONF_THRESHOLD].sample(n=50, random_state=8918), d.loc[d.conf < CONF_THRESHOLD].sample(n=50, random_state=8918)])


# In[40]:


sanity_check_d.describe()


# In[41]:


sanity_check_d.to_csv(f"{OUTPUT_DIR}/sanity_check_dataset.csv")


# ### Pulling Images

# In[42]:


md2img = pd.read_csv(f"{OUTPUT_DIR}/md2img.csv", engine='pyarrow')


# In[43]:


sanity_check_d = sanity_check_d.merge(md2img, how='left')


# In[46]:


for idx, row in sanity_check_d.iterrows():
    print(convert_md_path_to_img_path(row['parent_dir'], row['image_ref'], IMG_OUTPUT_DIR))


# In[47]:


assert len(glob(f"{IMG_OUTPUT_DIR}/*.jpg")) == len(sanity_check_d.index)


# In[ ]:





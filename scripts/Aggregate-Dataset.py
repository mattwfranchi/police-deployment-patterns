#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob 
import os 
import pandas as pd 
from tqdm.notebook import tqdm


# In[2]:


DIR_TO_SCAN = "/share/pierson/nexar_data/nypd-deployment-patterns/output"
OUTPUT_DIR = "/share/pierson/nexar_data/nypd-deployment-patterns/output"


# In[3]:


md_filenames = glob(f"{DIR_TO_SCAN}/*.csv")
md_filenames = [x for x in md_filenames if 'analysis_dataset' not in x]
md_filenames = [x for x in md_filenames if 'sanity_check' not in x]
md_filenames = [x for x in md_filenames if 'md2img' not in x]
len(md_filenames)


# In[4]:


mds = [pd.read_csv(x, engine='pyarrow') for x in tqdm(md_filenames)]


# In[5]:


agg_md = pd.concat(mds)


# In[6]:


agg_md.to_csv(f"{OUTPUT_DIR}/analysis_dataset.csv", index=False)


# In[ ]:





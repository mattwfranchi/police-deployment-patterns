#!/usr/bin/env python
# coding: utf-8

# In[6]:


from glob import glob 
import os 
import pandas as pd 
from tqdm.notebook import tqdm


# In[28]:


DIR_TO_SCAN = "/share/pierson/nexar_data/nypd-deployment-patterns/output"
OUTPUT_DIR = "/share/pierson/nexar_data/nypd-deployment-patterns/output"


# In[17]:


md_filenames = glob(f"{DIR_TO_SCAN}/*.csv")
md_filenames = [x for x in md_filenames if 'analysis_dataset' not in x]
md_filenames = [x for x in md_filenames if 'sanity_check' not in x]
md_filenames = [x for x in md_filenames if 'md2img' not in x]
len(md_filenames)


# In[18]:


mds = [pd.read_csv(x, engine='pyarrow', usecols=['image_ref']) for x in tqdm(md_filenames)]


# In[19]:


for idx, md in tqdm(enumerate(mds)): 
    md['parent_dir'] = md_filenames[idx]


# In[22]:


agg_md = pd.concat(mds)


# In[30]:


agg_md.to_csv(f"{OUTPUT_DIR}/md2img.csv", index=False)


# In[ ]:





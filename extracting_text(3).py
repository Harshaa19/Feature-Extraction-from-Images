# -*- coding: utf-8 -*-
"""
@author: SaiSriHarsha
"""

import re
import os
import requests
import pandas as pd
import multiprocessing
import time
import numpy as np
import requests
from PIL import Image
import glob
import matplotlib.pyplot as plt
from functools import wraps
import sys
import cv2

import warnings as w
w.filterwarnings('ignore')

#Selecting all images from my folder
img_fns=glob.glob('G:\My Drive\Colab Notebooks\git\images\*')
len(img_fns)

# Extracting the images ID from each image using regular expression
img_id = []
for i in range(len(img_fns)):
  img_id.append(img_fns[i].split('\\')[-1].split('.')[0])
print(img_id)

len(img_id)


# Plotting the 25 images present in the folder using matplotlib

fig, axs=plt.subplots(5,5,figsize=(10,10))
axs=axs.flatten()
for i in range(25):
  axs[i].imshow(plt.imread(img_fns[i]))
  axs[i].axis('off')
  axs[i].set_title(img_id[i])
plt.show()


# Creating new Data frame for storing all new Values

df = pd.DataFrame({
    'image_id': range(len(img_fns)),
    'text': [np.nan] * len(img_fns),  # Initially filled with NaN values for the text column
    'img_fns': range(len(img_fns)),
})

df['image_id']=img_id
df['img_fns']=img_fns
df.head()


#Using (Pretrained model) Paddle OCR for extraction of text from images


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from paddleocr import PaddleOCR

ocr=PaddleOCR(use_angle_cls=True, lang='en')


# Function to extract tuples from the nested list

def extract_parentheses_data(nested_list):
    result = []
    
    # Recursively iterate over the elements in the list
    def extract_tuples(item):
        if isinstance(item, tuple):
            result.append(item)
        elif isinstance(item, list):
            for sub_item in item:
                extract_tuples(sub_item)

    # Start extracting
    extract_tuples(nested_list)
    
    return result


extracted_data=[]
text_results=[]
for i in range(len(img_fns)):
  pil_image = Image.open(img_fns[i])
  image_np = np.array(pil_image)
  text_results=ocr.ocr(image_np, cls=True)
  extracted_data.append(extract_parentheses_data(text_results))


## converting extracted data into strings

extracted_data=[str(item) for item in extracted_data]

#Moving the extracted data to the data frame with column name 'Text'

df['text'] = extracted_data 
df.head()
df.to_csv(r"C:\Users\koppu\Documents\TextRecognization_Image\text_extracted.csv",index=False)

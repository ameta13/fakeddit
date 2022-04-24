import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
from urllib.error import HTTPError
import numpy as np
import sys

parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')

parser.add_argument('type', type=str, help='train, validate, or test')

args = parser.parse_args()

df = pd.read_csv(args.type, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

pbar = tqdm(total=len(df))

if not os.path.exists("images"):
  os.makedirs("images")
no_images, no_images_urls = 0, []
for index, row in df.iterrows():
  if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    try:
      urllib.request.urlretrieve(image_url, filename="images/" + row["id"] + ".jpg")
    except HTTPError:
      no_images_urls.append(row['image_url'])
      no_images += 1
  pbar.update(1)
print("done")
print('No urls = ', no_images_urls)
print('Count links without images = ', no_images)
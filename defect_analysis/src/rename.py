import sys
import re
import os
import glob
import pandas as pd
import numpy as np

#process the 64 high def images and renames them 1-64 for FIJI processing
def main():
  dir = '/Users/ryanlee/Documents/GitHub/faultdetection/defect_analysis/data/golden_binary_all_copy'
  gridheight = 8
  new_title = 'golden_binary_'
  for filename in os.listdir(dir):
    if filename.endswith('.tif'):
      parts = filename.split('.')
      col_idx = int(parts[1])
      row_idx = int(parts[2])
      flat_idx = (col_idx-1)*gridheight+row_idx
      new_filename = new_title+str(flat_idx).zfill(2)+'.tif'
      old_path = os.path.join(dir, filename)
      new_path = os.path.join(dir, new_filename)
      os.rename(old_path,new_path)

if __name__ == '__main__':
  main()

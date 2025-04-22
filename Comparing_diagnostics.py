#---------------------------#
#Loading necessary libraries
#---------------------------#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import pandas as pd
from astropy.io import fits
from astropy.table import Table, vstack, hstack
import astropy.cosmology as cosmo
import astropy.units as u
from scipy import stats
import seaborn as sns
import matplotlib.gridspec as gridspec
pd.pandas.set_option('display.max_columns', None)
import os, sys

path_1 = os.path.abspath("/Users/jahang/Library/CloudStorage/OneDrive-MacquarieUniversity/Luminosity Functions/Data_prelim")
path_2 = os.path.abspath("/Users/jahang/Library/CloudStorage/OneDrive-MacquarieUniversity/PhD/Codes/Classified")

for module_path in [path_1,path_2]:
	if module_path not in sys.path:
    	    sys.path.append(module_path)
    
from lum_functions import *

plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

def list_files_in_directory(folder_path):
  """Lists all files in the given directory.

  Args:
      folder_path: The path to the directory.

  Returns:
      A list of strings, where each string is a file name in the directory,
      or None if the directory does not exist.
  """
  if not os.path.exists(folder_path):
      print(f"Error: Directory '{folder_path}' not found.")
      return None

  file_names = os.listdir(folder_path)
  return file_names

files = list_files_in_directory(path_2)

#-------------------#
# Reading the files
#-------------------#
# X-ray
g09_main = pd.read_csv(files[0])
g09_hard = pd.read_csv(files[4])

g09_hard.rename({'ML_FLUX_0':'X_flux'},axis=1,inplace=True)
g09_main.rename({'ML_FLUX_1':'X_flux'},axis=1,inplace=True)
g09_Xray = pd.concat([g09_main,g09_hard])

# Optical
g09_opt = pd.read_csv(files[-1])
g23_opt = pd.read_csv(files[-3])

# IR
g09_allwise = pd.read_csv(files[-4])
g23_allwise = pd.read_csv(files[-5])

g09_catwise = pd.read_csv(files[-2])
g23_catwise = pd.read_csv(files[3])

# Radio
g09_radio = pd.read_csv(files[2])
g23_radio = pd.read_csv(files[1])

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
from upsetplot import UpSet
from upsetplot import from_memberships
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


def fix_col(gem_1):
    names = []
    for i in range(len(gem_1)):
        names.insert(i,gem_1.component_id.iloc[i].split(' ')[-1])
    
    gem_1.component_id = names

    return gem_1
    
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
del files[2]

os.chdir(path_2)
#-------------------#
# Reading the files
#-------------------#
# X-ray
g09_main = pd.read_csv(files[0])
g09_hard = pd.read_csv(files[4])
g09_hard.index = [323, 324]
g09_main = fix_col(g09_main)
g09_hard = fix_col(g09_hard)

g09_hard.rename({'ML_FLUX_0':'X_flux'},axis=1,inplace=True)
g09_main.rename({'ML_FLUX_1':'X_flux'},axis=1,inplace=True)
g09_Xray = pd.concat([g09_main,g09_hard])

# Optical
g09_opt = pd.read_csv(files[-1])
g09_opt = fix_col(g09_opt)
g23_opt = pd.read_csv(files[-3])

# IR
g09_allwise = pd.read_csv(files[-4])
g23_allwise = pd.read_csv(files[-5])

#g09_catwise = pd.read_csv(files[-2])
#g23_catwise = pd.read_csv(files[3])

# Radio
g09_radio = pd.read_csv(files[2])
g23_radio = pd.read_csv(files[1])

#------------------------#
# Diagnostic comparison
#------------------------#
#G09
# Optical Index
Optical_index = np.zeros(len(g09_opt))
Optical_index[(g09_opt.BPT_index==1)|(g09_opt.Blue_index==3)|(g09_opt.Blue_index==4)|(g09_opt.WHAN_index==3)|
              (g09_opt.WHAN_index==4)|(g09_opt.MEx_index==3)|(g09_opt.CEx_index==2)] = 1
g09_opt['Optical_index'] = Optical_index

# IR Index
#g09_ir = g09_catwise.merge(g09_allwise[['component_id','W1mag','W2mag','W3mag','W4mag','Kmag','mat_index']],
#                           on=['component_id'],how='inner')
g09_ir = g09_allwise
IR_index = np.zeros(len(g09_ir))
IR_index[(g09_ir.assef_index==1)|(g09_ir.mat_index==1)] = 1
g09_ir['IR_index'] = IR_index
# X-ray combinations
g09_opt_xray = g09_opt.merge(g09_Xray[['component_id','W1_fluxpm','X_flux','X_index']], 
                             on=['component_id'], how='inner')
    #len(g09_opt_xray[(g09_opt_xray.Optical_index==1)&(g09_opt_xray.X_index==1)]) X-ray+opt
    #g09_opt.Optical_index.value_counts().values[g09_opt.Optical_index.value_counts().keys()==1][0] Opt
g09_ir_xray = g09_ir.merge(g09_Xray[['component_id','W1_fluxpm','X_flux','X_index']], 
                           on=['component_id'], how='inner') #0
    #len(g09_ir_xray[(g09_ir_xray.IR_index==1)&(g09_ir_xray.X_index==1)])
g09_radio_xray = g09_radio.merge(g09_Xray[['component_id','W1_fluxpm','X_flux','X_index']], 
                                 on=['component_id'], how='inner') #0
    #len(g09_radio_xray[(g09_radio_xray.radio_index==1)&(g09_radio_xray.X_index==1)])
# Optical combinations
g09_ir_opt = g09_ir.merge(g09_opt[['component_id','logn2ha','logo3hb','logo2hb','HA_EW','absmag_u',
                          'absmag_g','logmstar','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                          'Optical_index']],on=['component_id'],how='inner')
    #len(g09_ir_opt[(g09_ir_opt.Optical_index==1)&(g09_ir_opt.IR_index==1)]) IR+opt
g09_radio_opt = g09_radio.merge(g09_opt[['component_id','logn2ha','logo3hb','logo2hb','HA_EW','absmag_u',
                                'absmag_g','logmstar','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                                'Optical_index']],on=['component_id'],how='inner')
     #len(g09_radio_opt[(g09_radio_opt.Optical_index==1)|(g09_radio_opt.radio_index==1)]) Radio+opt

# IR combinations
g09_radio_ir = g09_radio.merge(g09_ir[['component_id','W1mag','W2mag','W3mag','W4mag','assef_index','mat_index',
                              'IR_index']],on=['component_id'],how = 'inner')
#len(g09_radio_ir[(g09_radio_ir.IR_index==1)&(g09_radio_ir.radio_index==1)])


# Optical-IR-radio
g09_opt_ir_radio = g09_ir_opt.merge(g09_radio[['component_id','W3_flux','radio_flux','radio_index']],
                                    on=['component_id'],how='inner')
    # len(g09_opt_ir_radio[(g09_opt_ir_radio.Optical_index==1)&(g09_opt_ir_radio.IR_index==1)&(g09_opt_ir_radio.radio_index==1)])

# X-ray-Optical-IR_Radio
g09_Xray_opt_ir_radio = g09_opt_ir_radio.merge(g09_Xray[['component_id','W1_fluxpm','X_flux','X_index']],
                                               on=['component_id'],how='inner')
                                               
#len(g09_Xray_opt_ir_radio[(g09_Xray_opt_ir_radio.X_index==1)&(g09_Xray_opt_ir_radio.Optical_index==1)&
#                          (g09_Xray_opt_ir_radio.IR_index==1)&(g09_Xray_opt_ir_radio.radio_index==1)])

# G23
Optical_index = np.zeros(len(g23_opt))
Optical_index[(g23_opt.BPT_index==1)|(g23_opt.Blue_index==3)|(g23_opt.Blue_index==4)|(g23_opt.WHAN_index==3)|
              (g23_opt.WHAN_index==4)|(g23_opt.MEx_index==3)|(g23_opt.CEx_index==2)] = 1
g23_opt['Optical_index'] = Optical_index

# IR Index
#g23_ir = g23_catwise.merge(g23_allwise[['Source_Name','mat_index']],on=['Source_Name'],how='inner')
g23_ir = g23_allwise
IR_index = np.zeros(len(g23_ir))
IR_index[(g23_ir.assef_index==1)|(g23_ir.mat_index==1)] = 1
g23_ir['IR_index'] = IR_index

# X-ray combinations
#g23_opt_xray = g23_opt.merge(g23_Xray[['Source_Name','X_index']], on=['Source_Name'], how='inner')
    #len(g23_opt_xray[(g23_opt_xray.Optical_index==1)&(g23_opt_xray.X_index==1)]) X-ray+opt
    #g23_opt.Optical_index.value_counts().values[g23_opt.Optical_index.value_counts().keys()==1][0] Opt
#g23_ir_xray = g23_ir.merge(g23_Xray[['Source_Name','X_index']], on=['Source_Name'], how='inner') #0
#g23_radio_xray = g23_radio.merge(g23_Xray[['Source_Name','X_index']], on=['Source_Name'], how='inner') #0

# Optical combinations
g23_ir_opt = g23_ir.merge(g23_opt[['Source_Name','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                          'Optical_index']],on=['Source_Name'],how='inner')
                          #len(g23_ir_opt[(g23_ir_opt.Optical_index==1)&(g23_ir_opt.IR_index==1)]) IR+opt
g23_radio_opt = g23_radio.merge(g23_opt[['Source_Name','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                          'Optical_index']],on=['Source_Name'],how='inner')
            #len(g23_radio_opt[(g23_radio_opt.Optical_index==1)|(g23_radio_opt.radio_index==1)]) Radio+opt

# IR combinations
g23_radio_ir = g23_radio.merge(g23_ir[['Source_Name','assef_index','mat_index','IR_index']],on=['Source_Name'],
                                how = 'inner')
#len(g23_radio_ir[(g23_radio_ir.IR_index==1)&(g23_radio_ir.radio_index==1)]) # Radio+IR


# Optical-IR-radio
g23_opt_ir_radio = g23_ir_opt.merge(g23_radio[['Source_Name','radio_index']],on=['Source_Name'],how='inner')
    # len(g23_opt_ir_radio[(g23_opt_ir_radio.Optical_index==1)&(g23_opt_ir_radio.IR_index==1)&(g23_opt_ir_radio.radio_index==1)])

#------------#
# UpSet plot
#------------#
# G09
data = from_memberships(
    [
        ('X-ray',),
        ('Optical',),
        ('IR',),
        ('Radio',),
        ('X-ray', 'Optical'),
        ('X-ray', 'IR'),
        ('X-ray', 'Radio'),
        ('Optical', 'IR'),
        ('Optical','Radio'),
        ('IR','Radio'),
        ('Optical', 'IR', 'Radio'),
        ('X-ray','Optical', 'IR', 'Radio')
    ],
    [324, 4598, 6136, 22807, 62, 203, 124, 373, 552, 2637, 47, 7]
)


UpSet(data).plot()
plt.yscale('log')
plt.title('G09')

plt.savefig('../Plots/G09-upset.png',format='png')
plt.show(block=False)
plt.pause(2)
plt.close()

# G23
data = from_memberships(
    [
        ('Optical',),
        ('IR',),
        ('Radio',),
        ('Optical','IR'),
        ('Optical','Radio'),
        ('IR','Radio'),
        ('Optical', 'IR', 'Radio')
        
    ],
    [3663,4116,11649,285,4325,1929,32]
)

UpSet(data).plot()
plt.yscale('log')
plt.title('G23')

plt.savefig('../Plots/G23-upset.png',format='png')
plt.show(block=False)
plt.pause(2)
plt.close()
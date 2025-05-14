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
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats.mstats import trimmed_var
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.colors as mcolors
from sklearn import metrics
import plotly.express as px
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

os.chdir(path_2)

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
g09_opt = g09_opt[~(g09_opt.logn2ha.isna())&~(g09_opt.logo3hb.isna())&~(g09_opt.logo2hb.isna())]
#g09_opt = g09_opt[~(g09_opt.logn2ha.isna())&~(g09_opt.logo3hb.isna())&~(g09_opt.logo2hb.isna())&
#                  ~(g09_opt.logmstar.isna())&~(g09_opt.absmag_u.isna())&~(g09_opt.absmag_g.isna())]
g09_opt = fix_col(g09_opt)
g23_opt = pd.read_csv(files[-3])
g23_opt = g23_opt[~(g23_opt.logn2ha.isna())&~(g23_opt.logo3hb.isna())&~(g23_opt.logo2hb.isna())]
#g23_opt = g23_opt[~(g23_opt.logn2ha.isna())&~(g23_opt.logo3hb.isna())&~(g23_opt.logo2hb.isna())&
#                  ~(g23_opt.logmstar.isna())&~(g23_opt.absmag_u.isna())&~(g23_opt.absmag_g.isna())]

g09_opt = g09_opt[g09_opt.logo3hb>-4]
g09_opt = g09_opt[g09_opt.logn2ha>-3]
g09_opt = g09_opt[g09_opt.logmstar>4]

# IR
g09_allwise = pd.read_csv(files[-4])
g23_allwise = pd.read_csv(files[-5])

g09_catwise = pd.read_csv(files[-2])
g23_catwise = pd.read_csv(files[3])

# Radio
g09_radio = pd.read_csv(files[2])
g23_radio = pd.read_csv(files[1])
#------------------------#
# Diagnostic comparison
#------------------------#
# G09
#---------------#
# Optical Index
#---------------#
Optical_index = np.zeros(len(g09_opt))
Optical_index[(g09_opt.BPT_index==1)|(g09_opt.Blue_index==3)|(g09_opt.Blue_index==4)|(g09_opt.WHAN_index==3)|
              (g09_opt.WHAN_index==4)|(g09_opt.MEx_index==3)|(g09_opt.CEx_index==2)] = 1
g09_opt['Optical_index'] = Optical_index

#-----------#
# IR Index
#-----------#
#g09_ir = g09_catwise.merge(g09_allwise[['component_id','W1mag','W2mag','W3mag','W4mag','Kmag','mat_index']],
#                           on=['component_id'],how='inner')
g09_ir = g09_allwise
IR_index = np.zeros(len(g09_ir))
IR_index[(g09_ir.assef_index==1)|(g09_ir.mat_index==1)|(g09_ir.ki_index==1)] = 1
g09_ir['IR_index'] = IR_index

#--------------------#
# X-ray combinations
#--------------------#
g09_opt_xray = g09_opt.merge(g09_Xray[['component_id','W1_fluxpm','X_flux','X_index']], 
                             on=['component_id'], how='inner')
g09_ir_xray = g09_ir.merge(g09_Xray[['component_id','W1_fluxpm','X_flux','X_index']], 
                           on=['component_id'], how='inner')
g09_radio_xray = g09_radio.merge(g09_Xray[['component_id','W1_fluxpm','X_flux','X_index']], 
                                 on=['component_id'], how='inner')
#-----------------------#
# Optical combinations
#-----------------------#
g09_ir_opt = g09_ir.merge(g09_opt[['component_id','logn2ha','logo3hb','logo2hb','HA_EW','absmag_u',
                          'absmag_g','logmstar','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                          'Optical_index']],on=['component_id'],how='inner')
g09_radio_opt = g09_radio.merge(g09_opt[['component_id','logn2ha','logo3hb','logo2hb','HA_EW','absmag_u',
                                'absmag_g','logmstar','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                                'Optical_index']],on=['component_id'],how='inner')

#-----------------#
# IR combinations
#-----------------#
g09_radio_ir = g09_radio.merge(g09_ir[['component_id','W1mag','W2mag','W3mag','W4mag','assef_index','mat_index',
                              'ki_index','IR_index']],on=['component_id'],how = 'inner')
                              
#--------------------#
# Optical-IR-radio
#--------------------#
g09_opt_ir_radio = g09_ir_opt.merge(g09_radio[['component_id','W3_flux','radio_flux','radio_index']],
                                    on=['component_id'],how='inner')

#-----------------------#
# X-ray-Optical-IR_Radio
#-----------------------#
g09_Xray_opt_ir_radio = g09_opt_ir_radio.merge(g09_Xray[['component_id','W1_fluxpm','X_flux','X_index']],
                                               on=['component_id'],how='inner')
                                               
#------#
# G23
#---------------#
# Optical Index
#---------------#
Optical_index = np.zeros(len(g23_opt))
Optical_index[(g23_opt.BPT_index==1)|(g23_opt.Blue_index==3)|(g23_opt.Blue_index==4)|(g23_opt.WHAN_index==3)|
              (g23_opt.WHAN_index==4)|(g23_opt.MEx_index==3)|(g23_opt.CEx_index==2)] = 1
g23_opt['Optical_index'] = Optical_index

#-----------#
# IR Index
#-----------#
#g23_ir = g23_catwise.merge(g23_allwise[['Source_Name','mat_index']],on=['Source_Name'],how='inner')
g23_ir = g23_allwise
IR_index = np.zeros(len(g23_ir))
IR_index[(g23_ir.assef_index==1)|(g23_ir.mat_index==1)] = 1
g23_ir['IR_index'] = IR_index

#-----------------------#
# Optical combinations
#-----------------------#
g23_ir_opt = g23_ir.merge(g23_opt[['Source_Name','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                          'Optical_index']],on=['Source_Name'],how='inner')
                          #len(g23_ir_opt[(g23_ir_opt.Optical_index==1)&(g23_ir_opt.IR_index==1)]) IR+opt
g23_radio_opt = g23_radio.merge(g23_opt[['Source_Name','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                          'Optical_index']],on=['Source_Name'],how='inner')

#-----------------------#
# IR combinations
#-----------------------#
g23_radio_ir = g23_radio.merge(g23_ir[['Source_Name','assef_index','mat_index','ki_index','IR_index']],
                                on=['Source_Name'],how = 'inner')

#-----------------------#
# Optical-IR-radio
#-----------------------#
g23_opt_ir_radio = g23_ir_opt.merge(g23_radio[['Source_Name','radio_index']],on=['Source_Name'],how='inner')

#-------------#
# Parameters
#-------------#
par_cols = ['logo3hb','logn2ha','logo2hb','logmstar','absmag_u','absmag_g','W1mag','W2mag','W3mag','X_flux',
            'W1_fluxpm','radio_flux','W3_flux']
            
par_cols = ['logo3hb','logn2ha','logo2hb','logmstar','absmag_u','absmag_g','W1mag','W2mag','W3mag','radio_flux','W3_flux']
#-----------------#
# Normalisation
#-----------------#
input_matrix = g09_opt_ir_radio[par_cols]
input_matrix['u_g'] = input_matrix.absmag_u - input_matrix.absmag_g
input_matrix['W1_W2'] = input_matrix.W1mag - input_matrix.W2mag
input_matrix['W2_W3'] = input_matrix.W2mag - input_matrix.W3mag
input_matrix.radio_flux = np.log10(input_matrix.radio_flux)
#input_matrix.X_flux = np.log10(input_matrix.X_flux)
input_matrix.W3_flux = np.log10(input_matrix.W3_flux)
#input_matrix.W1_fluxpm = np.log10(input_matrix.W1_fluxpm)

# logmstar has an outlier with the component_id == 'SB0001_component_28117a' and has been dropped
#input_matrix.drop(index=26,axis=0,inplace=True) 

# In reality one should identify the features (columns) with high variance with functions like 
# df.apply(trimmed_var) and use those features to perform further analysis. However, from an 
# astrophysical perspective and for the purposes of this work we need all of the features
# available. Also, the outliers are not huge in number to consider any subset at this level
# which can be visualised by uncommenting the line below the next one:

input_matrix = input_matrix.drop(columns=['absmag_u','absmag_g','W1mag','W2mag','W3mag'])
#input_matrix.boxplot(figsize=(12,6))
input_matrix_scaled =  pd.DataFrame(RobustScaler().fit_transform(input_matrix),columns=input_matrix.columns)
#---------------------#
# Initialising DBSCAN
#---------------------#
# par_cols = ['logn2ha','logo3hb']
# input_matrix = input_matrix[par_cols]
# input_matrix_scaled =  pd.DataFrame(RobustScaler().fit_transform(input_matrix),columns=input_matrix.columns)

neighbour = 4
nn = NearestNeighbors(n_neighbors=neighbour)
nbrs = nn.fit(input_matrix_scaled)

distances, indices = nbrs.kneighbors(input_matrix_scaled)
distances = np.sort(distances[:,neighbour-1])  # 4th nearest neighbour
plt.plot(distances)
plt.title('k-distance Graph (for DBSCAN)')
plt.xlabel('Data points sorted by distance')
plt.ylabel(f'{neighbour}-NN distance')
#plt.grid(True)
plt.show()#block=False)
#plt.pause(2)
#plt.close()

named_colors = mcolors.get_named_colors_mapping()
col = list(named_colors.keys())
col.sort()
col.pop(0)
eps = float(input('Enter the eps value: '))
# eps=np.linspace(0,1,100)
# eps = np.delete(eps,0)
# plt.figure(figsize=(20,20))
# i=1
# for eps in eps:
#     if i<100:
#         plt.subplot(10,10,i)
db = DBSCAN(eps=eps, min_samples=neighbour)
labels = db.fit_predict(input_matrix_scaled)

#identifying the points which makes up our core points
sample_cores=np.zeros_like(labels,dtype=bool)

sample_cores[db.core_sample_indices_]=True

#Calculating the number of clusters

n_clusters=len(set(labels))- (1 if -1 in labels else 0)
print('No of clusters:',n_clusters)

#------------#
# Plotting
#------------#

input_matrix['DBSCAN'] = labels
input_matrix.DBSCAN+=1

#for i in range(len(input_matrix)):
    #plt.scatter(input_matrix.iloc[labels==i, 0], input_matrix.iloc[labels==i, 1], color=col[i])
#    plt.scatter(input_matrix.logn2ha.iloc[i],input_matrix.logo3hb.iloc[i], color = col[input_matrix.DBSCAN.iloc[i]])
sns.scatterplot(data=input_matrix,x='logn2ha',y='logo3hb',hue='DBSCAN',palette = col,legend=False)
#    i+=1
plt.show()
#--------#
# PCA
#--------#
# pca = PCA(n_components=2, random_state = 42)
#
# X_t = pca.fit_transform(input_matrix_scaled)
# X_pca = pd.DataFrame(X_t, columns = ['PC1','PC2'])
# X_pca.head()
#
# labels = labels.astype(str)
#
# fig=px.scatter(
#     data_frame=X_pca,
#     x="PC1",
#     y="PC2",
#     color=labels,
#     title="ha"
#
# )
# fig.update_layout(xaxis_title="PC1",yaxis_title="PC2")
# fig.show()
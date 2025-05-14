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
from sklearn.mixture import GaussianMixture
import matplotlib.patches as mpatches
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
named_colors = mcolors.get_named_colors_mapping()
col = list(named_colors.keys())
col.sort()
col = [col[11],col[27],col[32]]
#col.pop(0)

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
                          'NII_corr','OIII_corr','HA_corr','HB_corr',
                          'Optical_index']],on=['component_id'],how='inner')
g09_radio_opt = g09_radio.merge(g09_opt[['component_id','logn2ha','logo3hb','logo2hb','HA_EW','absmag_u',
                                'absmag_g','logmstar','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                                'NII_corr','OIII_corr','HA_corr','HB_corr',
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
                                'NII_corr','OIII_corr','HA_corr','HB_corr',
                          'Optical_index']],on=['Source_Name'],how='inner')
                          #len(g23_ir_opt[(g23_ir_opt.Optical_index==1)&(g23_ir_opt.IR_index==1)]) IR+opt
g23_radio_opt = g23_radio.merge(g23_opt[['Source_Name','BPT_index','Blue_index','WHAN_index','MEx_index','CEx_index',
                            'NII_corr','OIII_corr','HA_corr','HB_corr',
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
            'W1_fluxpm','radio_flux','W3_flux','NII_corr','OIII_corr','HA_corr','HB_corr']
            
par_cols = ['logo3hb','logn2ha','logo2hb','logmstar','absmag_u','absmag_g','W1mag','W2mag','W3mag',
            'radio_flux','W3_flux','NII_corr','OIII_corr','HA_corr','HB_corr']
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
# Initialising GMM
#---------------------#
par_cols = ['NII_corr','OIII_corr','HA_corr','HB_corr',]
input_matrix = input_matrix[par_cols]
input_matrix_scaled =  pd.DataFrame(RobustScaler().fit_transform(input_matrix),columns=input_matrix.columns)

gmm = GaussianMixture(n_components=3, random_state=21, init_params='random')
labels = gmm.fit_predict(input_matrix_scaled)


def kewley(x):
    return 0.61/(x-0.47) + 1.19

def kauffmann(x):
    return 0.61/(x-0.05)+1.3
    
# x = np.linspace(-5,0.4,len(input_matrix))
# x_kauf=np.linspace(-5,0,len(input_matrix))
# input_matrix['GMM'] = labels
# sns.scatterplot(data=input_matrix,x='logn2ha',y='logo3hb',hue='GMM',legend=False)
# sns.lineplot(x=x, y=kewley(x))
# sns.lineplot(x=x_kauf, y=kauffmann(x_kauf))
# plt.xlim(-2,1.2)
# plt.ylim(-1.5,1.5)
# plt.show()

g09_opt_ir_radio['GMM'] = labels
#-----------------------------------#
# Comparison with other diagnostics
#-----------------------------------#
# Optical
#------------#
plt.figure(figsize=(10,10))
#sns.color_palette().as_hex()
plt.subplot(221)
sns.kdeplot(data=g09_opt_ir_radio,x='logn2ha',y='logo3hb', hue='GMM',fill=True, legend=False, palette=col)
sns.kdeplot(x=g09_opt_ir_radio.logn2ha,y=g09_opt_ir_radio.logo3hb,
            hue=g09_opt_ir_radio.BPT_index,palette=['blue','green','purple'],legend=False)
            
handles = [mpatches.Patch(facecolor='blue', label="AGN"),
           mpatches.Patch(facecolor='green', label="SFG"),
           mpatches.Patch(facecolor='purple', label="Composite"),
           mpatches.Patch(facecolor=col[0], label="Cluster 0"),
           mpatches.Patch(facecolor=col[1], label="Cluster 1"),
           mpatches.Patch(facecolor=col[2], label="Cluster 2")]
           
plt.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower left',ncol=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$\rm\log_{10}([NII]/H\alpha)$', fontsize=13)
plt.ylabel(r'$\rm\log_{10}([OIII]/H\beta)$', fontsize=13)
plt.xlim(-2,1.2)
plt.ylim(-1.5,1.5)
plt.tight_layout()

plt.subplot(222)

sns.kdeplot(data=g09_opt_ir_radio,x='logmstar',y='logo3hb', hue='GMM',fill=True, legend=False,palette=col)
sns.kdeplot(x=g09_opt_ir_radio.logmstar,y=g09_opt_ir_radio.logo3hb,
            hue=g09_opt_ir_radio.MEx_index,palette=['green','purple','blue'],legend=False)
# handles = [mpatches.Patch(facecolor='blue', label="AGN"),
#            mpatches.Patch(facecolor='green', label="SFG"),
#            mpatches.Patch(facecolor='purple', label="Composite"),
#            mpatches.Patch(facecolor='#1f77b4', label="Cluster 1"),
#            mpatches.Patch(facecolor='#ff7f0e', label="Cluster 0")]
           
plt.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower left',ncol=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(8,12)
plt.ylim(-1.5,1.5)
plt.xlabel(r'$\rm\log_{10}(M_*/M_\odot)$', fontsize=13)
plt.ylabel(r'$\rm\log_{10}([OIII]/H_\beta)$', fontsize=13)
plt.tight_layout()

plt.subplot(223)

sns.kdeplot(data=g09_opt_ir_radio,x='logo2hb',y='logo3hb', hue='GMM',fill=True, legend=False,palette=col)
sns.kdeplot(x=g09_opt_ir_radio.logo2hb,y=g09_opt_ir_radio.logo3hb,
            hue=g09_opt_ir_radio.Blue_index,palette=['green','purple','purple','blue'],legend=False)
            
# handles = [mpatches.Patch(facecolor='blue', label="AGN"),
#            mpatches.Patch(facecolor='green', label="SFG"),
#            mpatches.Patch(facecolor='purple', label="Comp/LINER"),
#            mpatches.Patch(facecolor='#1f77b4', label="Cluster 1"),
#            mpatches.Patch(facecolor='#ff7f0e', label="Cluster 0")]

handles = [mpatches.Patch(facecolor='blue', label="AGN"),
           mpatches.Patch(facecolor='green', label="SFG"),
           mpatches.Patch(facecolor='purple', label="Comp/LINER"),
           mpatches.Patch(facecolor=col[0], label="Cluster 0"),
           mpatches.Patch(facecolor=col[1], label="Cluster 1"),
           mpatches.Patch(facecolor=col[2], label="Cluster 2")]
           
plt.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower left',ncol=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$\rm\log_{10}([OII]/H\beta)$', fontsize=13)
plt.ylabel(r'$\rm\log_{10}([OIII]/H\beta)$', fontsize=13)
plt.xlim(-2,2)
plt.ylim(-1.5,1.5)
plt.tight_layout()

g09_opt_ir_radio['u_g'] = g09_opt_ir_radio.absmag_u-g09_opt_ir_radio.absmag_g
plt.subplot(224)

sns.kdeplot(data=g09_opt_ir_radio,x='u_g',y='logo3hb', hue='GMM',fill=True, legend=False,palette=col)
sns.kdeplot(x=g09_opt_ir_radio.u_g,y=g09_opt_ir_radio.logo3hb,
            hue=g09_opt_ir_radio.CEx_index,palette=['green','blue'],legend=False)

handles = [mpatches.Patch(facecolor='blue', label="AGN"),
           mpatches.Patch(facecolor='green', label="SFG"),
           mpatches.Patch(facecolor=col[0], label="Cluster 0"),
           mpatches.Patch(facecolor=col[1], label="Cluster 1"),
           mpatches.Patch(facecolor=col[2], label="Cluster 2")]

plt.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower left',ncol=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(-1.5,1.5)
plt.xlim(0,2)
plt.xlabel(r'$u-g$', fontsize=13)
plt.ylabel(r'$\rm\log_{10}([OIII]/H\beta)$', fontsize=13)
plt.tight_layout()
#plt.savefig('/Users/jahang/Desktop/Optical_panel.png',format='png',bbox_inches='tight')
plt.show()

#------------#
# Infrared
#------------#
g09_opt_ir_radio['W1_W2'] = g09_opt_ir_radio.W1mag-g09_opt_ir_radio.W2mag
g09_opt_ir_radio['W2_W3'] = g09_opt_ir_radio.W2mag-g09_opt_ir_radio.W3mag
g09_opt_ir_radio['K_W2'] = g09_opt_ir_radio.Kmag-g09_opt_ir_radio.W2mag

fig = plt.figure(figsize=(12,5))

g = sns.JointGrid(xlim=(11,17),ylim=(-0.25,1.25))
sns.kdeplot(data=g09_opt_ir_radio,x='W2mag',y='W1_W2', hue='GMM',fill=True, legend=False,ax=g.ax_joint,
            palette=col)
sns.kdeplot(x=g09_opt_ir_radio.W2mag,y=g09_opt_ir_radio.W1_W2,
            hue=g09_opt_ir_radio.assef_index,palette=['blue','green'],legend=False,ax=g.ax_joint)
sns.kdeplot(data=g09_opt_ir_radio,x='W2mag', hue='GMM',fill=True, legend=False,ax=g.ax_marg_x,
            palette=col)
sns.kdeplot(data=g09_opt_ir_radio,y='W1_W2', hue='GMM',fill=True, legend=False,ax=g.ax_marg_y,
            palette=col)
            
handles = [mpatches.Patch(facecolor='blue', label="AGN"),
           mpatches.Patch(facecolor='green', label="SFG"),
           mpatches.Patch(facecolor=col[0], label="Cluster 0"),
           mpatches.Patch(facecolor=col[1], label="Cluster 1"),
           mpatches.Patch(facecolor=col[2], label="Cluster 2")]
           
g.fig.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower right',ncol=2,
            bbox_to_anchor=(0.5,0.1))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
g.ax_joint.set_xlim(11,17)
g.ax_joint.set_ylim(-0.3,1.3)
g.ax_joint.set_xlabel(r'$\rm W_{2}\,(mag)$', fontsize=13)
g.ax_joint.set_ylabel(r'$\rm W_{1}-W_{2}\,(mag)$', fontsize=13)

g1 = sns.JointGrid()
sns.kdeplot(data=g09_opt_ir_radio,x='W2_W3',y='W1_W2', hue='GMM',fill=True, legend=False,ax=g1.ax_joint,
            palette=col)
sns.kdeplot(data=g09_opt_ir_radio,x='W2_W3',y='W1_W2',hue='mat_index',
            palette=['green','blue'],legend=False,ax=g1.ax_joint)
sns.kdeplot(data=g09_opt_ir_radio,x='W2_W3', hue='GMM',fill=True, legend=False,ax=g1.ax_marg_x,
            palette=col)
sns.kdeplot(data=g09_opt_ir_radio,y='W1_W2', hue='GMM',fill=True, legend=False,ax=g1.ax_marg_y,
            palette=col)
            
# handles = [mpatches.Patch(facecolor='blue', label="AGN"),
#            mpatches.Patch(facecolor='green', label="SFG"),
#            mpatches.Patch(facecolor='#1f77b4', label="Cluster 1"),
#            mpatches.Patch(facecolor='#ff7f0e', label="Cluster 0")]
           
g1.fig.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower right',ncol=2,
            bbox_to_anchor=(0.5,0.1))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
g1.ax_joint.set_xlim(0,5)
g1.ax_joint.set_ylim(-0.3,1.3)
g1.ax_joint.set_xlabel(r'$\rm W_{2} - W_{3}\,(mag)$', fontsize=13)
g1.ax_joint.set_ylabel('')
#g1.ax_joint.set_ylabel(r'$\rm W_{1}-W_{2}\,(mag)$', fontsize=13)

g2 = sns.JointGrid()
sns.kdeplot(data=g09_opt_ir_radio,y='W2_W3',x='K_W2', hue='GMM',fill=True, legend=False,ax=g2.ax_joint,
            palette=col)
sns.kdeplot(data=g09_opt_ir_radio,y='W2_W3',x='K_W2',hue='ki_index',
            palette=['green','blue'],legend=False,ax=g2.ax_joint)
sns.kdeplot(data=g09_opt_ir_radio,y='W2_W3', hue='GMM',fill=True, legend=False,ax=g2.ax_marg_y,
            palette=col)
sns.kdeplot(data=g09_opt_ir_radio,x='K_W2', hue='GMM',fill=True, legend=False,ax=g2.ax_marg_x,
            palette=col)
            
# handles = [mpatches.Patch(facecolor='blue', label="AGN"),
#            mpatches.Patch(facecolor='green', label="SFG"),
#            mpatches.Patch(facecolor='#1f77b4', label="Cluster 1"),
#            mpatches.Patch(facecolor='#ff7f0e', label="Cluster 0")]
           
#g2.fig.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower right',ncol=2,
#            bbox_to_anchor=(0.5,0.1))
plt.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower left',ncol=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
g2.ax_joint.set_xlim(-0.3,3)
g2.ax_joint.set_ylim(0,5)
g2.ax_joint.set_xlabel(r'$\rm K - W_{2}\,(mag)$', fontsize=13)
g2.ax_joint.set_ylabel(r'$\rm W_{2}-W_{3}\,(mag)$', fontsize=13)

gs = gridspec.GridSpec(1, 3)
mg0 = SeabornFig2Grid(g, fig, gs[0])
mg1 = SeabornFig2Grid(g1, fig, gs[1])
mg2 = SeabornFig2Grid(g2, fig, gs[2])
#gs.tight_layout(fig)
#plt.savefig('/Users/jahang/Desktop/IR_panel.png',format='png',bbox_inches='tight')
plt.show()

#------------#
# Radio
#------------#
#fig = plt.figure(figsize=(12,5))

g = sns.JointGrid()
sns.kdeplot(x=np.log10(g09_opt_ir_radio.radio_flux),y=np.log10(g09_opt_ir_radio.W3_flux), 
            hue=g09_opt_ir_radio.GMM,fill=True, legend=False, ax=g.ax_joint,palette=col)
sns.kdeplot(x=np.log10(g09_opt_ir_radio.radio_flux),y=np.log10(g09_opt_ir_radio.W3_flux),
            hue=g09_opt_ir_radio.radio_index,palette=['green','blue'],legend=False,ax=g.ax_joint)
sns.kdeplot(x=np.log10(g09_opt_ir_radio.radio_flux), hue=g09_opt_ir_radio.GMM,
            fill=True, legend=False,ax=g.ax_marg_x,palette=col)
sns.kdeplot(y=np.log10(g09_opt_ir_radio.W3_flux), hue=g09_opt_ir_radio.GMM,fill=True,
            legend=False,ax=g.ax_marg_y,palette=col)
            
handles = [mpatches.Patch(facecolor='blue', label="AGN"),
           mpatches.Patch(facecolor='green', label="SFG"),
           mpatches.Patch(facecolor=col[0], label="Cluster 0"),
           mpatches.Patch(facecolor=col[1], label="Cluster 1"),
           mpatches.Patch(facecolor=col[2], label="Cluster 2")]
           
g.fig.legend(handles=handles,fontsize=10,markerscale=1.2,frameon=True,loc='lower right',ncol=1,
            bbox_to_anchor=(0.8,0.1))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
g.ax_joint.set_xlim(-4.5,-1.5)
g.ax_joint.set_ylim(-4,-1.5)
#g.ax_joint.set_xscale('log')
#g.ax_joint.set_yscale('log')
g.ax_joint.set_xlabel(r'$\rm S_{943\,MHz}\,(Jy)$', fontsize=13)
g.ax_joint.set_ylabel(r'$\rm S_{W3}\,(Jy)$', fontsize=13)
plt.tight_layout()
#plt.savefig('/Users/jahang/Desktop/radio_panel.png',format='png',bbox_inches='tight')
plt.show(block=False)
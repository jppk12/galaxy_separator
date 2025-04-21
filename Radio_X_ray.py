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

module_path = os.path.abspath("/Users/jahang/Library/CloudStorage/OneDrive-MacquarieUniversity/Luminosity Functions/Data_prelim")
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
#-------------------#
# Loading the data
#-------------------#
g09_hard = pd.read_csv('/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G09/eROSITA/G09_eRASShard.csv')
g09_main = pd.read_csv('/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G09/eROSITA/G09_eRASSmain.csv')
g09_allwise = pd.read_csv('/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G09/WISE/G09_AllWISEmags.csv')
g23_allwise = pd.read_csv('/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G23/WISE/G23_AllWISEmags.csv')

#--------------------#
# The classification
#--------------------#
def x_ray_classify(gem_1, x_flux, ir_flux):
    x_ray_col = np.zeros(len(gem_1))
    x_ray_col[np.log10(gem_1[x_flux]/(gem_1[ir_flux]*1e-23))>0.7] = 1
    
    
    gem_1['X_index'] = x_ray_col
    return gem_1
    
def radio_classify(gem_1, radio_flux, ir_mag, conv = 29.045, scale = 1e-3):
    radio_index = np.zeros(len(gem_1))
    ir_flux = conv*10**(gem_1[ir_mag]/-2.5)
    radio_flux = gem_1[radio_flux]*scale
    radio_index[radio_flux>ir_flux] = 1
    gem_1['W3_flux'] = ir_flux
    gem_1['radio_index'] = radio_index
    
    return gem_1

g09_main = x_ray_classify(g09_main,'ML_FLUX_1','W1_fluxpm')
g09_hard = x_ray_classify(g09_hard,'ML_FLUX_0','W1_fluxpm')

g09_allwise = radio_classify(g09_allwise, 'flux_int', 'W3mag')
g23_allwise = radio_classify(g23_allwise, 'Total_flux', 'W3mag', scale = 1)

#------------------#
# Plotting
#------------------#
def radio_plot(gem_1, w1, w2, scale = 1e-3):
    wise_x=np.linspace(-5,1,len(gem_1))
    sns.scatterplot(x=np.log10(gem_1[gem_1.radio_index==1][w1]*scale), y=np.log10(gem_1[gem_1.radio_index==1][w2]),
    label = 'AGN', marker='^',edgecolor='black')
     
    sns.scatterplot(x=np.log10(gem_1[gem_1.radio_index==0][w1]*scale), y=np.log10(gem_1[gem_1.radio_index==0][w2]),
    label = 'SFG', marker='*',edgecolor='black')
					
    sns.lineplot(x=wise_x,y=wise_x,label='One-to-One',color='black')
    #plt.ylim(-1,2)
    plt.xlabel(r'$\rm S_{943\,MHz}\,(Jy)$', fontsize=13)
    plt.ylabel(r'$\rm S_{W3}\,(Jy)$', fontsize=13)
    plt.legend(fontsize=12)
    return
    
plt.figure(figsize=(10,6))
plt.subplot(121)
radio_plot(g09_allwise,'flux_int','W3_flux')
plt.title('G09')

plt.subplot(122)
radio_plot(g23_allwise,'Total_flux','W3_flux', scale = 1)
plt.title('G23')

plt.show(block=False)
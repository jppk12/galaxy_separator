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
#--------------------------------------#
# Load the data with spectral info
#--------------------------------------#
file = input('Enter the path to the file or press enter to use the default: ')
if file == '':
	g09_path = '/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G09/GAMA/G09_catwise_speclines.csv'
	g23_path = '/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G23/GAMA/G23_catwise_speclines.csv'

g09 = pd.read_csv(g09_path)
g23 = pd.read_csv(g23_path)

#-----------------------------#
# Treating the emission lines
#-----------------------------#
def emission_corr(gem_1,NIIR_FLUX,NIIB_FLUX,OIIR_FLUX,OIIB_FLUX,OIIIR_FLUX,OIIIB_FLUX,
                    HA_FLUX,HA_EW,HB_FLUX,HB_EW):
    
    gem_1['NII_corr'] = N_corr(gem_1.NIIR_FLUX)
    gem_1['OIII_corr'] = O_corr(gem_1.OIIIB_FLUX, gem_1.OIIIR_FLUX)
    gem_1['OII_corr'] = O_corr(gem_1.OIIB_FLUX, gem_1.OIIR_FLUX)
    gem_1['HA_corr'] = H_corr(gem_1.HA_FLUX, gem_1.HA_EW)
    gem_1['HB_corr'] = H_corr(gem_1.HB_FLUX, gem_1.HB_EW)
    gem_1['OII_corr'] = O2_corr(gem_1.OIIB_EW, gem_1.OIIR_EW)
    gem_1['HB_EW_corr'] = N_corr(gem_1.HB_EW)
    
    gem_1['logn2ha'] = np.log10(gem_1.NII_corr/gem_1.HA_corr)
    gem_1['logo3hb'] = np.log10(gem_1.OIII_corr/gem_1.HB_corr)
    gem_1['logo2hb'] = np.log10(gem_1.OII_corr/gem_1.HB_EW_corr)
    
    return gem_1
    
g09 = emission_corr(g09,'NIIR_FLUX','NIIB_FLUX','OIIR_FLUX','OIIB_FLUX','OIIIR_FLUX','OIIIB_FLUX',
                    'HA_FLUX','HA_EW','HB_FLUX','HB_EW')
g23 = emission_corr(g23,'NIIR_FLUX','NIIB_FLUX','OIIR_FLUX','OIIB_FLUX','OIIIR_FLUX','OIIIB_FLUX',
                    'HA_FLUX','HA_EW','HB_FLUX','HB_EW')

#---------------------#
#Selection criteria
#---------------------#

def kewley(x):
    return 0.61/(x-0.47) + 1.19

def kauffmann(x):
    return 0.61/(x-0.05)+1.3

def classify(gem_1):
    #x = np.linspace(-5,0.4,len(gem_1))
    #x_kauf=np.linspace(-5,0,len(gem_1))
        
    k_grp = np.zeros(len(gem_1))
    k_grp[(gem_1.logo3hb>kauffmann(gem_1.logn2ha))&(gem_1.logo3hb<kewley(gem_1.logn2ha))] = 4
    k_grp[gem_1.logn2ha > 0.05] = 4
    k_grp[(gem_1.logo3hb>kewley(gem_1.logn2ha))|(gem_1.logn2ha>0.47)] = 1
    k_grp[k_grp == 0] = 3
    gem_1['BPT_index'] = k_grp
    
    return gem_1
    
g09 = classify(g09)
g23 = classify(g23)

#----------------------#
#BPT diagram
#----------------------#
plt.figure(figsize=(12,8))

def plot_bpt(gem_1):
    x = np.linspace(-5,0.4,len(gem_1))
    x_kauf=np.linspace(-5,0,len(gem_1))
    x_line = np.linspace(-0.2,1,100)
    
    sns.scatterplot(x=gem_1.logn2ha[gem_1.BPT_index==1],      y=gem_1.logo3hb[gem_1.BPT_index==1],marker='^',label='AGN',s=40, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logn2ha[gem_1.BPT_index==4], y=gem_1.logo3hb[gem_1.BPT_index==4],marker='*',label='Composite',s=90, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logn2ha[gem_1.BPT_index==3], y=gem_1.logo3hb[gem_1.BPT_index==3],marker='*', color = 'purple',label='SFG',s=90, edgecolor = 'black')
    
    sns.lineplot(x=x, y = kewley(x), color = 'black',label='Kewley(2001)')
    sns.lineplot(x=x_kauf,y = kauffmann(x_kauf), linestyle = '--',label='Kauffmann(2003)',color='black')
    
    # Cid fernandes et al., (2010)
    sns.lineplot(x=x_line, y=(1.01*x_line)+0.48, color='black', linestyle = '-.',linewidth=2)
    plt.text(-1.8,1.4, 'Sy2', fontsize = 12)
    plt.text(-1.8,-1, 'SFGs', fontsize = 12)
    plt.text(0.5,-1, 'LINERs', fontsize = 12)
    plt.xlim(-2,1.5)
    plt.ylim(-2,2)
    plt.legend(fontsize=12,markerscale=1.2,frameon=True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'$\log_{10}([NII]/H\alpha)$', fontsize=13)
    plt.ylabel(r'$\log_{10}([OIII]/H\beta)$', fontsize=13)
    plt.xlim(-2,1.2)
    plt.ylim(-1.8,1.6)
    plt.tight_layout()
    #plt.savefig('/Users/jahang/OneDrive - Macquarie University/Luminosity Functions/Paper_data/BPT_z.pdf',
    #            format='pdf',bbox_inches='tight')
    
plt.subplot(121)
plot_bpt(g09)
plt.subplot(122)
plot_bpt(g23)
plt.show(block=False)
plt.pause(2)
plt.close()

#-----------------------------#
# The Blue Diagram
#-----------------------------#
# SFG - AGN
#----------#
def blue_agn(x):
    return (0.11/(x-0.92)) + 0.85
    
def blue_liner(x):
    return (0.95*x) - 0.40
    
def comp_a(x):
    return -(x-1)**2 - 0.1*x +0.25
    
def comp_b(x):
    return (x-0.2)**2 - 0.60
    

#----------------------#
# Blue diagram
#----------------------#
plt.figure(figsize=(12,8))

def plot_bpt(gem_1):
    gem_1 = gem_1[(gem_1.HA_FLUX/gem_1.HA_FLUX_ERR)>5]
    print(gem_1.BPT_index.value_counts())
    x = np.linspace(-5,0.4,len(gem_1))
    x_agn = np.linspace(-3,0.9,len(gem_1))
    x_liner = np.linspace(0.72,2,len(gem_1))
    sns.scatterplot(x=gem_1.logo2hb[gem_1.BPT_index==1],      y=gem_1.logo3hb[gem_1.BPT_index==1],marker='^',label='AGN',s=40, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logo2hb[gem_1.BPT_index==4], y=gem_1.logo3hb[gem_1.BPT_index==4],marker='*',label='Composite',s=90, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logo2hb[gem_1.BPT_index==3], y=gem_1.logo3hb[gem_1.BPT_index==3],marker='*', color = 'purple',label='SFG',s=90, edgecolor = 'black')
    
    sns.lineplot(x=x_agn,y = blue_agn(x_agn), color = 'red',label='Lamareille (2010)')
    sns.lineplot(x=x_liner,y = blue_liner(x_liner), color='red')
    plt.axhline(y=0.3, xmax=0.75, color = 'red')
    
    plt.text(-1.8,1.4, 'Sy2', fontsize = 12)
    plt.text(-1.8,-1, 'SFGs', fontsize = 12)
    plt.text(0.5,-1, 'LINERs', fontsize = 12)
    plt.xlim(-2,1.5)
    plt.ylim(-2,2)
    plt.legend(fontsize=12,markerscale=1.2,frameon=True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'$\log_{10}([NII]/H\alpha)$', fontsize=13)
    plt.ylabel(r'$\log_{10}([OIII]/H\beta)$', fontsize=13)
    plt.xlim(-2,2)
    plt.ylim(-1.8,1.6)
    plt.tight_layout()
    #plt.savefig('/Users/jahang/OneDrive - Macquarie University/Luminosity Functions/Paper_data/BPT_z.pdf',
    #            format='pdf',bbox_inches='tight')
    
    
plt.subplot(121)
plot_bpt(g09)
plt.subplot(122)
plot_bpt(g23)
plt.show(block=False)
plt.pause(2)
plt.close()
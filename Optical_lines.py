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
#------------------#
# Some functions
#------------------#

def emission_corr(gem_1,n2r = 'NIIR_FLUX',o2r = 'OIIR_FLUX',o2b = 'OIIB_FLUX',o3r = 'OIIIR_FLUX',
                  o3b = 'OIIIB_FLUX',ha = 'HA_FLUX',ha_ew = 'HA_EW',hb = 'HB_FLUX',hb_ew = 'HB_EW'):
    
    gem_1['NII_corr'] = N_corr(gem_1[n2r])
    gem_1['OIII_corr'] = O_corr(gem_1[o3b], gem_1[o3r])
    gem_1['OII_corr'] = O_corr(gem_1[o2b], gem_1[o2r])
    gem_1['HA_corr'] = H_corr(gem_1[ha],gem_1[ha_ew])
    gem_1['HB_corr'] = H_corr(gem_1[hb],gem_1[hb_ew])
    
    gem_1['logn2ha'] = np.log10(gem_1.NII_corr/gem_1.HA_corr)
    gem_1['logo3hb'] = np.log10(gem_1.OIII_corr/gem_1.HB_corr)
    gem_1['logo2hb'] = np.log10(gem_1.OII_corr/gem_1.HB_corr)
    
    return gem_1

#-------------------------#
# BPT Selection criteria
#-------------------------#

def kewley(x):
    return 0.61/(x-0.47) + 1.19

def kauffmann(x):
    return 0.61/(x-0.05)+1.3

def classify(gem_1,ha_ew='HA_EW',n2_ew = 'NIIR_EW'):
        
    k_grp = np.zeros(len(gem_1))
    k_grp[(gem_1.logo3hb>kauffmann(gem_1.logn2ha))&(gem_1.logo3hb<kewley(gem_1.logn2ha))] = 4
    k_grp[gem_1.logn2ha > 0.05] = 4
    k_grp[(gem_1.logo3hb>kewley(gem_1.logn2ha))|(gem_1.logn2ha>0.47)] = 1
    k_grp[k_grp == 0] = 3
    gem_1['BPT_index'] = k_grp
    
    blue_index = np.zeros(len(gem_1))
    blue_index[(gem_1.logo3hb>blue_agn(gem_1.logo2hb))&(gem_1.logo3hb>blue_liner(gem_1.logo2hb))] = 4 # AGN
    blue_index[(gem_1.logo3hb>blue_agn(gem_1.logo2hb))&(gem_1.logo3hb<=blue_liner(gem_1.logo2hb))] = 3 # LINER
    blue_index[(gem_1.logo3hb<=blue_agn(gem_1.logo2hb))&(gem_1.logo3hb>0.3)] = 2 # SFG COMP
    blue_index[(gem_1.logo3hb<=blue_agn(gem_1.logo2hb))&(gem_1.logo3hb<=0.3)] = 1 #SFG
    blue_index[(gem_1.logo2hb>=0.92)&(gem_1.logo3hb>blue_liner(gem_1.logo2hb))] = 4
    blue_index[(gem_1.logo2hb>=0.92)&(gem_1.logo3hb<blue_liner(gem_1.logo2hb))] = 3
    
    
    gem_1['Blue_index'] = blue_index
    
    whan_index = np.zeros(len(gem_1))
    whan_index[(gem_1.logn2ha<-0.4)&(gem_1[ha_ew]>3)] = 1 #psf
    whan_index[(gem_1.logn2ha>-0.4)&(gem_1[ha_ew]>3)&(gem_1[ha_ew]<=6)] = 2 #wagn
    whan_index[(gem_1.logn2ha>-0.4)&(gem_1[ha_ew]>6)] = 3#sagn
    whan_index[gem_1[ha_ew]<=3] = 0 #fagn
    whan_index[(gem_1[ha_ew]<0.5)&(gem_1[n2_ew]<0.5)] = 0 #passive
    
    gem_1['WHAN_index'] = whan_index
    return gem_1
        
        
#-----------------------------#
# The Blue Diagram
#-----------------------------#

def blue_agn(x):
    return (0.11/(x-0.92)) + 0.85
    
def blue_liner(x):
    return (0.95*x) - 0.40
    
def comp_a(x):
    return -(x-1)**2 - 0.1*x +0.25
    
def comp_b(x):
    return (x-0.2)**2 - 0.60
#--------------------#
# How to plot
#--------------------#
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
    #plt.xlim(-2,1.5)
    #plt.ylim(-2,2)
    plt.legend(fontsize=12,markerscale=1.2,frameon=True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'$\rm\log_{10}([NII]/H\alpha)$', fontsize=13)
    plt.ylabel(r'$\rm\log_{10}([OIII]/H\beta)$', fontsize=13)
    plt.xlim(-2,1.2)
    plt.ylim(-1.8,1.6)
    plt.tight_layout()
    #plt.savefig('/Users/jahang/OneDrive - Macquarie University/Luminosity Functions/Paper_data/BPT_z.pdf',
    #            format='pdf',bbox_inches='tight')
    
    return
def plot_blue(gem_1):
    x = np.linspace(-5,0.4,len(gem_1))
    x_agn = np.linspace(-3,0.9,len(gem_1))
    x_liner = np.linspace(0.72,2,len(gem_1))
    x_comp = np.linspace(0.1,1.1,100)
    
    sns.scatterplot(x=gem_1.logo2hb[gem_1.Blue_index==4], y=gem_1.logo3hb[gem_1.Blue_index==4],marker='^',label='AGN',s=40, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logo2hb[gem_1.Blue_index==2], y=gem_1.logo3hb[gem_1.Blue_index==2],marker='*',label='Comp',s=90, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logo2hb[gem_1.Blue_index==3], y=gem_1.logo3hb[gem_1.Blue_index==3],marker='^', color = 'purple',label='LINER',s=90, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logo2hb[gem_1.Blue_index==1], y=gem_1.logo3hb[gem_1.Blue_index==1],marker='*', color = 'purple',label='SFG',s=90, edgecolor = 'black')
    
    sns.lineplot(x=x_agn,y = blue_agn(x_agn), color = 'red',label='Lamareille (2010)')
    sns.lineplot(x=x_liner,y = blue_liner(x_liner), color='red')
    sns.lineplot(x=x_comp,y=comp_a(x_comp),color='red')
    sns.lineplot(x=x_comp,y=comp_b(x_comp),color='red')
    plt.axhline(y=0.3, xmax=0.75, color = 'red')
    
    plt.text(-1.8,1.4, 'Sy2', fontsize = 12)
    plt.text(-1.8,-1, 'SFGs', fontsize = 12)
    plt.text(1.5,-1, 'LINERs', fontsize = 12)
    #plt.xlim(-2,1.5)
    #plt.ylim(-2,2)
    plt.legend(fontsize=12,markerscale=1.2,frameon=True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'$\rm\log_{10}([NII]/H\alpha)$', fontsize=13)
    plt.ylabel(r'$\rm\log_{10}([OIII]/H\beta)$', fontsize=13)
    plt.xlim(-2,2)
    plt.ylim(-1.8,1.6)
    plt.tight_layout()
    #plt.savefig('/Users/jahang/OneDrive - Macquarie University/Luminosity Functions/Paper_data/BPT_z.pdf',
    #            format='pdf',bbox_inches='tight')
    return
    
def whan_plot(gem_1):
    
    sns.scatterplot(x=gem_1.logn2ha[gem_1.WHAN_index==3], y=gem_1.HA_EW[gem_1.WHAN_index==3],marker='^',label='sAGN',s=40, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logn2ha[gem_1.WHAN_index==2], y=gem_1.HA_EW[gem_1.WHAN_index==2],marker='^',label='wAGN',s=40, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logn2ha[gem_1.WHAN_index==1], y=gem_1.HA_EW[gem_1.WHAN_index==1],marker='*', color = 'purple',label='SFG',s=40, edgecolor = 'black')
    
    sns.scatterplot(x=gem_1.logn2ha[gem_1.WHAN_index==0], y=gem_1.HA_EW[gem_1.WHAN_index==0],marker='*', color = 'magenta',label='Retired & Passive',s=90, edgecolor = 'black')

    
    plt.plot([-0.4,-0.4],[3,1e3],linestyle='--', color='black')
    plt.plot([-0.4,2],[6,6],linestyle='--', color='black')
    plt.axhline(3,linestyle='--', color='black')
    
    plt.xlim(-6,2)
    plt.ylim(1e-2,1e3)
    plt.yscale('log')
    
    plt.xlabel(r'$\rm\log_{10}([NII]/H\alpha)$', fontsize=13)
    plt.ylabel(r'$\rm\,EW_{H_\alpha}$', fontsize=13)
    return
#--------------------------------------#
# Load the data with spectral info
#--------------------------------------#

#file = '/Users/jahang/Downloads/gal_line_dr7_v5_2.csv'
#data = pd.read_csv(file)
#data['HA_EW'] = data.H_ALPHA_FLUX/data.H_ALPHA_CONT
#data['HB_EW'] = data.H_BETA_FLUX/data.H_BETA_CONT

g09_path = '/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G09/GAMA/G09_catwise_speclines.csv'
g23_path = '/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G23/GAMA/G23_catwise_speclines.csv'
g09 = pd.read_csv(g09_path)
g23 = pd.read_csv(g23_path)

#-----------------------------#
# Treating the emission lines
#-----------------------------#    
g09 = emission_corr(g09,n2r='NIIR_FLUX',o2r='OIIR_FLUX',o2b='OIIB_FLUX',o3r='OIIIR_FLUX',
                    o3b='OIIIB_FLUX',ha='HA_FLUX',ha_ew='HA_EW',hb='HB_FLUX',hb_ew='HB_EW')

g23 = emission_corr(g23,n2r='NIIR_FLUX',o2r='OIIR_FLUX',o2b='OIIB_FLUX',o3r='OIIIR_FLUX',
                    o3b='OIIIB_FLUX',ha='HA_FLUX',ha_ew='HA_EW',hb='HB_FLUX',hb_ew='HB_EW')

g09 = classify(g09)
g23 = classify(g23)

# data = emission_corr(data,n2r='NII_6548_FLUX',o3r='OIII_5007_FLUX',o2b='OII_3726_FLUX',o2r='OII_3729_FLUX',
#                      ha='H_ALPHA_FLUX', ha_ew='HA_EW',hb='H_BETA_FLUX',hb_ew='HB_EW',o3b='OIII_5007_FLUX')
#
# data.OIII_corr = data.OIII_corr/2
# data.logo3hb = np.log10(data.OIII_corr/data.HB_corr)
#
#
# n=-999999
# g09 = g09[((g09.HA_FLUX/g09.HA_FLUX_ERR)>=n)&((g09.HB_FLUX/g09.HB_FLUX_ERR)>=n)&
#          ((g09.NIIR_FLUX/g09.NIIR_FLUX_ERR)>=n)&((g09.NIIB_FLUX/g09.NIIB_FLUX_ERR)>=n)&
#          ((g09.OIIR_FLUX/g09.OIIR_FLUX_ERR)>=n)&((g09.OIIB_FLUX/g09.OIIB_FLUX_ERR)>=n)&
#          ((g09.OIIIR_FLUX/g09.OIIIR_FLUX_ERR)>=n)&((g09.OIIIB_FLUX/g09.OIIIB_FLUX_ERR)>=n)]
#
# g23 = g23[((g23.HA_FLUX/g23.HA_FLUX_ERR)>=n)&((g23.HB_FLUX/g23.HB_FLUX_ERR)>=n)&
#          ((g23.NIIR_FLUX/g23.NIIR_FLUX_ERR)>=n)&((g23.NIIB_FLUX/g23.NIIB_FLUX_ERR)>=n)&
#          ((g23.OIIR_FLUX/g23.OIIR_FLUX_ERR)>=n)&((g23.OIIB_FLUX/g23.OIIB_FLUX_ERR)>=n)&
#          ((g23.OIIIR_FLUX/g23.OIIIR_FLUX_ERR)>=n)&((g23.OIIIB_FLUX/g23.OIIIB_FLUX_ERR)>=n)]
#----------------------#
# GAMA
#----------------------#
plt.figure(figsize=(10,6))
plt.subplot(121)
plot_bpt(g09)
plt.title('G09')
plt.subplot(122)
plot_bpt(g23)
plt.title('G23')
plt.show(block=False)
plt.pause(2)
plt.close()
#---------------#
# Classified data
#---------------#
cols = ['island_id', 'component_id', 'component_name','uberID','CATAID', 'ra_catwise',
       'dec_catwise', 'ra_hms_cont', 'dec_dms_cont', 'ra_deg_cont',
       'dec_deg_cont','BPT_index']
collated_g09 = g09[cols]

cols = ['Source_Name', 'uberID','CATAID','RA', 'DEC','BPT_index']
collated_g23 = g23[cols]

#-------------#
# SDSS
#-------------#
# data=data[((data.NII_6548_FLUX/data.NII_6548_FLUX_ERR)>=5)&
#     ((data.OIII_5007_FLUX/data.OIII_5007_FLUX_ERR)>=5)&
#     ((data.OII_3726_FLUX/data.OII_3726_FLUX_ERR)>=5)&
#     ((data.OII_3729_FLUX/data.OII_3729_FLUX_ERR)>=5)&
#     ((data.H_ALPHA_FLUX/data.H_ALPHA_FLUX_ERR)>=5)&
#     ((data.H_BETA_FLUX/data.H_BETA_FLUX_ERR)>=5)]
#
# data = classify(data)
# data1 = data[data.BPT_index==3]
# plt.figure(figsize=(12,7))
# plt.subplot(121)
# plot_bpt(data)
# plt.xlim(-2.5,1.5)
#
# plt.subplot(122)
# plot_blue(data)
# sns.kdeplot(x=data.logo2hb, y=data.logo3hb, hue=data.BPT_index, palette=["C0", "C1", "C2"])
#
# plt.show()
# plt.pause(2)
# plt.close()

# Blue classification on G09 and G23 sources without Halpha line
#----------------------#
# Blue diagram
#----------------------#
plt.figure(figsize=(10,6))    

plt.subplot(121)
plot_blue(g09)
plt.title('G09')
plt.subplot(122)
plot_blue(g23)
plt.title('G23')
plt.show(block=False)
plt.pause(2)
plt.close()

collated_g09 = pd.merge(collated_g09,g09[['component_id','Blue_index']],on='component_id')
collated_g23 = pd.merge(collated_g23,g23[['Source_Name','Blue_index']],on='Source_Name')


#-----------------#
# WHAN Diagram
#-----------------#
plt.figure(figsize=(10,6))    

plt.subplot(121)
whan_plot(g09)
plt.title('G09')
plt.subplot(122)
whan_plot(g23)
plt.title('G23')
plt.show(block=False)
plt.pause(2)
plt.close()

collated_g09 = pd.merge(collated_g09,g09[['component_id','WHAN_index']],on='component_id')
collated_g23 = pd.merge(collated_g23,g23[['Source_Name','WHAN_index']],on='Source_Name')

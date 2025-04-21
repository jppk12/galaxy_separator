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

g09_path = '/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G09/WISE/G09_CatWISEmags.csv'
g23_path = '/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G23/WISE/G23_CatWISEmags.csv'

g09 = pd.read_csv(g09_path)
g23 = pd.read_csv(g23_path)
g09_allwise = pd.read_csv('/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G09/WISE/G09_AllWISEmags.csv')
g23_allwise = pd.read_csv('/Users/jahang/Documents/PhD/AGN Catalogue/Xmatched/G23/WISE/G23_AllWISEmags.csv')


def assef_sel(w2):
    l=np.zeros(len(w2))
    for i in range(len(l)):
        if w2[i]>13.86:
            l[i]=0.65*np.exp(0.153*(w2[i]-13.86)**2)
        else:
            l[i]=0.65
    return l
	
def mateos_wedge(w2_w3):
    
    #w2_w3 = gem_1[w2]-gem_1[w3]
    y_bottom = 0.315*(w2_w3) - 0.222
    y_top = 0.315*(w2_w3) + 0.796
    y_left = -3.172*(w2_w3) + 7.624
     
    return y_bottom, y_top, y_left
    
def classify(gem_1,w1,w2):
	assef_col=np.ones(len(gem_1))
	assef_col[(gem_1[w1] - gem_1[w2]) <= assef_sel(gem_1[w2])] = 3
	gem_1['assef_index']=assef_col
    
	return gem_1
	
def mat_classify(gem_1, K, w1, w2, w3):
    mat_col = np.zeros(len(gem_1))
    y_b, y_t, y_l = mateos_wedge(gem_1[w2]-gem_1[w3])
    mat_col[((gem_1[w1]-gem_1[w2])>=y_b)&((gem_1[w1]-gem_1[w2])<=y_t)&((gem_1[w1]-gem_1[w2])>=y_l)] = 1
    gem_1['mat_index'] = mat_col
    
    ki_col = np.zeros(len(gem_1))    
    ki_col[((gem_1[K]-gem_1[w2]) > 1.489)&((gem_1[w2]-gem_1[w3]) > 1.835)] = 1
    gem_1['ki_index'] = ki_col
    
    return gem_1
    
g09 = classify(g09,w1='W1mproPM',w2='W2mproPM')
g23 = classify(g23,w1='W1mproPM',w2='W2mproPM')
g09_allwise = mat_classify(g09_allwise,'Kmag','W1mag','W2mag','W3mag')
g23_allwise = mat_classify(g23_allwise,'Kmag','W1mag','W2mag','W3mag')

#-----------#
# KI & KIM
#-----------#
g09_ki = g09_allwise[~g09_allwise.Kmag.isna()]
g23_ki = g23_allwise[~g23_allwise.Kmag.isna()]


g09 = g09[(g09.snrW1pm>=5)&(g09.snrW2pm>=5)]
g23 = g23[(g23.snrW1pm>=5)&(g23.snrW2pm>=5)]
#-------------------#
# How to plot
#-------------------#
def wise_plot(gem_1, w1, w2):
    wise_x=np.linspace(5,19,len(gem_1))
    sns.scatterplot(y=gem_1[gem_1.assef_index==1][w1]-gem_1[gem_1.assef_index==1][w2],
    x=gem_1[gem_1.assef_index==1][w2],label = 'AGN', marker='^',edgecolor='black')
     
    sns.scatterplot(y=gem_1[gem_1.assef_index==3][w1]-gem_1[gem_1.assef_index==3][w2], 
    x=gem_1[gem_1.assef_index==3][w2],label = 'SFG', marker='*',edgecolor='black')
					
    sns.lineplot(x=wise_x,y=assef_sel(wise_x),label='Assef et al. (2018)',color='black')
    plt.ylim(-1,2)
    plt.xlabel(r'$\rm W_{2}\,(mag)$', fontsize=13)
    plt.ylabel(r'$\rm W_{1}-W_{2}\,(mag)$', fontsize=13)
    
    return
	
def mat_plot(gem_1, w1, w2, w3):
    x = np.linspace(0,7,len(gem_1))
    y_b, y_t, y_l = mateos_wedge(x)
    
    sns.scatterplot(x = gem_1[w2]-gem_1[w3], y = gem_1[w1]-gem_1[w2], hue=gem_1.mat_index)
    
    sns.lineplot(x = x, y = y_b, color = 'black')
    sns.lineplot(x = x, y = y_t, color = 'black')
    sns.lineplot(x = x, y = y_l, color = 'black')
    
    plt.xlim(-1,7.5)
    plt.ylim(-2,3)
    
    plt.xlabel(r'$\rm W_{2} - W_{3}\,(mag)$', fontsize=13)
    plt.ylabel(r'$\rm W_{1} - W_{2}\,(mag)$', fontsize=13)
    return

def ki_plot(gem_1, K, w2, w3):
    x = np.linspace(0,6,len(gem_1))
    
    sns.scatterplot(x=gem_1[gem_1.ki_index==1][K] - gem_1[gem_1.ki_index==1][w2], 
                    y=gem_1[gem_1.ki_index==1][w2]- gem_1[gem_1.ki_index==1][w3],
                    label = 'AGN')
    
    sns.scatterplot(x=gem_1[gem_1.ki_index==0][K] - gem_1[gem_1.ki_index==0][w2], 
                    y=gem_1[gem_1.ki_index==0][w2]- gem_1[gem_1.ki_index==0][w3],
                    label = 'SFG')
    plt.axhline(y=1.835, xmin=-1,xmax=7)
    plt.axvline(x=1.489, ymin=-1,ymax=7)
    
    plt.xlabel(r'$K - W_{2}$',fontsize=12)
    plt.ylabel(r'$W_{2} - W_{3}$',fontsize=12)
    plt.xlim(-1.5,5)
    plt.ylim(-1,5)
    plt.legend(fontsize=12)
    return
    
def fix_col(gem_1):
    names = []
    for i in range(len(gem_1)):
        names.insert(i,gem_1.component_id.iloc[i].split(' ')[-1])
    
    gem_1.component_id = names

    return gem_1
    
g09 = fix_col(g09)
#g23 = fix_col(g23)

test_g09 = g09_allwise.merge(g09[['component_id','assef_index']],how='inner',on=['component_id'])
test_g23 = g23_allwise.merge(g23[['Source_Name','assef_index']],how='inner',on=['Source_Name'])

# Assef diagnostic
plt.figure(figsize=(10,6))    

plt.subplot(121)
wise_plot(g09,w1='W1mproPM',w2='W2mproPM')
plt.title('G09')
plt.subplot(122)
wise_plot(g23,w1='W1mproPM',w2='W2mproPM')
plt.title('G23')
plt.show(block=False)
plt.pause(2)
plt.close()

# Mateos classification
plt.figure(figsize=(10,6))    

plt.subplot(121)
mat_plot(g09_allwise,w1='W1mag',w2='W2mag', w3='W3mag')
sns.kdeplot(x = test_g09.W2mag - test_g09.W3mag, y = test_g09.W1mag - test_g09.W2mag, hue = test_g09.assef_index,
palette=['black','blue'])
plt.title('G09')
plt.subplot(122)
mat_plot(g23_allwise,w1='W1mag',w2='W2mag', w3='W3mag')
sns.kdeplot(x = test_g23.W2mag - test_g23.W3mag, y = test_g23.W1mag - test_g23.W2mag, 
            hue = test_g23.assef_index,palette=['black','blue'])
plt.title('G23')
plt.show(block=False)
plt.pause(2)
plt.close()

test_g09 = g09_ki.merge(g09[['component_id','assef_index']],how='inner',on=['component_id'])
test_g23 = g23_ki.merge(g23[['Source_Name','assef_index']],how='inner',on=['Source_Name'])

# KI classification

g09_opt = pd.read_csv('G09_opt_indices.csv')
g23_opt = pd.read_csv('G23_opt_indices.csv')
g09_opt = fix_col(g09_opt)
ki_optg09 = g09_ki.merge(g09_opt[['component_id','BPT_index']],how = 'inner',on=['component_id'])
ki_optg23 = g23_ki.merge(g23_opt[['Source_Name','BPT_index']],how = 'inner',on=['Source_Name'])


plt.figure(figsize=(10,6))   
plt.subplot(221)
ki_plot(g09_ki,'Kmag','W2mag','W3mag')
#sns.kdeplot(x=g09_ki.Kmag-g09_ki.W2mag,y=g09_ki.W2mag-g09_ki.W3mag,hue=g09_ki.mat_index,
#            palette=['black','blue'])
sns.kdeplot(x=test_g09.Kmag-test_g09.W2mag,y=test_g09.W2mag-test_g09.W3mag,hue=test_g09.assef_index,
            palette=['blue','red'])
plt.title('G09')

plt.subplot(222)
ki_plot(g23_ki,'Kmag','W2mag','W3mag')
#sns.kdeplot(x=g23_ki.Kmag-g23_ki.W2mag,y=g23_ki.W2mag-g23_ki.W3mag,hue=g23_ki.mat_index,
#            palette=['black','blue'])
sns.kdeplot(x=test_g23.Kmag-test_g23.W2mag,y=test_g23.W2mag-test_g23.W3mag,hue=test_g23.assef_index,
            palette=['blue','red'])
            
plt.title('G23')

plt.subplot(223)
sns.kdeplot(x=ki_optg09.Kmag-g09_ki.W2mag,y=ki_optg09.W2mag-ki_optg09.W3mag,hue=ki_optg09.BPT_index,
            palette=['black','blue','red'])
plt.axhline(y=1.835, xmin=-1,xmax=7)
plt.axvline(x=1.489, ymin=-1,ymax=7)

plt.xlim(-1.5,5)
plt.ylim(-1,5)

plt.subplot(224)
sns.kdeplot(x=ki_optg23.Kmag-ki_optg23.W2mag,y=ki_optg23.W2mag-ki_optg23.W3mag,hue=ki_optg23.BPT_index,
            palette=['black','blue','red'])
plt.axhline(y=1.835, xmin=-1,xmax=7)
plt.axvline(x=1.489, ymin=-1,ymax=7)

plt.xlim(-1.5,5)
plt.ylim(-1,5)

plt.show(block=False)
#plt.pause(2)
#plt.close()

#---------------#
# Classified data
#---------------#
cols = ['island_id', 'component_id', 'component_name', 'ra_catwise','dec_catwise','ra_deg_cont','dec_deg_cont',
        'W1mproPM','W2mproPM','assef_index','W1mag','W2mag','W3mag','W4mag','mat_index']
collated_g09_catwise = g09[cols[:10]]
collated_g09_allwise = g09_allwise[cols[:6]+cols[10:]]

cols = ['Source_Name','RA', 'DEC','RAdeg','DEdeg','W1mproPM','W2mproPM','assef_index','RAJ2000','DEJ2000',
        'W1mag','W2mag','W3mag','W4mag','mat_index']
collated_g23_catwise = g23[cols[:8]]
collated_g23_allwise = g23_allwise[cols[:3]+cols[8:]]


collated_g09_catwise.to_csv('G09_catwise_indices.csv')
collated_g23_catwise.to_csv('G23_catwise_indices.csv')
collated_g09_allwise.to_csv('G09_allwise_indices.csv')
collated_g23_allwise.to_csv('G23_allwise_indices.csv')
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
	
def mat_classify(gem_1, w1, w2, w3):
    mat_col = np.zeros(len(gem_1))
    y_b, y_t, y_l = mateos_wedge(gem_1[w2]-gem_1[w3])
    mat_col[((gem_1[w1]-gem_1[w2])>=y_b)&((gem_1[w1]-gem_1[w2])<=y_t)&((gem_1[w1]-gem_1[w2])>=y_l)] = 1
    
    gem_1['mat_index'] = mat_col
    return gem_1
    
    
g09 = classify(g09,w1='W1mproPM',w2='W2mproPM')
g23 = classify(g23,w1='W1mproPM',w2='W2mproPM')
g09_allwise = mat_classify(g09_allwise,'W1mag','W2mag','W3mag')
g23_allwise = mat_classify(g23_allwise,'W1mag','W2mag','W3mag')


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
    plt.xlabel(r'$\rm W_{2}\,(mag))$', fontsize=13)
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

plt.figure(figsize=(10,6))    

plt.subplot(121)
wise_plot(g09,w1='W1mproPM',w2='W2mproPM')
plt.title('G09')
plt.subplot(122)
wise_plot(g23,w1='W1mproPM',w2='W2mproPM')
plt.title('G23')
plt.show()#block=False)
#plt.pause(2)
#plt.close()

plt.figure(figsize=(10,6))    

plt.subplot(121)
mat_plot(g09_allwise,w1='W1mag',w2='W2mag', w3='W3mag')
sns.kdeplot(x = test_g09.W2mag - test_g09.W3mag, y = test_g09.W1mag - test_g09.W2mag, hue = test_g09.assef_index,
palette=['black','blue'])
plt.title('G09')
plt.subplot(122)
mat_plot(g23_allwise,w1='W1mag',w2='W2mag', w3='W3mag')
sns.kdeplot(x = test_g23.W2mag - test_g23.W3mag, y = test_g23.W1mag - test_g23.W2mag, hue = test_g23.assef_index,
palette=['black','blue'])
plt.title('G23')
plt.show(block=False)

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Definitions
#
def FullData(fmat):
    """
    -------------------------------------------------------------
    Access and fully load the *.mat and return a dictionary
    -------------------------------------------------------------
    OUT   dset    Dictionary containing all the data within *.mat
    IN    fmat    Filename
    -------------------------------------------------------------
    """
    # Load the data
    dataset = scipy.io.loadmat(fmat)

    # Initiate dictionary
    dset = {}

    for k in dataset.keys():     # Loop over keys
        if  k.startswith('__'):  # Skip the special keys
            continue
        dset[k] = dataset[k]
    return dset
  

def Plot_exp(dset):  #,ititle,name,path):
    """
    -------------------------------------------------------------
    Experiment comparison "internal":
    -------------------------------------------------------------
    OUT    name.png   Plot stored at the given path
    IN     d1         ARTS outputs
           ititle     Title of figure
           name       Name to store the figure
           path       Path to store the figure
    -------------------------------------------------------------
    """
                 
    plt.matplotlib.rc('font', family='serif', size = 12)
    fig = plt.figure(figsize=(9,6))
    plt.plot(dset['D']['f_grid'][0][0][0][0][0][0]/1e9, dset['arts_tb'][0,:]-dset['arts_cl'][0,:],linewidth=1,color='darkblue', label = 'Heavy Rain Only')
    plt.title( 'Cloud Scenarios', fontsize='12', fontweight='bold')
    plt.ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
    plt.xlabel(r'Frequency [GHz]', color='k')
    plt.grid('true')
    plt.axvline(x=10 ,ls='-',color='k')
    plt.axvline(x=19 ,ls='-',color='k')
    plt.axvline(x=22 ,ls='-',color='k')
    plt.axvline(x=37 ,ls='-',color='k')
    plt.axvline(x=85 ,ls='-',color='k')
    plt.axvline(x=166 ,ls='-',color='k')
    plt.xlim([0,175])

    
    #fig.suptitle(str(ititle) ,fontweight='bold' )
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.899)
    #plt.savefig(path+'/'+str(name)+'.png')
    #plt.close()
    return
  
  
  
main_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/RT_simulations/Output/'
  
# HEAVY RAIN ONLY
f_arts    = main_dir + 'RainOnly_HR_exp1/' + 'GMI_Fascod_RainOnly_HR_exp1.mat'
arts_exp1 = FullData(f_arts)
Plot_exp(arts_exp1)
  

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
  

def Plot_exp(dset1, dset2, dset3, dset4, dset5):  #,ititle,name,path):
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
    
    f_grid = dset1['D']['f_grid'][0][0][0][0][0][0]/1e9
                 
    plt.matplotlib.rc('font', family='serif', size = 12)
    fig = plt.figure(figsize=(9,6))
    plt.plot(f_grid, dset1['arts_tb'][0,:]-dset1['arts_cl'][0,:],linewidth=1,color='darkblue', label = 'Heavy Rain Only')
    plt.plot(f_grid, dset2['arts_tb'][0,:]-dset2['arts_cl'][0,:],linewidth=1,color='cyan', label = 'Light Rain Only')
    plt.plot(f_grid, dset3['arts_tb'][0,:]-dset3['arts_cl'][0,:],linewidth=1,color='darkred', label = 'Graupel Only (ICON)')
    plt.plot(f_grid, dset4['arts_tb'][0,:]-dset4['arts_cl'][0,:],linewidth=1,color='red', label = 'Graupel Only (GEM)')
    plt.plot(f_grid, dset5['arts_tb'][0,:]-dset5['arts_cl'][0,:],linewidth=1,color='magenta', label = 'Graupel Only (Evans Agg.)')

    
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
    plt.legend()

    
    #fig.suptitle(str(ititle) ,fontweight='bold' )
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.899)
    #plt.savefig(path+'/'+str(name)+'.png')
    #plt.close()
    return
 
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
main_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/RT_simulations/Output/'
  
# HEAVY RAIN ONLY
f_arts    = main_dir + 'RainOnly_HR_exp1/' + 'GMI_Fascod_RainOnly_HR_exp1.mat'
arts_exp1 = FullData(f_arts)

# LIGHT RAIN ONLY
f_arts    = main_dir + 'RainOnly_LR_exp2/' + 'GMI_Fascod_RainOnly_LR_exp2.mat'
arts_exp2 = FullData(f_arts)

# GRAUPEL ONLY (TEST 1A)
f_arts    = main_dir + 'GraupelOnly_ICON_exp3a_field07/' + 'GMI_Fascod_GraupelOnly_ICON_exp3a_field07.mat'
arts_exp3 = FullData(f_arts)

# GRAUPEL ONLY (TEST 1A)
f_arts    = main_dir + 'GraupelOnly_GEM_exp3a_field07/' + 'GMI_Fascod_GraupelOnly_GEM_exp3a_field07.mat'
arts_exp3gem = FullData(f_arts)

# GRAUPEL ONLY (TEST 1A)
f_arts    = main_dir + 'GraupelOnly_EvansAgg_exp3a_field07/' + 'GMI_Fascod_GraupelOnly_EvansAgg_exp3a_field07.mat'
arts_exp3Evans = FullData(f_arts)

#---- PLOT
Plot_exp(arts_exp1, arts_exp2, arts_exp3, arts_exp3gem,arts_exp3Evans )



  

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
  

def Plot_exp(dset1, dset2):  #,ititle,name,path):
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
 


def Plot_exp_GRAUPEL(dd, dset1, dset2, dset3):  #,ititle,name,path):
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
    
    f_grid = dd['D']['f_grid'][0][0][0][0][0][0]/1e9
                 
    plt.matplotlib.rc('font', family='serif', size = 12)
    fig = plt.figure(figsize=(9,6))    
    
    # EXPA
    plt.plot(f_grid, dset1[0],linewidth=1,color='darkblue', label = '8-ColAgg')
    plt.plot(f_grid, dset1[1],linewidth=1,color='blue', label = 'Column1')
    plt.plot(f_grid, dset1[2],linewidth=1,color='darkred', label = 'EvansAgg')
    plt.plot(f_grid, dset1[3],linewidth=1,color='red', label = 'LargeBlockAgg')
    plt.plot(f_grid, dset1[4],linewidth=1,color='darkgreen', label = 'LargeColumnAgg')

    
    plt.plot(np.nan, linewidth=1,color='k', label = 'EXP A')
    plt.plot(np.nan, linewidth=1,color='k', label = 'EXP B', linestyle='--')
    plt.plot(np.nan, linewidth=1,color='k', label = 'EXP C', linestyle=':')
    plt.legend()

    
    # EXPB
    plt.plot(f_grid, dset2[0],linewidth=1,color='darkblue', label = '8-ColAgg', linestyle='--')
    plt.plot(f_grid, dset2[1],linewidth=1,color='blue', label = 'Column1', linestyle='--')
    plt.plot(f_grid, dset2[2],linewidth=1,color='darkred', label = 'EvansAgg', linestyle='--')
    plt.plot(f_grid, dset2[3],linewidth=1,color='red', label = 'LargeBlockAgg', linestyle='--')
    plt.plot(f_grid, dset2[4],linewidth=1,color='darkgreen', label = 'LargeColumnAgg', linestyle='--')
    
    # EXPC 
    plt.plot(f_grid, dset3[0],linewidth=1,color='darkblue', label = '8-ColAgg', linestyle=':')
    plt.plot(f_grid, dset3[1],linewidth=1,color='blue', label = 'Column1', linestyle=':')
    plt.plot(f_grid, dset3[2],linewidth=1,color='darkred', label = 'EvansAgg', linestyle=':')
    plt.plot(f_grid, dset3[3],linewidth=1,color='red', label = 'LargeBlockAgg', linestyle=':')
    plt.plot(f_grid, dset3[4],linewidth=1,color='darkgreen', label = 'LargeColumnAgg', linestyle=':')
    
    
    plt.title( 'Cloud Scenarios (Graupel-only)', fontsize='12', fontweight='bold')
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

   
    return


#------------------------------------------------------------------------------------------
# only rain
#------------------------------------------------------------------------------------------
main_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/RT_simulations/Output/'
  
# HEAVY RAIN ONLY
f_arts    = main_dir + 'RainOnly_HR_exp1/' + 'GMI_Fascod_RainOnly_HR_exp1.mat'
arts_exp1 = FullData(f_arts)

# LIGHT RAIN ONLY
f_arts    = main_dir + 'RainOnly_LR_exp2/' + 'GMI_Fascod_RainOnly_LR_exp2.mat'
arts_exp2 = FullData(f_arts)

# ----------------------------- plot
Plot_exp(arts_exp1, arts_exp2)

plt.matplotlib.rc('font', family='serif', size = 12)
fig = plt.figure(figsize=(9,6))    
plt.plot( arts_exp1['D']['particle_bulkprop_field'][0][0][0][0][0][0], arts_exp1['D']['z_field'][0][0][0][0][0] / 1e3, linewidth=1,color='darkblue', label = 'Heavy Rain Only')
plt.plot( arts_exp2['D']['particle_bulkprop_field'][0][0][0][0][0][0] , arts_exp2['D']['z_field'][0][0][0][0][0]/ 1e3, linewidth=1,color='cyan', label = 'Light Rain Only')
plt.title( 'Cloud Scenarios (Rain-Only)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'mass content [g/m3]', color='k')
plt.grid('true')
plt.legend()
plt.ylim([0,20])



#------------------------------------------------------------------------------------------
# only graupel
#------------------------------------------------------------------------------------------

graupels_folder = ['8-ColumnAggregate', 'ColumnType1', 'EvansSnowAggregate', 
                  'LargeBlockAggregate', 'LargeColumnAggregate'] 

# GRAUPEL ONLY (TEST 1A)
graupel_tbs_exp1 = []
expA_wc          = []
for i_habits in graupels_folder: 
    f_arts    = main_dir + 'GraupelOnly_ssd_' + i_habits + '/GMI_Fascod_GraupelOnly_ssd_'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    graupel_tbs_exp1.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expA_wc.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])
z_field = arts_exp['D']['z_field'][0][0][0][0][0] / 1e3


# GRAUPEL ONLY (TEST 1B)
graupel_tbs_expB = []
expB_wc          = []
for i_habits in graupels_folder: 
    f_arts    = main_dir + 'GraupelOnly_ssd_EXPB' + i_habits + '/GMI_Fascod_GraupelOnly_ssd_EXPB'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    graupel_tbs_expB.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expB_wc.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])

# GRAUPEL ONLY (TEST 1C)
graupel_tbs_expC = []
expC_wc          = []
for i_habits in graupels_folder: 
    f_arts    = main_dir + 'GraupelOnly_ssd_EXPC' + i_habits + '/GMI_Fascod_GraupelOnly_ssd_EXPC'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    graupel_tbs_expC.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expC_wc.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])
    
#---- PLOT
Plot_exp_GRAUPEL(arts_exp, graupel_tbs_exp1, graupel_tbs_expB, graupel_tbs_expC )

plt.matplotlib.rc('font', family='serif', size = 12)
fig = plt.figure(figsize=(9,6))    
plt.plot( expA_wc[0], z_field, linewidth=1,color='k', label = 'EXP A', linestyle='-')
plt.plot( expB_wc[1], z_field, linewidth=1,color='k', label = 'EXP B', linestyle='--')
plt.plot( expC_wc[2], z_field, linewidth=1,color='k', label = 'EXP C', linestyle=':')
plt.title( 'Cloud Scenarios (Rain-Only)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'mass content [g/m3]', color='k')
plt.grid('true')
plt.legend()
plt.ylim([0,20])





#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

  

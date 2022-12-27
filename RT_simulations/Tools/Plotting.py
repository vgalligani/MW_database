import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Definitions
#
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
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
  
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
def Plot_exp(dset1, dset2, labels):  #,ititle,name,path):
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
    plt.plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dset1['arts_tb'][0,:]-dset1['arts_cl'][0,:],linewidth=1,color='darkblue', label = 'Heavy Rain Only')
    plt.plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dset2['arts_tb'][0,:]-dset2['arts_cl'][0,:],linewidth=1,color='cyan', label = 'Light Rain Only')
    
    plt.title( 'Cloud Scenarios (Rain-only)', fontsize='12', fontweight='bold')
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
#------------------------------------------------------------------------------------------
def Plot_exp_RainGrau(dd, dset1, dset2, dset3, infotitle):  #,ititle,name,path):
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
    plt.plot(f_grid, dset1[1],linewidth=1,color='darkred', label = 'EvansAgg')
    plt.plot(f_grid, dset1[2],linewidth=1,color='red', label = 'LargeBlockAgg')

    plt.plot(np.nan, linewidth=1,color='k', label = 'EXP A')
    plt.plot(np.nan, linewidth=1,color='k', label = 'EXP B', linestyle='--')
    plt.plot(np.nan, linewidth=1,color='k', label = 'EXP C', linestyle=':')
    plt.legend()
    
    # EXPB
    plt.plot(f_grid, dset2[0],linewidth=1,color='darkblue', label = '8-ColAgg', linestyle='--')
    plt.plot(f_grid, dset2[1],linewidth=1,color='darkred', label = 'EvansAgg', linestyle='--')
    plt.plot(f_grid, dset2[2],linewidth=1,color='red', label = 'LargeBlockAgg', linestyle='--')
    
    # EXPC 
    plt.plot(f_grid, dset3[0],linewidth=1,color='darkblue', label = '8-ColAgg', linestyle=':')
    plt.plot(f_grid, dset3[1],linewidth=1,color='darkred', label = 'EvansAgg', linestyle=':')
    plt.plot(f_grid, dset3[2],linewidth=1,color='red', label = 'LargeBlockAgg', linestyle=':')
    
    plt.title( 'Cloud Scenarios ('+infotitle+' + Graup.)', fontsize='12', fontweight='bold')
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
#------------------------------------------------------------------------------------------
def Plot_exp_hail(dset1, dset2, dset3, dset4, labels_in):  #,ititle,name,path):
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
    plt.plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dset1['arts_tb'][0,:]-dset1['arts_cl'][0,:],linewidth=1,color='darkblue', label = labels_in[0])
    plt.plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dset2['arts_tb'][0,:]-dset2['arts_cl'][0,:],linewidth=1,color='cyan', label = labels_in[1])
    plt.plot(dset3['D']['f_grid'][0][0][0][0][0][0]/1e9, dset3['arts_tb'][0,:]-dset3['arts_cl'][0,:],linewidth=1,color='darkred', label = labels_in[2])
    plt.plot(dset4['D']['f_grid'][0][0][0][0][0][0]/1e9, dset4['arts_tb'][0,:]-dset4['arts_cl'][0,:],linewidth=1,color='magenta', label = labels_in[3])

    plt.title( 'Cloud Scenarios (Hail-only)', fontsize='12', fontweight='bold')
    plt.ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
    plt.xlabel(r'Frequency [GHz]', color='k')
    plt.grid('true')
    plt.axvline(x=10 ,ls='-',color='k')
    plt.axvline(x=19 ,ls='-',color='k')
    plt.axvline(x=22 ,ls='-',color='k')
    plt.axvline(x=37 ,ls='-',color='k')
    plt.axvline(x=85 ,ls='-',color='k')
    plt.axvline(x=166 ,ls='-',color='k')
    plt.xlim([4,120])
    plt.legend()


    #fig.suptitle(str(ititle) ,fontweight='bold' )
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.899)
    #plt.savefig(path+'/'+str(name)+'.png')
    #plt.close()
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
plt.title( 'Cloud Scenarios (Grau-Only)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'mass content [g/m3]', color='k')
plt.grid('true')
plt.legend()
plt.ylim([0,20])

#------------------------------------------------------------------------------------------
# HR + GRAU 
#------------------------------------------------------------------------------------------
graupels_folder = ['8-ColumnAggregate', 'EvansSnowAggregate', 
                  'LargeBlockAggregate'] 

tbs_exp1     = []
expA_wc_rain = []
expA_wc_grau = []

tbs_exp2     = []
expB_wc_rain = []
expB_wc_grau = []

tbs_exp3     = []
expC_wc_rain = []
expC_wc_grau = []

for i_habits in graupels_folder: 
    f_arts    = main_dir + 'HRG_ssd_EXPa' + i_habits + '/GMI_Fascod_HRG_ssd_EXPa'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    tbs_exp1.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expA_wc_rain.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0,:])
    expA_wc_grau.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])

    f_arts    = main_dir + 'HRG_ssd_EXPb' + i_habits + '/GMI_Fascod_HRG_ssd_EXPb'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    tbs_exp2.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expB_wc_rain.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0,:])
    expB_wc_grau.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])

    f_arts    = main_dir + 'HRG_ssd_EXPc' + i_habits + '/GMI_Fascod_HRG_ssd_EXPc'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    tbs_exp3.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expC_wc_rain.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0,:])
    expC_wc_grau.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])
z_field = arts_exp['D']['z_field'][0][0][0][0][0] / 1e3

Plot_exp_RainGrau(arts_exp, tbs_exp1, tbs_exp2, tbs_exp3, 'HR' )

plt.matplotlib.rc('font', family='serif', size = 12)
fig = plt.figure(figsize=(9,6))    
plt.plot( expA_wc_rain[0], z_field, linewidth=1,color='k', label = 'rain', linestyle='-')
plt.plot( expA_wc_grau[0], z_field, linewidth=1,color='darkblue', label = 'exp A', linestyle='-')
plt.plot( expB_wc_grau[0], z_field, linewidth=1,color='darkblue', label = 'exp B', linestyle='--')
plt.plot( expC_wc_grau[0], z_field, linewidth=1,color='darkblue', label = 'exp C', linestyle=':')
plt.legend()
plt.title( 'Cloud Scenarios (HR+GRAU)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'mass content [g/m3]', color='k')
plt.grid('true')
plt.ylim([0,20])

#------------------------------------------------------------------------------------------
# LR + GRAU 
#------------------------------------------------------------------------------------------
graupels_folder = ['8-ColumnAggregate', 'EvansSnowAggregate', 
                  'LargeBlockAggregate'] 

tbs_exp1     = []
expA_wc_rain = []
expA_wc_grau = []

tbs_exp2     = []
expB_wc_rain = []
expB_wc_grau = []

tbs_exp3     = []
expC_wc_rain = []
expC_wc_grau = []

for i_habits in graupels_folder: 
    f_arts    = main_dir + 'LRG_ssd_EXPa' + i_habits + '/GMI_Fascod_LRG_ssd_EXPa'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    tbs_exp1.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expA_wc_rain.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0,:])
    expA_wc_grau.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])

    f_arts    = main_dir + 'LRG_ssd_EXPb' + i_habits + '/GMI_Fascod_LRG_ssd_EXPb'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    tbs_exp2.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expB_wc_rain.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0,:])
    expB_wc_grau.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])

    f_arts    = main_dir + 'LRG_ssd_EXPc' + i_habits + '/GMI_Fascod_LRG_ssd_EXPc'+ i_habits + '.mat'
    arts_exp  = FullData(f_arts)
    tbs_exp3.append( arts_exp['arts_tb'][0,:]-arts_exp['arts_cl'][0,:]) 
    expC_wc_rain.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0,:])
    expC_wc_grau.append( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1,:])
z_field = arts_exp['D']['z_field'][0][0][0][0][0] / 1e3

Plot_exp_RainGrau(arts_exp, tbs_exp1, tbs_exp2, tbs_exp3, 'LR' )

plt.matplotlib.rc('font', family='serif', size = 12)
fig = plt.figure(figsize=(9,6))    
plt.plot( expA_wc_rain[0], z_field, linewidth=1,color='k', label = 'rain', linestyle='-')
plt.plot( expA_wc_grau[0], z_field, linewidth=1,color='darkblue', label = 'exp A', linestyle='-')
plt.plot( expB_wc_grau[0], z_field, linewidth=1,color='darkblue', label = 'exp B', linestyle='--')
plt.plot( expC_wc_grau[0], z_field, linewidth=1,color='darkblue', label = 'exp C', linestyle=':')
plt.legend()
plt.title( 'Cloud Scenarios (LR+GRAU)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'mass content [g/m3]', color='k')
plt.grid('true')
plt.ylim([0,20])
  
   
#------------------------------------------------------------------------------------------
# Hail Only
#------------------------------------------------------------------------------------------ 
# simple cloudbox 1e-2 (1cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPa0.01/' + 'GMI_Fascod_HailOnly_ssd_EXPa0.01.mat'
arts_exp1 = FullData(f_arts)

# simple cloudbox 5e-2 (5cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPa0.05/' + 'GMI_Fascod_HailOnly_ssd_EXPa0.05.mat'
arts_exp2 = FullData(f_arts)

# Exponetial cloubox 1e-2 (1cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPb0.01/' + 'GMI_Fascod_HailOnly_ssd_EXPb0.01.mat'
arts_exp3 = FullData(f_arts)

# Exponetial cloudbox 5e-2 (5cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPb0.05/' + 'GMI_Fascod_HailOnly_ssd_EXPb0.05.mat'
arts_exp4 = FullData(f_arts)

# ----------------------------- plot
Plot_exp_hail(arts_exp1, arts_exp2, arts_exp3, arts_exp4, ['Simple cloudbox (mono 1cm)','Simple cloudbox (mono 5cm)','Exp cloudbox (mono 1cm)','Exp. cloudbox (mono 5cm)'])

mass_1 = 500 * 4 * 3.14 * (1e-2 **3) / 24;
mass_2 = 500 * 4 * 3.14 * (5e-2 **3) / 24;

plt.matplotlib.rc('font', family='serif', size = 12)
fig = plt.figure(figsize=(9,6))    
plt.plot( arts_exp1['D']['particle_bulkprop_field'][0][0][0][0][0][0]*mass_1, arts_exp1['D']['z_field'][0][0][0][0][0] / 1e3, linewidth=1,color='k', label = 'Simple cloudbox')
plt.plot( arts_exp3['D']['particle_bulkprop_field'][0][0][0][0][0][0]*mass_1, arts_exp3['D']['z_field'][0][0][0][0][0]/ 1e3, linewidth=1,color='gray', label = 'Exp. cloudbox')

plt.plot( arts_exp2['D']['particle_bulkprop_field'][0][0][0][0][0][0]*mass_2, arts_exp1['D']['z_field'][0][0][0][0][0] / 1e3, linewidth=1,color='k', label = 'Simple cloudbox')
plt.plot( arts_exp4['D']['particle_bulkprop_field'][0][0][0][0][0][0]*mass_2, arts_exp3['D']['z_field'][0][0][0][0][0]/ 1e3, linewidth=1,color='gray', label = 'Exp. cloudbox')

plt.title( 'Cloud Scenarios (Hail-Only)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'mass content [g/m3]', color='k')
plt.grid('true')
plt.legend()
plt.ylim([0,20])


# ssp (resosnance issues!?)
ssp_001 = FullData(main_dir + 'HailOnly_ssd_EXPb0.01/' + 'SSP.mat')
ext_mat_1cm = ssp_001['test']['ext_mat_data'][0][0][:,1]
abs_vec_1cm = ssp_001['test']['abs_vec_data'][0][0][:,1]
pha_mat_1cm = ssp_001['test']['pha_mat_data'][0][0][:,1,:,0,0,0,:]

ssp_005 = FullData(main_dir + 'HailOnly_ssd_EXPb0.05/' + 'SSP.mat')
ext_mat_5cm = ssp_005['test']['ext_mat_data'][0][0][:,1]
abs_vec_5cm = ssp_005['test']['abs_vec_data'][0][0][:,1]
pha_mat_5cm = ssp_005['test']['pha_mat_data'][0][0][:,1,:,0,0,0,:]

fig = plt.figure(figsize=(9,6))    
plt.semilogy(ssp_001['test']['f_grid'][0][0][:]/1e9,  ext_mat_1cm-abs_vec_1cm, 'darkgreen', label = '1cm')
plt.semilogy(ssp_005['test']['f_grid'][0][0][:]/1e9, ext_mat_5cm-abs_vec_5cm, 'darkblue', label = '5cm')

plt.semilogy(ssp_001['test']['f_grid'][0][0][:]/1e9, abs_vec_1cm, 'darkgreen', linestyle='--')
plt.semilogy(ssp_005['test']['f_grid'][0][0][:]/1e9, abs_vec_5cm, 'darkblue', linestyle='--')

plt.title('Single Scttering Properties (Hail-Only)', fontsize='12', fontweight='bold')
plt.ylabel(r'Frequency', color='k')
plt.xlabel(r'SSP', color='k')
plt.grid('true')
plt.legend()
plt.xlim([4,120])
       

         
         
         







#------------------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------------------ 
# Effective density
def calc_rho_eff( a, b):
    
    Dmax    = np.arange(1e-6,1e-2, 100e-6)    
    rho_eff = []
    fixed_rho = []
    for idmax in Dmax: 
        rho_eff_ = a*(idmax**b)
        rho_eff.append( rho_eff_ / ( (3.14/6)* (idmax**3) )) 
        fixed_rho.append(917)
    return rho_eff

Dmax      = np.arange(1e-6,1e-2, 100e-6) 
fixed_rho = []
fixed_rho2 = []
for idmax in Dmax: 
    fixed_rho.append(917)
    fixed_rho2.append(400)

    
fig = plt.figure(figsize=(9,6))    
plt.loglog( np.arange(1e-6,1e-2, 100e-6), calc_rho_eff( 65.4, 3)    ,color='darkblue', label = '8-ColAgg' )
plt.loglog( np.arange(1e-6,1e-2, 100e-6), calc_rho_eff( 0.038, 2.05),color='blue', label = 'Column1' )
plt.loglog( np.arange(1e-6,1e-2, 100e-6), calc_rho_eff( 0.20, 2.39) ,color='darkred', label = 'EvansAgg' )
plt.loglog( np.arange(1e-6,1e-2, 100e-6), calc_rho_eff( 0.35, 2.27) ,color='red', label = 'LargeBlockAgg' )
plt.loglog( np.arange(1e-6,1e-2, 100e-6), calc_rho_eff( 0.28, 2.44),color='darkgreen', label = 'LargeColumnAgg' )
plt.loglog( np.arange(1e-6,1e-2, 100e-6), fixed_rho,color='black', label = 'Ice density (917)' )
plt.loglog( np.arange(1e-6,1e-2, 100e-6), fixed_rho2,color='black', linestyle='--', label = 'Graupel density (400)' )

plt.ylim([10,5000])
plt.xlabel('Dmax [m]')
plt.ylabel('Effective Density [kg/m]')
plt.grid(True)
plt.legend()

    
 

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
def Plot_TWOexps(dset1, dset2, dhailset, dhailset_mass, colorcycle): #,ititle,name,path):
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
    
    for i in range(dhailset.shape[1]):
        plt.plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dhailset[:,i,0] - dhailset[:,i,1], 
                 linewidth=1, color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass)+' kg/m$^2$')
        
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
    plt.xlim([5,100])
    
    plt.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.35))
    #fig.suptitle(str(ititle) ,fontweight='bold' )
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.899)
    #plt.savefig(path+'/'+str(name)+'.png')
    #plt.close()
    return
 
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
def Plot_TWOexps_wCombinedexps(dset1, dset2, dhailset, dhailset_mass, dcombined1, dcombined2, colorcycle, freqLim): #,ititle,name,path):
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
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=[9,12])
    axes[0].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dset1['arts_tb'][0,:]-dset1['arts_cl'][0,:],linewidth=1,color='darkblue', label = 'Heavy Rain (HR) Only')
    axes[0].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dset2['arts_tb'][0,:]-dset2['arts_cl'][0,:],linewidth=1,color='cyan', label = 'Light Rain (LR) Only')
    
    for i in range(dhailset.shape[1]):
        axes[0].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dhailset[:,i,0] - dhailset[:,i,1], 
                 linewidth=1, color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass[i])+' kg/m$^2$')
        
    axes[0].set_title( 'Cloud Scenarios (Individual species)', fontsize='12', fontweight='bold')
    axes[0].set_ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
    axes[0].grid('true')
    axes[0].axvline(x=10 ,ls='-',color='k')
    axes[0].axvline(x=19 ,ls='-',color='k')
    axes[0].axvline(x=22 ,ls='-',color='k')
    axes[0].axvline(x=37 ,ls='-',color='k')
    axes[0].axvline(x=85 ,ls='-',color='k')
    axes[0].axvline(x=166 ,ls='-',color='k')
    axes[0].set_xlim([5,freqLim])
    plt.legend(ncol=3)
    #plt.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.35))
    # ADD LEGEND
    p2 = axes[0].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p2[0], p2[1]-0.65, p2[2], p2[3]/10])   # [left, bottom, width, height] or Bbox  
    plt.plot( np.nan, np.nan, linewidth=1, color='darkblue', label = 'Heavy Rain Only')
    plt.plot(  np.nan, np.nan, linewidth=1, color='cyan', label = 'Light Rain Only')
    for i in range(dhailset.shape[1]):
        plt.plot( np.nan, np.nan, linewidth=1, color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass[i])+' kg/m$^2$')
    plt.legend(ncol=2)
    ax_cbar.axis('off')
    plt.legend(ncol=3)
    

    for i in range(dhailset.shape[1]):
        axes[1].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dcombined1[:,i,0] - dcombined1[:,i,1], 
                 linewidth=1, linestyle='-', color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass[i])+' kg/m$^2$')    
        axes[1].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dcombined2[:,i,0] - dcombined2[:,i,1], 
                 linewidth=1, linestyle='--', color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass[i])+' kg/m$^2$')            
 
    axes[1].set_title( 'Cloud Scenarios (Combined RWC+HWC)', fontsize='12', fontweight='bold')
    axes[1].set_xlabel(r'Frequency [GHz]', color='k')
    axes[1].set_ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
    axes[1].grid('true')
    axes[1].axvline(x=10 ,ls='-',color='k')
    axes[1].axvline(x=19 ,ls='-',color='k')
    axes[1].axvline(x=22 ,ls='-',color='k')
    axes[1].axvline(x=37 ,ls='-',color='k')
    axes[1].axvline(x=85 ,ls='-',color='k')
    axes[1].axvline(x=166 ,ls='-',color='k')
    axes[1].set_xlim([5,freqLim])
    #plt.legend(ncol=3)
    
    # ADD LEGEND
    p2 = axes[1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p2[0]-0.08, p2[1]-0.08, p2[2], 0.04])   # [left, bottom, width, height] or Bbox 
    ax_cbar.plot(np.nan, np.nan, linewidth=1, linestyle='-', color='k', label='w/ HR')
    ax_cbar.plot(np.nan, np.nan, linewidth=1, linestyle='--', color='k', label='w/ LR') 
    ax_cbar.axis('off')
    plt.legend(ncol=3)
    

    #fig.suptitle(str(ititle) ,fontweight='bold' )
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.899)
    #plt.savefig(path+'/'+str(name)+'.png')
    #plt.close()
    return
    
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------    
def Plot_THREEexps_wCombinedexps(dset1, dset2, dhailset, dhailset_mass, dcombined1, dcombined2, dgrau1, dgrau2, dgrau3, dgrau_SSP, colorcycle, freqLim):
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
    colorcycle_greens = ['darkgreen', 'teal', 'darkslategray']
    f_grid = dset1['D']['f_grid'][0][0][0][0][0][0]/1e9
                 
    plt.matplotlib.rc('font', family='serif', size = 12)
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=[9,12])
    axes[0].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dset1['arts_tb'][0,:]-dset1['arts_cl'][0,:],linewidth=1,color='darkblue', label = 'Heavy Rain (HR) Only')
    axes[0].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dset2['arts_tb'][0,:]-dset2['arts_cl'][0,:],linewidth=1,color='cyan', label = 'Light Rain (LR) Only')
    
    for i in range(dhailset.shape[1]):
        axes[0].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dhailset[:,i,0] - dhailset[:,i,1], 
                 linewidth=1, color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass[i])+' kg/m$^2$')
    for i in range(dgrau1.shape[1]):
        axes[0].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau1[:,i,0] - dgrau1[:,i,1], 
                 linewidth=1, linestyle='-.', color=colorcycle_greens[i], label = r'Grau Only ('+ dgrau_SSP[i]+')')        
        axes[0].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau2[:,i,0] - dgrau2[:,i,1], 
                 linewidth=1, linestyle='--', color=colorcycle_greens[i], label = r'Grau Only ('+ dgrau_SSP[i]+')')   
        axes[0].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau3[:,i,0] - dgrau3[:,i,1], 
                 linewidth=1, linestyle='--', color=colorcycle_greens[i], label = r'Grau Only ('+ dgrau_SSP[i]+')')   
        
    axes[0].set_title( 'Cloud Scenarios (Individual species)', fontsize='12', fontweight='bold')
    axes[0].set_ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
    axes[0].grid('true')
    axes[0].axvline(x=10 ,ls='-',color='gray')
    axes[0].axvline(x=19 ,ls='-',color='gray')
    axes[0].axvline(x=22 ,ls='-',color='gray')
    axes[0].axvline(x=37 ,ls='-',color='gray')
    axes[0].axvline(x=85 ,ls='-',color='gray')
    axes[0].axvline(x=166 ,ls='-',color='gray')
    axes[0].set_xlim([5,freqLim])
    plt.legend(ncol=3)
    #plt.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.35))
    
    # ADD LEGEND
    p2 = axes[0].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p2[0], p2[1]-0.65, p2[2], p2[3]/10])   # [left, bottom, width, height] or Bbox  
    plt.plot( np.nan, np.nan, linewidth=1, color='darkblue', label = 'Heavy Rain Only')
    plt.plot(  np.nan, np.nan, linewidth=1, color='cyan', label = 'Light Rain Only')
    for i in range(dhailset.shape[1]):
        plt.plot( np.nan, np.nan, linewidth=1, color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass[i])+' kg/m$^2$')
    for i in range(dgrau1.shape[1]):
        plt.plot(np.nan, np.nan, linewidth=1, color=colorcycle_greens[i], label = r'Grau Only ('+ dgrau_SSP[i]+')')   
    plt.legend(ncol=2)
    ax_cbar.axis('off')
    plt.legend(ncol=3)
  
    #--- axes 2 para combined experiments. 
    for i in range(dhailset.shape[1]):
        axes[1].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dcombined1[:,i,0] - dcombined1[:,i,1], 
                 linewidth=1, linestyle='-', color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass[i])+' kg/m$^2$')    
        axes[1].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dcombined2[:,i,0] - dcombined2[:,i,1], 
                 linewidth=1, linestyle='--', color=colorcycle[i], label = r'Hail Only '+str(dhailset_mass[i])+' kg/m$^2$')            
 
    axes[1].set_title( 'Cloud Scenarios (Combined RWC+HWC)', fontsize='12', fontweight='bold')
    axes[1].set_xlabel(r'Frequency [GHz]', color='k')
    axes[1].set_ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
    axes[1].grid('true')
    axes[1].axvline(x=10 ,ls='-',color='k')
    axes[1].axvline(x=19 ,ls='-',color='k')
    axes[1].axvline(x=22 ,ls='-',color='k')
    axes[1].axvline(x=37 ,ls='-',color='k')
    axes[1].axvline(x=85 ,ls='-',color='k')
    axes[1].axvline(x=166 ,ls='-',color='k')
    axes[1].set_xlim([5,freqLim])
    #plt.legend(ncol=3)
    
    # ADD LEGEND
    p2 = axes[1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p2[0]-0.08, p2[1]-0.08, p2[2], 0.04])   # [left, bottom, width, height] or Bbox 
    ax_cbar.plot(np.nan, np.nan, linewidth=1, linestyle='-', color='k', label='w/ HR')
    ax_cbar.plot(np.nan, np.nan, linewidth=1, linestyle='--', color='k', label='w/ LR') 
    ax_cbar.axis('off')
    plt.legend(ncol=3)
    

    #fig.suptitle(str(ititle) ,fontweight='bold' )
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.899)
    #plt.savefig(path+'/'+str(name)+'.png')
    #plt.close()
    return
 
   
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------    
def Plot_Individual_exps_paper(dset1, dset2, dhailset, dhailset_mass, dcombined1, dcombined2, dgrau1, dgrau2, dgrau3, dgrau_SSP, colorcycle, freqLim):
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
    colorcycle_greens = ['darkgreen', 'teal', 'darkslategray']
    f_grid = dset1['D']['f_grid'][0][0][0][0][0][0]/1e9
                 
    plt.matplotlib.rc('font', family='serif', size = 12)
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, figsize=[9,12])
    
    #- axes[0] rwc only
    axes[0].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dset1['arts_tb'][0,:]-dset1['arts_cl'][0,:],linewidth=1,color='darkblue', label = 'Heavy Rain (HR) Only')
    axes[0].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dset2['arts_tb'][0,:]-dset2['arts_cl'][0,:],linewidth=1,color='cyan', label = 'Light Rain (LR) Only')
    axes[0].set_title('RWC only', fontsize='12', fontweight='bold')
    axes[0].legend(loc='upper right')
   
    # axes[1] hwc only 
    for i in range(dhailset.shape[1]):
        axes[1].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dhailset[:,i,0] - dhailset[:,i,1], 
                 linewidth=1, color=colorcycle[i], label = r'HWP: '+str(dhailset_mass[i])+' kg/m$^2$')
    axes[1].set_title('HWC only', fontsize='12', fontweight='bold')
        
    # axes[2] gwc polny
    for i in range(dgrau1.shape[1]):
        axes[2].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau1[:,i,0] - dgrau1[:,i,1], 
                 linewidth=1, linestyle='-.', color=colorcycle_greens[i], label = r'Grau Only ('+ dgrau_SSP[i]+')')        
        axes[2].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau2[:,i,0] - dgrau2[:,i,1], 
                 linewidth=1, linestyle='--', color=colorcycle_greens[i], label = r'Grau Only ('+ dgrau_SSP[i]+')')   
        axes[2].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau3[:,i,0] - dgrau3[:,i,1], 
                 linewidth=1, linestyle='--', color=colorcycle_greens[i], label = r'Grau Only ('+ dgrau_SSP[i]+')')   
    axes[1].legend()
        
    axes[2].set_xlabel(r'Frequency [GHz]', color='k')
    axes[2].set_title( 'GWC only', fontsize='12', fontweight='bold')
    for iaxes in [0,1,2]:
        axes[iaxes].set_ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
        axes[iaxes].grid('true')
        axes[iaxes].axvline(x=10 ,ls='-',color='gray')
        axes[iaxes].axvline(x=19 ,ls='-',color='gray')
        axes[iaxes].axvline(x=22 ,ls='-',color='gray')
        axes[iaxes].axvline(x=37 ,ls='-',color='gray')
        axes[iaxes].axvline(x=85 ,ls='-',color='gray')
        axes[iaxes].axvline(x=166 ,ls='-',color='gray')
        axes[iaxes].set_xlim([5,freqLim])
    #plt.legend(ncol=3)
    #plt.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.35))

  
    return

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------    
def Plot_Individual_exps_paper_1GRAUspecies(dset1, dset2, dhailset, dhailset_mass, dcombined1, dcombined2, dgrau1, dgrau2, dgrau3, 
                                            dgrau_SSP, colorcycle, freqLim, GrauspeciesNr):
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
    colorcycle_greens = ['darkgreen', 'teal', 'darkslategray']
    f_grid = dset1['D']['f_grid'][0][0][0][0][0][0]/1e9

    plt.matplotlib.rc('font', family='serif', size = 12)
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, figsize=[9,12])

    #- axes[0] rwc only
    axes[0].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dset1['arts_tb'][0,:]-dset1['arts_cl'][0,:],linewidth=1,color='darkblue', label = 'Heavy Rain (HR) Only')
    axes[0].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dset2['arts_tb'][0,:]-dset2['arts_cl'][0,:],linewidth=1,color='cyan', label = 'Light Rain (LR) Only')
    axes[0].set_title('RWC only', fontsize='12', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim([-105,5])

    # axes[1] hwc only 
    for i in range(dhailset.shape[1]):
        axes[1].plot(dset2['D']['f_grid'][0][0][0][0][0][0]/1e9, dhailset[:,i,0] - dhailset[:,i,1], 
                 linewidth=1, color=colorcycle[i], label = r'HWP: '+str(dhailset_mass[i])+' kg/m$^2$')
    axes[1].set_title('HWC only', fontsize='12', fontweight='bold')
    axes[1].set_ylim([-105,5])

    # axes[2] gwc polny
    for i in [GrauspeciesNr]: #range(dgrau1.shape[1]):
        axes[2].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau1[:,i,0] - dgrau1[:,i,1], 
                 linewidth=1, linestyle='-', color=colorcycle_greens[0], label = r'ifac=1') #r'Grau Only ('+ dgrau_SSP[i]+')')        
        axes[2].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau2[:,i,0] - dgrau2[:,i,1], 
                 linewidth=1, linestyle='-', color=colorcycle_greens[1], label = r'ifac=2') #r'Grau Only ('+ dgrau_SSP[i]+')')   
        axes[2].plot(dset1['D']['f_grid'][0][0][0][0][0][0]/1e9, dgrau3[:,i,0] - dgrau3[:,i,1], 
                 linewidth=1, linestyle='-', color=colorcycle_greens[2], label = r'ifac=5') #r'Grau Only ('+ dgrau_SSP[i]+')')   
    axes[1].legend()

    axes[2].set_xlabel(r'Frequency [GHz]', color='k')
    axes[2].set_title( 'GWC only ('+dgrau_SSP[i]+')', fontsize='12', fontweight='bold')
    axes[2].set_ylim([-105,5])
    for iaxes in [0,1,2]:
        axes[iaxes].set_ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
        axes[iaxes].grid('true')
        axes[iaxes].axvline(x=10 ,ls='-',color='gray')
        axes[iaxes].axvline(x=19 ,ls='-',color='gray')
        axes[iaxes].axvline(x=22 ,ls='-',color='gray')
        axes[iaxes].axvline(x=37 ,ls='-',color='gray')
        axes[iaxes].axvline(x=85 ,ls='-',color='gray')
        axes[iaxes].axvline(x=166 ,ls='-',color='gray')
        axes[iaxes].set_xlim([5,freqLim])
    axes[2].legend(loc='lower right')
    #plt.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.35))



    return   
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------    
def Plot_Combined_exps_paper_1GRAUspecies(f_grid, dcombined1, dcombined2,  dcombined3, dhailset_mass, 
                                            dgrau_SSP, colorcycle, freqLim, GrauspeciesNr):
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
    colorcycle_greens = ['darkgreen', 'teal', 'darkslategray']

    plt.matplotlib.rc('font', family='serif', size = 12)
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, figsize=[9,12])

    #- axes[0] rwc LR + HAIL
    #--- axes 2 para combined experiments. 
    for i in range(dcombined1.shape[1]):
        axes[0].plot(f_grid/1e9, dcombined1[:,i,0] - dcombined1[:,i,1], 
                 linewidth=1, linestyle='-', color=colorcycle[i], label = r'HWP: '+str(dhailset_mass[i])+' kg/m$^2$')   
    axes[0].set_title('RWC-LR + HWC', fontsize='12', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim([-105,5])

    # axes[1] rwc HR + HAIL
    for i in range(dcombined2.shape[1]):
        axes[1].plot(f_grid/1e9, dcombined2[:,i,0] - dcombined2[:,i,1], 
                 linewidth=1, linestyle='--', color=colorcycle[i], label = r'HWP: '+str(dhailset_mass[i])+' kg/m$^2$')   
    axes[1].set_title('RWC-HR + HWC', fontsize='12', fontweight='bold')
    axes[1].set_ylim([-105,5])
    axes[1].legend(loc='lower right')

    # axes[1] rwc HR + HAIL + GWC    
    for i in range(dcombined3.shape[1]):

            
    for iaxes in [0,1,2]:
        axes[iaxes].set_ylabel(r'$\Delta$(Cloudy-Clear) [K]', color='k')
        axes[iaxes].grid('true')
        axes[iaxes].axvline(x=10 ,ls='-',color='gray')
        axes[iaxes].axvline(x=19 ,ls='-',color='gray')
        axes[iaxes].axvline(x=22 ,ls='-',color='gray')
        axes[iaxes].axvline(x=37 ,ls='-',color='gray')
        axes[iaxes].axvline(x=85 ,ls='-',color='gray')
        axes[iaxes].axvline(x=166 ,ls='-',color='gray')
        axes[iaxes].set_xlim([5,freqLim])

        


    return   
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
main_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/RT_simulations/Output/'


#------------------------------------------------------------------------------------------
# RAIN ONLY
#------------------------------------------------------------------------------------------
# HEAVY RAIN ONLY
exp_name  = 'BulkSIMS_RainOnly_HR'
f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
arts_exp_RainOnly_HR = FullData(f_arts)

# LIGHT RAIN ONLY
exp_name  = 'BulkSIMS_RainOnly_LR' 
f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
arts_exp_RainOnly_LR = FullData(f_arts)

# define some variables that I will use throughout
z_field = arts_exp_RainOnly_HR['D']['z_field'][0][0][0][0][0]
f_grid  = arts_exp_RainOnly_LR['D']['f_grid'][0][0][0][0][0][0]

#------------------------------------------------------------------------------------------
# HAIL ONLY
#------------------------------------------------------------------------------------------
arts_exp_HailOnly = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2 )); arts_exp_HailOnly[:] = np.nan
arts_exp_mass     = np.zeros(( len(f_grid), len(z_field) )); arts_exp_mass[:] = np.nan

for item,i in enumerate([2, 4, 6, 10]):
    exp_name  = 'BulkSIMS_HailOnly_HWC'+str(i) 
    f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
    arts_exp = FullData(f_arts)
    arts_exp_HailOnly[:,item,0] = arts_exp['arts_tb'][0,:]
    arts_exp_HailOnly[:,item,1] = arts_exp['arts_cl'][0,:]
    arts_exp_mass[item,:]       = arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0]
    
#------------------------------------------------------------------------------------------
# RWC+HAIL
#------------------------------------------------------------------------------------------
arts_exp_HRWCLR = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2 )); arts_exp_HRWCLR[:] = np.nan
arts_exp_HRWCHR = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2 )); arts_exp_HRWCHR[:] = np.nan
for item,i in enumerate([2, 4, 6, 10]):
    exp_name  = 'BulkSIMS_RWCLR_HWC'+str(i) 
    f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
    arts_exp = FullData(f_arts)
    arts_exp_HRWCLR[:,item,0] = arts_exp['arts_tb'][0,:]
    arts_exp_HRWCLR[:,item,1] = arts_exp['arts_cl'][0,:]    
    exp_name  = 'BulkSIMS_RWCHR_HWC'+str(i) 
    f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
    arts_exp = FullData(f_arts)
    arts_exp_HRWCHR[:,item,0] = arts_exp['arts_tb'][0,:]
    arts_exp_HRWCHR[:,item,1] = arts_exp['arts_cl'][0,:]       

#------------------------------------------------------------------------------------------
# GRAU ONLY
#------------------------------------------------------------------------------------------
arts_exp_GRAUPOnlyIFAC2 = np.zeros(( len(f_grid), 3, 2 )); arts_exp_GRAUPOnlyIFAC2[:] = np.nan
arts_exp_GRAUmassIFAC2  = np.zeros(( len(z_field) )); arts_exp_GRAUmassIFAC2[:] = np.nan

for item,i in enumerate(['8-ColumnAggregate','EvansSnowAggregate','LargeBlockAggregate']):
    exp_name  = 'BulkSIMS_GrauOnly_GWC'+i 
    f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
    arts_exp = FullData(f_arts)
    arts_exp_GRAUPOnlyIFAC2[:,item,0] = arts_exp['arts_tb'][0,:]
    arts_exp_GRAUPOnlyIFAC2[:,item,1] = arts_exp['arts_cl'][0,:]
    if item == 0: 
        arts_exp_GRAUmassIFAC2[:]    = arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0]
        
arts_exp_GRAUPOnlyIFAC5 = np.zeros(( len(f_grid), 3, 2 )); arts_exp_GRAUPOnlyIFAC5[:] = np.nan
arts_exp_GRAUmassIFAC5  = np.zeros(( len(z_field) )); arts_exp_GRAUmassIFAC5[:] = np.nan

for item,i in enumerate(['8-ColumnAggregate','EvansSnowAggregate','LargeBlockAggregate']):
    exp_name  = 'BulkSIMS_GrauOnly_ifac5_GWC'+i 
    f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
    arts_exp = FullData(f_arts)
    arts_exp_GRAUPOnlyIFAC5[:,item,0] = arts_exp['arts_tb'][0,:]
    arts_exp_GRAUPOnlyIFAC5[:,item,1] = arts_exp['arts_cl'][0,:]
    if item == 0: 
        arts_exp_GRAUmassIFAC5[:]    = arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0]

arts_exp_GRAUPOnlyIFAC1 = np.zeros(( len(f_grid), 3, 2 )); arts_exp_GRAUPOnlyIFAC1[:] = np.nan
arts_exp_GRAUmassIFAC1  = np.zeros(( len(z_field) )); arts_exp_GRAUmassIFAC1[:] = np.nan

for item,i in enumerate(['8-ColumnAggregate','EvansSnowAggregate','LargeBlockAggregate']):
    exp_name  = 'BulkSIMS_GrauOnly_ifac1_GWC'+i 
    f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
    arts_exp = FullData(f_arts)
    arts_exp_GRAUPOnlyIFAC1[:,item,0] = arts_exp['arts_tb'][0,:]
    arts_exp_GRAUPOnlyIFAC1[:,item,1] = arts_exp['arts_cl'][0,:]
    if item == 0: 
        arts_exp_GRAUmassIFAC1[:]    = arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0]
 
#------------------------------------------------------------------------------------------
# RWC+HAIL+GRAU (EVANS)
#------------------------------------------------------------------------------------------
arts_exp_LR_H_GRAU_EVANS = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2, 3 )); arts_exp_LR_H_GRAU_EVANS[:] = np.nan
arts_exp_HR_H_GRAU_EVANS = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2, 3 )); arts_exp_HR_H_GRAU_EVANS[:] = np.nan
for item,i in enumerate([2, 4, 6, 10]):  # BulkSIMS_RWC_HR_HWC_10GWC_ifac1EvansSnowAggregate
    for item2, ifacifac in [1]:  #,2,5]:
        exp_name  = 'BulkSIMS_RWC_HR_HWC_'+str(i)+'GWC_ifac'+str(ifacifac)+'EvansSnowAggregate') 
        f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
        arts_exp = FullData(f_arts)
        arts_exp_HR_H_GRAU_EVANS[:,item,0,item2] = arts_exp['arts_tb'][0,:]
        arts_exp_HR_H_GRAU_EVANS[:,item,1,item2] = arts_exp['arts_cl'][0,:]    
        exp_name  = 'BulkSIMS_RWC_LR_HWC_'+str(i)+'GWC_ifac'+str(ifacifac)+'EvansSnowAggregate') 
        f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
        arts_exp = FullData(f_arts)
        arts_exp_LR_H_GRAU_EVANS[:,item,0,item2] = arts_exp['arts_tb'][0,:]
        arts_exp_LR_H_GRAU_EVANS[:,item,1,item2] = arts_exp['arts_cl'][0,:]   

#------------------------------------------------------------------------------------------
# RWC+HAIL+GRAU (8-ColumnAggregate)
#------------------------------------------------------------------------------------------
arts_exp_LR_H_GRAU_8COL = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2, 3 )); arts_exp_LR_H_GRAU_8COL[:] = np.nan
arts_exp_HR_H_GRAU_8COL = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2, 3 )); arts_exp_HR_H_GRAU_8COL[:] = np.nan
for item,i in enumerate([2, 4, 6, 10]):  # BulkSIMS_RWC_HR_HWC_10GWC_ifac1EvansSnowAggregate
    for item2, ifacifac in [1]:  #,2,5]:
        exp_name  = 'BulkSIMS_RWC_HR_HWC_'+str(i)+'GWC_ifac'+str(ifacifac)+'8-ColumnAggregate') 
        f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
        arts_exp = FullData(f_arts)
        arts_exp_HR_H_GRAU_8COL[:,item,0,item2] = arts_exp['arts_tb'][0,:]
        arts_exp_HR_H_GRAU_8COL[:,item,1,item2] = arts_exp['arts_cl'][0,:]    
        exp_name  = 'BulkSIMS_RWC_LR_HWC_'+str(i)+'GWC_ifac'+str(ifacifac)+'8-ColumnAggregate') 
        f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
        arts_exp = FullData(f_arts)
        arts_exp_LR_H_GRAU_8COL[:,item,0,item2] = arts_exp['arts_tb'][0,:]
        arts_exp_LR_H_GRAU_8COL[:,item,1,item2] = arts_exp['arts_cl'][0,:]   
   
#------------------------------------------------------------------------------------------
# RWC+HAIL+GRAU (LargeBlock)
#------------------------------------------------------------------------------------------
arts_exp_LR_H_GRAU_LBLOCK = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2, 3 )); arts_exp_LR_H_GRAU_LBLOCK[:] = np.nan
arts_exp_HR_H_GRAU_LBLOCK = np.zeros(( len(f_grid), len([2, 4, 6, 10]), 2, 3 )); arts_exp_HR_H_GRAU_LBLOCK[:] = np.nan
for item,i in enumerate([2, 4, 6, 10]):  # BulkSIMS_RWC_HR_HWC_10GWC_ifac1EvansSnowAggregate
    for item2, ifacifac in [1]:  #,2,5]:
        exp_name  = 'BulkSIMS_RWC_HR_HWC_'+str(i)+'GWC_ifac'+str(ifacifac)+'LargeBlockAggregate') 
        f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
        arts_exp = FullData(f_arts)
        arts_exp_HR_H_GRAU_LBLOCK[:,item,0,item2] = arts_exp['arts_tb'][0,:]
        arts_exp_HR_H_GRAU_LBLOCK[:,item,1,item2] = arts_exp['arts_cl'][0,:]    
        exp_name  = 'BulkSIMS_RWC_LR_HWC_'+str(i)+'GWC_ifac'+str(ifacifac)+'LargeBlockAggregate') 
        f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
        arts_exp = FullData(f_arts)
        arts_exp_LR_H_GRAU_LBLOCK[:,item,0,item2] = arts_exp['arts_tb'][0,:]
        arts_exp_LR_H_GRAU_LBLOCK[:,item,1,item2] = arts_exp['arts_cl'][0,:]       
    
#------------------------------------------------------------------------------------------
# PLOTS ALL
#------------------------------------------------------------------------------------------
GHzfreqLim = 100
Plot_exp(arts_exp_RainOnly_HR, arts_exp_RainOnly_LR, ['HR','LR'])

Plot_TWOexps_wCombinedexps(arts_exp_RainOnly_HR, arts_exp_RainOnly_LR, arts_exp_HailOnly, [2, 4, 6, 10], arts_exp_HRWCHR, 
                           arts_exp_HRWCLR, ['darkred', 'magenta', 'salmon', 'red'], GHzfreqLim)

Plot_THREEexps_wCombinedexps(arts_exp_RainOnly_HR, arts_exp_RainOnly_LR, arts_exp_HailOnly, [2, 4, 6, 10], arts_exp_HRWCHR, 
                           arts_exp_HRWCLR, arts_exp_GRAUPOnlyIFAC1, arts_exp_GRAUPOnlyIFAC2, arts_exp_GRAUPOnlyIFAC5, 
                             ['8-ColumnAggregate','EvansSnowAggregate','LargeBlockAggregate'], 
                             ['darkred', 'magenta', 'salmon', 'red'], GHzfreqLim)
Plot_Individual_exps_paper(arts_exp_RainOnly_HR, arts_exp_RainOnly_LR, arts_exp_HailOnly, [2, 4, 6, 10], arts_exp_HRWCHR, 
                           arts_exp_HRWCLR, arts_exp_GRAUPOnlyIFAC1, arts_exp_GRAUPOnlyIFAC2, arts_exp_GRAUPOnlyIFAC5, 
                             ['8-ColumnAggregate','EvansSnowAggregate','LargeBlockAggregate'], 
                             ['darkred', 'magenta', 'salmon', 'red'], GHzfreqLim)

#- Plot the individual experiments: RWC-ONL, HWC-ONL (w/ different HWP) and GWC-ONLY for different species (AND DIFFERENT IFAC)
Plot_Individual_exps_paper_1GRAUspecies(arts_exp_RainOnly_HR, arts_exp_RainOnly_LR, arts_exp_HailOnly, [2, 4, 6, 10], arts_exp_HRWCHR, 
                           arts_exp_HRWCLR, arts_exp_GRAUPOnlyIFAC1, arts_exp_GRAUPOnlyIFAC2, arts_exp_GRAUPOnlyIFAC5, 
                             ['8-ColumnAggregate','EvansSnowAggregate','LargeBlockAggregate'], 
                             ['darkred', 'magenta', 'salmon', 'red'], 90, 2)
#- Plot combined experiments: RWC-LR + HWC, and RWC-HR + HWC 
Plot_Combined_exps_paper_1GRAUspecies(f_grid, arts_exp_HRWCLR,arts_exp_HRWCHR,  [2, 4, 6, 10], 
                                           [0,0,1], ['darkred', 'magenta', 'salmon', 'red'], 90, 3)

    
#------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#- PLOT ALL IWCs
colorsin =  ['darkred', 'magenta', 'salmon', 'red']
fig = plt.figure(figsize=(9,6))    
for item,i in enumerate([2, 4, 6, 10]):
    exp_name  = 'BulkSIMS_RWCLR_HWC'+str(i) 
    f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
    arts_exp = FullData(f_arts)
    if item == 0:
        plt.plot( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][0]*1000 , z_field/1e3, linewidth=1, linestyle='--', color='blue')
    plt.plot( arts_exp['D']['particle_bulkprop_field'][0][0][0][0][0][1][0:5]*1000 , z_field[0:5]/1e3, linewidth=2, linestyle='-', color=colorsin[item])
    
    exp_name  = 'BulkSIMS_RWCHR_HWC'+str(i) 
    f_arts    = main_dir + exp_name + '/' + 'GMI_Fascod_'+ exp_name + '.mat'
    arts_exp1 = FullData(f_arts)
    if item == 0:
        plt.plot( arts_exp1['D']['particle_bulkprop_field'][0][0][0][0][0][0]*1000 , z_field/1e3, linewidth=1, linestyle='-', color='darkblue')
    plt.plot( arts_exp1['D']['particle_bulkprop_field'][0][0][0][0][0][1][0:5]*1000 , z_field[0:5]/1e3, linewidth=2, linestyle='-', color=colorsin[item])

plt.plot( arts_exp_GRAUmassIFAC1[50:80]*1000 , z_field[50:80]/1e3, linewidth=2, linestyle='-.', color='darkgreen')    
plt.plot( arts_exp_GRAUmassIFAC2[50:80]*1000 , z_field[50:80]/1e3, linewidth=2, linestyle='-', color='darkgreen')    
plt.plot( arts_exp_GRAUmassIFAC5[50:80]*1000 , z_field[50:80]/1e3, linewidth=2, linestyle='--', color='darkgreen')    

plt.title( 'Cloud Scenarios', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'Mass Content [g/m3]', color='k')
plt.legend()
plt.ylim([0,15])



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

# NOTICE THAT EXPS. 1-6 HAVE TOTAL MASS OF 0.2
# simple cloudbox 5e-2 (5cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPa0.005/' + 'GMI_Fascod_HailOnly_ssd_EXPa0.005.mat'
arts_exp1 = FullData(f_arts)

# simple cloudbox 1e-2 (1cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPa0.01/' + 'GMI_Fascod_HailOnly_ssd_EXPa0.01.mat'
arts_exp2 = FullData(f_arts)

# simple cloudbox 5e-2 (5cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPa0.05/' + 'GMI_Fascod_HailOnly_ssd_EXPa0.05.mat'
arts_exp3 = FullData(f_arts)

# Exponetial cloudbox (0.05 cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPb0.005/' + 'GMI_Fascod_HailOnly_ssd_EXPb0.005.mat'
arts_exp4 = FullData(f_arts)

# Exponetial cloubox 1e-2 (1cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPb0.01/' + 'GMI_Fascod_HailOnly_ssd_EXPb0.01.mat'
arts_exp5 = FullData(f_arts)

# Exponetial cloudbox 5e-2 (5cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPb0.05/' + 'GMI_Fascod_HailOnly_ssd_EXPb0.05.mat'
arts_exp6 = FullData(f_arts)

# Simple cloudbox EXP. PSD 0.2*5
f_arts    = main_dir + 'HailOnly_ssd_EXPc999/' + 'GMI_Fascod_HailOnly_ssd_EXPc999.mat'
arts_exp7 = FullData(f_arts)

# Simple cloudbox EXP. PSD 0.4*5 
f_arts    = main_dir + 'HailOnly_ssd_EXPd999/' + 'GMI_Fascod_HailOnly_ssd_EXPd999.mat'
arts_exp8 = FullData(f_arts)

# Simple cloudbox EXP. PSD 7.5*5 
f_arts    = main_dir + 'HailOnly_ssd_EXPe999/' + 'GMI_Fascod_HailOnly_ssd_EXPe999.mat'
arts_exp9 = FullData(f_arts)

# Simple cloudbox EXP. PSD 1.5*5 
f_arts    = main_dir + 'HailOnly_ssd_EXPf999/' + 'GMI_Fascod_HailOnly_ssd_EXPf999.mat'
arts_exp10 = FullData(f_arts)

# Simple cloudbox EXP. PSD 2*5 
f_arts    = main_dir + 'HailOnly_ssd_EXPg999/' + 'GMI_Fascod_HailOnly_ssd_EXPg999.mat'
arts_exp11 = FullData(f_arts)

# NOTICE THAT EXPS. 12-14 HAVE TOTAL MASS OF 10
# simple cloudbox 5e-2 (0.5cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPa_10gm20.005/' + 'GMI_Fascod_HailOnly_ssd_EXPa_10gm20.005.mat'
arts_exp12 = FullData(f_arts)

# simple cloudbox 5e-2 (1cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPa_10gm20.01/' + 'GMI_Fascod_HailOnly_ssd_EXPa_10gm20.01.mat'
arts_exp13 = FullData(f_arts)

# simple cloudbox 5e-2 (5cm)
f_arts    = main_dir + 'HailOnly_ssd_EXPa_10gm20.05/' + 'GMI_Fascod_HailOnly_ssd_EXPa_10gm20.05.mat'
arts_exp14 = FullData(f_arts)

# ----------------------------- PLOT ALL EXP PSD

Plot_exp_hail(arts_exp7, arts_exp8, arts_exp9, arts_exp10, arts_exp11, arts_exp1, arts_exp2, arts_exp3,arts_exp12, arts_exp13, arts_exp14,
              labels_in=['1 g/m$^2$', '2 g/m$^2$', '4 g/m$^2$', '7.5 g/m$^2$','10 g/m$^2$', 'Monodisperse (0.5 cm)', 
                         'Monodisperse (1 cm)', 'Monodisperse (5 cm)'])


# ----------------------------- HAIL WATER CONTENT  
# 0.2, 0.4, 0.8, 1.5, 2  ==> WP = 1 g/m2, 2 g/m2, 4 g/m2, 7.5 g/m2, 10 g/m2
cloud_hwc      = np.zeros(arts_exp1['D']['particle_bulkprop_field'][0][0][0][0][0][0].shape[0]);
cloud_hwc[:] = np.nan

plt.matplotlib.rc('font', family='serif', size = 12)
fig = plt.figure(figsize=(9,6))    
cloud_hwc[0:4] = 0.2                       
plt.plot( cloud_hwc[0:10], arts_exp1['D']['z_field'][0][0][0][0][0][0:10] / 1e3, 'darkblue', linewidth=1.5, label='1 g/m$^2$')
cloud_hwc[0:4] = 0.4                          
plt.plot( cloud_hwc[0:10], arts_exp1['D']['z_field'][0][0][0][0][0][0:10] / 1e3, 'darkred', linewidth=1.5, label='2 g/m$^2$')
cloud_hwc[0:4] = 0.8                          
plt.plot( cloud_hwc[0:10], arts_exp1['D']['z_field'][0][0][0][0][0][0:10] / 1e3, 'darkgreen', linewidth=1.5, label='4 g/m$^2$')
cloud_hwc[0:4] = 1.5                          
plt.plot( cloud_hwc[0:10], arts_exp1['D']['z_field'][0][0][0][0][0][0:10] / 1e3, 'red', linewidth=1.5, label='7.5 g/m$^2$')
cloud_hwc[0:4] = 2                          
plt.plot( cloud_hwc[0:10], arts_exp1['D']['z_field'][0][0][0][0][0][0:10] / 1e3, 'blue', linewidth=1.5, label='10 g/m$^2$')
plt.title( 'Cloud Scenarios (Hail-Only)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'mass content [g/m$^3$]', color='k')
plt.grid('true') 
plt.legend()
plt.ylim([0,3])







# ----------------------------- plot
Plot_exp_hail(arts_exp1, arts_exp2, arts_exp3, arts_exp4, arts_exp5, arts_exp6, 
              ['Simple cloudbox (mono 0.05cm)', 'Simple cloudbox (mono 1cm)','Simple cloudbox (mono 5cm)',
               'Exp cloudbox (mono 0.05cm)', 'Exp cloudbox (mono 1cm)','Exp. cloudbox (mono 5cm)'])

mass_1 = 500 * 4 * 3.1416 * (0.005**3) / 24;
mass_2 = 500 * 4 * 3.1416 * (1e-2 **3) / 24;
mass_3 = 500 * 4 * 3.1416 * (5e-2 **3) / 24;

plt.matplotlib.rc('font', family='serif', size = 12)
fig = plt.figure(figsize=(9,6))    
plt.plot( arts_exp1['D']['particle_bulkprop_field'][0][0][0][0][0][0]*mass_1*1000, arts_exp1['D']['z_field'][0][0][0][0][0] / 1e3, linewidth=1,color='k', label = 'Simple cloudbox')
plt.title( 'Cloud Scenarios (Hail-Only)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'mass content [kg/m3]', color='k')
plt.grid('true')
plt.legend()
plt.ylim([0,20])

















fig = plt.figure(figsize=(9,6))    
plt.plot( arts_exp1['D']['particle_bulkprop_field'][0][0][0][0][0][0], arts_exp1['D']['z_field'][0][0][0][0][0] / 1e3, linewidth=1,color='k', label = 'Simple cloudbox 1cm')
plt.plot( arts_exp3['D']['particle_bulkprop_field'][0][0][0][0][0][0], arts_exp3['D']['z_field'][0][0][0][0][0]/ 1e3, linewidth=1,color='gray', label = 'Exp. cloudbox 1cm')
plt.plot( arts_exp2['D']['particle_bulkprop_field'][0][0][0][0][0][0], arts_exp1['D']['z_field'][0][0][0][0][0] / 1e3, linewidth=1,color='darkblue', label = 'Simple cloudbox 5cm')
plt.plot( arts_exp4['D']['particle_bulkprop_field'][0][0][0][0][0][0], arts_exp3['D']['z_field'][0][0][0][0][0]/ 1e3, linewidth=1,color='cyan', label = 'Exp. cloudbox 5cm')
plt.title( 'Cloud Scenarios (Hail-Only)', fontsize='12', fontweight='bold')
plt.ylabel(r'Height [km]', color='k')
plt.xlabel(r'PND [1/m3]', color='k')
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

    
 

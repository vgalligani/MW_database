
from scipy.interpolate import griddata
from collections import defaultdict
from scipy import stats 
from os import listdir
from pyhdf import SD
import gc
from numpy import genfromtxt;
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_categoryPF_hi_altfilter(PF_all, select, vkey):
    
    
    import netCDF4 as nc
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/etopo1_bedrock.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['lat'][:]
    topo_lon = ds.variables['lon'][:]
    topo_dat = ds.variables['Band1'][:]/1e3
    lons_topo, lats_topo = np.meshgrid(topo_lon,topo_lat)
    
    var    = PF_all[vkey][select].copy()
    latlat = PF_all['LAT'][select].copy()
    lonlon = PF_all['LON'][select].copy()
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                        (lonlon,latlat), method='nearest')
    varfilt = var[ np.where(sat_alt < 2.4) ]      
    percentiles = np.percentile(varfilt, [99.99, 99.9, 99, 90])
    

    
    return varfilt, latlat[ np.where(sat_alt < 2.4) ], lonlon[ np.where(sat_alt < 2.4) ], percentiles

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_categoryPF_altfilter(PF_all, select, vkey):
    
    import netCDF4 as nc
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/etopo1_bedrock.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['lat'][:]
    topo_lon = ds.variables['lon'][:]
    topo_dat = ds.variables['Band1'][:]/1e3
    lons_topo, lats_topo = np.meshgrid(topo_lon,topo_lat) 

    var    = PF_all[vkey][select].copy()
    latlat = PF_all['LAT'][select].copy()
    lonlon = PF_all['LON'][select].copy()
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                        (lonlon,latlat), method='nearest')
    varfilt = var[ np.where(sat_alt < 2.4) ]      

    percentiles = np.percentile(varfilt, [10, 1, 0.1, 0.01])

    
    return varfilt, latlat[ np.where(sat_alt < 2.4) ] , lonlon[ np.where(sat_alt < 2.4) ] , percentiles


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def plot_PCT_percentiles_paper(dir, filename, Kurpf,  selectKurpf, realKurpf, realselectKurpf, PFtype):

    xlim1 = -70 
    xlim2 = -50
    ylim1 = -40 
    ylim2 = -20
  
    import seaborn as sns
    from scipy.interpolate import griddata

    # Get altitude
    import netCDF4 as nc
    #fn = '/home/victoria.galligani/Work/Tools/etopo1_bedrock.nc'
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/etopo1_bedrock.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['lat'][:]
    topo_lon = ds.variables['lon'][:]
    topo_dat = ds.variables['Band1'][:]/1e3
    lons_topo, lats_topo = np.meshgrid(topo_lon,topo_lat)

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12

    prov = genfromtxt("/home/victoria.galligani/Work/Tools/provincias.txt", delimiter='')
    samerica = genfromtxt("/home/victoria.galligani/Work/Tools/samerica.txt", delimiter='')

    # replace highest temperatures with gray
    cmap1 =  plt.cm.get_cmap('tab20c')
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)


    Stats = open(dir+filename+'_'+PFtype+'_info.txt', 'w')  

    #------------------------- Figure 
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True,figsize=[15,5])
    
    #------ MIN37PCT
    axes[0].plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    axes[0].plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    axes[0].set_title(PFtype+' MIN37PCT')
    MIN37PCT_cat, latlat, lonlon, percentiles = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN37PCT')
    counter = 0
    # here mask latlat and lonlon above 2.4 km altitude
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    for i in percentiles:
        LON  = lonlon[np.where( (MIN37PCT_cat < i) & (sat_alt < 2.4) )]        
        LAT = latlat[np.where(  (MIN37PCT_cat < i) & (sat_alt < 2.4) )]                  
        if counter < 1:
            axes[0].scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            axes[0].scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))        
        counter = counter+1
    axes[0].set_ylabel('Latitude')
    axes[0].set_xlim([xlim1,xlim2])
    axes[0].set_ylim([ylim1,ylim2])
    axes[0].set_xlabel('Longtiude')
    # 
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    #-colorbar
    cbar = fig.colorbar(img, ax=axes[0],ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal", label='Percentile (%)')
    labels = ['10', '1', '0.1', '0.01']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)
    print('MIN37PCT_cat percentiles:', percentiles, file=Stats)  
    

    #------ pixels
    axes[1].plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    axes[1].plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    axes[1].set_title(PFtype+' NPIXELS GMI')
    pixels_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'NPIXELS_GMI')
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in reversed(percentiles):
        LON  = lonlon[np.where( (pixels_cat > i) & (sat_alt < 1.7) )]        
        LAT = latlat[np.where(  (pixels_cat > i) & (sat_alt < 1.7) )]               
        if counter < 1:
            axes[1].scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            axes[1].scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))        
        counter = counter+1
    axes[1].set_xlim([xlim1,xlim2])
    axes[1].set_ylim([ylim1,ylim2])
    axes[1].set_xlabel('Longtiude')

    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    #-colorbar
    cbar = fig.colorbar(img,  ax=axes[1], ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal", label='Percentile (%)')
    labels = ['90', '99', '99.9', '99.99']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)
    print('NPIXELS_cat percentiles:', percentiles, file=Stats)  
    
  
    #------ max45ht
    axes[2].plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    axes[2].plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    axes[2].set_title('KuRPF MAXHT40')
    MAXHT40_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(realKurpf, realselectKurpf, 'MAXHT40')
    # here mask latlat and lonlon above 2.4 km altitude
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in reversed(percentiles):
        LON  = lonlon[np.where( (MAXHT40_cat > i) & (sat_alt < 2.4) )]        
        LAT = latlat[np.where(  (MAXHT40_cat > i) & (sat_alt < 2.4) )]     
        if counter < 1:
            axes[2].scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            axes[2].scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    axes[2].set_xlabel('Longtiude')

    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    cbar = fig.colorbar(img,  ax=axes[2], ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal", label='Percentile (%)')
    labels = ['90', '99', '99.9', '99.99']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)
    axes[2].set_xlim([xlim1,xlim2])
    axes[2].set_ylim([ylim1,ylim2])    
    print('MAXHT40 percentiles:', percentiles, file=Stats)  
    Stats.close()

    fig.savefig('/home/victoria.galligani/Dropbox/FigsPaper/FIG8.png', dpi=300,transparent=False)   


        
    return fig
    
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def merge_GPCTF_dicts_keys(Kurpf_path):

    Kurpf_data = defaultdict(list)

    files = listdir(Kurpf_path)
    for i in files: 
        print(i)
        Kurpf = read_KuRPF(Kurpf_path+i)
        #Kurpf_data['ORBIT'].append(Kurpf['ORBIT'])
        #Kurpf_data['YEAR'].append(Kurpf['YEAR'])
        #Kurpf_data['MONTH'].append(Kurpf['MONTH'])
        #Kurpf_data['DAY'].append(Kurpf['DAY'])
        #Kurpf_data['HOUR'].append(Kurpf['HOUR'])
        Kurpf_data['ACTRK'].append(Kurpf['ACTRK'])             # Cross track center location (#pixels)
        Kurpf_data['ALTRK'].append(Kurpf['ALTRK'])             # Along track center location (# pixels)   
        Kurpf_data['H10ATMIN37'].append(Kurpf['H10ATMIN37'])   # 10H GHz TB at min 37 PCT location (K)
        Kurpf_data['H19ATMIN37'].append(Kurpf['H19ATMIN37'])   # 1H GHz TB at min 37 PCT location (K)
        Kurpf_data['LANDOCEAN'].append(Kurpf['LANDOCEAN'])     
        Kurpf_data['LAT'].append(Kurpf['LAT'])                 # Geographical center latitude (degree)
        Kurpf_data['LON'].append(Kurpf['LON'])                 # Geographical center longitude (degree)
        Kurpf_data['MIN1833'].append(Kurpf['MIN1833'])
        Kurpf_data['MIN1838'].append(Kurpf['MIN1838'])
        Kurpf_data['MIN165V'].append(Kurpf['MIN165V'])     
        Kurpf_data['MIN85PCT'].append(Kurpf['MIN85PCT'])
        Kurpf_data['MIN37PCT'].append(Kurpf['MIN37PCT'])
        Kurpf_data['NPIXELS_GMI'].append(Kurpf['NPIXELS_GMI'])   # Number of GMI pixels (#)
        Kurpf_data['NRAINAREA_GMI'].append(Kurpf['NRAINAREA_GMI'])  
        Kurpf_data['NRAINPIXELS_GMI'].append(Kurpf['NRAINPIXELS_GMI']) # Number of GMI pixels with precip (#)
        Kurpf_data['VOLRAIN_GMI'].append(Kurpf['VOLRAIN_GMI'])
 
               
        gc.collect
        del Kurpf

    return Kurpf_data

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def read_KuRPF(ifile):
        
    # open the hdf file
    hdf  = SD.SD(ifile)
    # Make dictionary and read the HDF file
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    Kurpf_data = {}         
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        Kurpf_data[name] = data
    hdf.end()

    del data, ds, dsets, dsNames
    return Kurpf_data

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def merge_KURPF_dicts_keys(Kurpf_path):

    Kurpf_data = defaultdict(list)

    files = listdir(Kurpf_path)
    for i in files: 
        print(i)
        Kurpf = read_KuRPF(Kurpf_path+i)
        #Kurpf_data['ORBIT'].append(Kurpf['ORBIT'])
        #Kurpf_data['YEAR'].append(Kurpf['YEAR'])
        #Kurpf_data['MONTH'].append(Kurpf['MONTH'])
        #Kurpf_data['DAY'].append(Kurpf['DAY'])
        #Kurpf_data['HOUR'].append(Kurpf['HOUR'])
        Kurpf_data['LAT'].append(Kurpf['LAT'])
        Kurpf_data['LON'].append(Kurpf['LON'])
        Kurpf_data['R_LAT'].append(Kurpf['R_LAT'])
        Kurpf_data['R_LON'].append(Kurpf['R_LON'])
        Kurpf_data['R_MAJOR'].append(Kurpf['R_MAJOR'])
        Kurpf_data['R_MINOR'].append(Kurpf['R_MINOR'])
        Kurpf_data['R_SOLID'].append(Kurpf['R_SOLID'])        
        Kurpf_data['NPIXELS_GMI'].append(Kurpf['NPIXELS_GMI'])
        Kurpf_data['NRAINPIXELS_GMI'].append(Kurpf['NRAINPIXELS_GMI'])
        #Kurpf_data['NRAINAREA_GMI'].append(Kurpf['NRAINAREA_GMI'])
        #Kurpf_data['VOLRAIN_GMI'].append(Kurpf['VOLRAIN_GMI'])
        #Kurpf_data['NLT250'].append(Kurpf['NLT250'])
        #Kurpf_data['NLT225'].append(Kurpf['NLT225'])
        #Kurpf_data['NLT200'].append(Kurpf['NLT200'])
        #Kurpf_data['NLT175'].append(Kurpf['NLT175'])
        #Kurpf_data['N37LT250'].append(Kurpf['N37LT250'])
        #Kurpf_data['N37LT225'].append(Kurpf['N37LT225'])
        #Kurpf_data['N37LT200'].append(Kurpf['N37LT200'])
        Kurpf_data['MIN85PCT'].append(Kurpf['MIN85PCT'])
        #Kurpf_data['MIN85PCTLAT'].append(Kurpf['MIN85PCTLAT'])
        #Kurpf_data['MIN85PCTLON'].append(Kurpf['MIN85PCTLON'])
        Kurpf_data['MIN37PCT'].append(Kurpf['MIN37PCT'])
        #Kurpf_data['MIN37PCTLAT'].append(Kurpf['MIN37PCTLAT'])
        #Kurpf_data['MIN37PCTLON'].append(Kurpf['MIN37PCTLON'])
        Kurpf_data['MIN1833'].append(Kurpf['MIN1833'])
        Kurpf_data['MIN1838'].append(Kurpf['MIN1838'])
        Kurpf_data['MIN165V'].append(Kurpf['MIN165V'])
        #Kurpf_data['MIN165H'].append(Kurpf['MIN165H'])
        #Kurpf_data['V19ATMIN37'].append(Kurpf['V19ATMIN37'])
        #Kurpf_data['H19ATMIN37'].append(Kurpf['H19ATMIN37'])
        Kurpf_data['LANDOCEAN'].append(Kurpf['LANDOCEAN'])
        Kurpf_data['MAXHT20'].append(Kurpf['MAXHT20'])
        Kurpf_data['NPIXELS'].append(Kurpf['NPIXELS'])
        Kurpf_data['MAXNSZ'].append(Kurpf['MAXNSZ'])
        Kurpf_data['MAXHT40'].append(Kurpf['MAXHT40'])
        Kurpf_data['VOLRAIN_KU'].append(Kurpf['VOLRAIN_KU'])

               
        gc.collect
        del Kurpf

    return Kurpf_data


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
xlim_min = -69; # (actually used -70 in search)
xlim_max = -50; 
ylim_min = -40; 
ylim_max = -19; 

opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
        'ylim_min': ylim_min, 'ylim_max': ylim_max}


Kurpf_path = '/home/victoria.galligani/Work/Studies/Hail_MW/GPM.PF/KURPF/'
Kurpf_data = merge_KURPF_dicts_keys(Kurpf_path)
# So far this generates e.g. Kurpf_data['LON'][0:37]. To to join ... 
GPCTF = {}
for key in Kurpf_data.keys():
    GPCTF[key] =  np.concatenate(Kurpf_data[key][:])
del Kurpf_data

selectGPCTF = np.logical_and(np.logical_and(GPCTF['LON'] >= opts['xlim_min'], GPCTF['LON'] <= opts['xlim_max']), 
                np.logical_and(GPCTF['LAT'] >= opts['ylim_min'], GPCTF['LAT'] <= opts['ylim_max']))

Kurpf_path = '/home/victoria.galligani/Work/Studies/Hail_MW/GPM.PF/GPCTF/'
Kurpf_data = merge_GPCTF_dicts_keys(Kurpf_path)
# So far this generates e.g. Kurpf_data['LON'][0:37]. To to join ... 
GPCTFGPCTF = {}
for key in Kurpf_data.keys():
    GPCTFGPCTF[key] =  np.concatenate(Kurpf_data[key][:])
del Kurpf_data
selectGPCTFGPCTF = np.logical_and(np.logical_and(GPCTFGPCTF['LON'] >= opts['xlim_min'], GPCTFGPCTF['LON'] <= opts['xlim_max']), 
                np.logical_and(GPCTFGPCTF['LAT'] >= opts['ylim_min'], GPCTFGPCTF['LAT'] <= opts['ylim_max']))


filename = 'paper_GMI_parameters.png'
plot_PCT_percentiles_paper(dir_name, filename, GPCTFGPCTF, selectGPCTFGPCTF,  GPCTF, selectGPCTF, 'GPCTF')










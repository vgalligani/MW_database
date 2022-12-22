#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/12/2022

@author: victoria.galligani

funciones que necesito para correr stats over RPF_GPM
pf_20xxxx_level2.HDF en conjunto con main_clim_ku.py 
UPDATED PARA CONTEMPLAR ULTIMO ANALISIS PARA EL PAPER
"""


from scipy.interpolate import griddata
from collections import defaultdict
from scipy import stats 


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def merge_KuRPF_dicts_all(Kurpf_path):

    Kurpf_data = defaultdict(list)

    files = listdir(Kurpf_path)
    for i in files: 
        print(i)
        print(psutil.virtual_memory().percent)
        print('========================================')
        Kurpf = read_KuRPF(Kurpf_path+i)
        for key, value in Kurpf.items():
            Kurpf_data[key].append(value)
            gc.collect()
        del Kurpf
        gc.collect()
    return Kurpf_data

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
def plot_PCT_percentiles_GMI(dir, filename, Kurpf, selectKurpf, PFtype):

    import seaborn as sns
    from scipy.interpolate import griddata

    # Get altitude
    import netCDF4 as nc
    #fn = '/home/victoria.galligani/Work/Tools/etopo1_bedrock.nc'
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/ETOPO1_Bed_c_gmt4.grd_0.1deg_SA.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['x'][:]
    topo_lon = ds.variables['y'][:]
    topo_dat = ds.variables['z'][:]/1e3
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
    fig = plt.figure(figsize=(6,5))     
    gs1 = gridspec.GridSpec(1, 1)
    #------ MIN37PCT
    ax1 = plt.subplot(gs1[0,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title(PFtype+' MIN37PCT intensity category')
    MIN37PCT_cat, latlat, lonlon, percentiles = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN37PCT')
    # here mask latlat and lonlon above 2.4 km altitude
    #sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
    #                   (lonlon,latlat), method='nearest')
    counter = 0
    for i in percentiles:
        LON = lonlon[( np.where( (MIN37PCT_cat < i) ))]         
        LAT = latlat[( np.where( (MIN37PCT_cat < i) ))]                
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    p2 = ax1.get_position().get_points().flatten()
    # 
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    #-colorbar
    ax_cbar = fig.add_axes([p2[0]-0.08, 0.01, p2[2], 0.02])
    cbar = fig.colorbar(img, cax=ax_cbar, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['10', '1', '0.1', '0.01']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)


    #fig.savefig(dir+filename+'only37.png', dpi=300,transparent=False)        

    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(2, 2)
    #------ MIN37PCT
    ax1 = plt.subplot(gs1[0,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title(PFtype+' MIN37PCT intensity category')
    MIN37PCT_cat, latlat, lonlon, percentiles = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN37PCT')
    # here mask latlat and lonlon above 2.4 km altitude
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in percentiles:
        LON = lonlon[( np.where( (MIN37PCT_cat < i) & (sat_alt < 2.4) ))]         
        LAT = latlat[( np.where( (MIN37PCT_cat < i) & (sat_alt < 2.4) ))]   
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)

    print('MIN37PCT_cat percentiles:', percentiles, file=Stats)

    #------ MIN85PCT
    ax1 = plt.subplot(gs1[0,1])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title(PFtype+' MIN85PCT intensity category')
    MIN85PCT_cat, latlat, lonlon, percentiles = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN85PCT')
    # here mask latlat and lonlon above 2.4 km altitude
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in percentiles:
        LON = lonlon[( np.where( (MIN85PCT_cat < i) & (sat_alt < 2.4) ))]         
        LAT = latlat[( np.where( (MIN85PCT_cat < i) & (sat_alt < 2.4) ))]   
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)

    print('MIN85PCT_cat percentiles:', percentiles, file=Stats)


    #------ MIN165V
    ax1 = plt.subplot(gs1[1,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title(PFtype+' MIN165V intensity category')
    MIN165V_cat, latlat, lonlon, percentiles = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN165V')
    # here mask latlat and lonlon above 2.4 km altitude
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in percentiles:
        LON = lonlon[( np.where( (MIN165V_cat < i) & (sat_alt < 2.4) ))]         
        LAT = latlat[( np.where( (MIN165V_cat < i) & (sat_alt < 2.4) ))]  
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    p1 = ax1.get_position().get_points().flatten()
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)

    print('MIN165V_cat percentiles:', percentiles, file=Stats)


    #------ MIN165V
    ax1 = plt.subplot(gs1[1,1])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title(PFtype+' MIN1838 intensity category')
    MIN1838_cat, latlat, lonlon, percentiles = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN1838')
    # here mask latlat and lonlon above 2.4 km altitude
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in percentiles:
        LON = lonlon[( np.where( (MIN1838_cat < i) & (sat_alt < 2.4) ))]         
        LAT = latlat[( np.where( (MIN1838_cat < i) & (sat_alt < 2.4) ))]  
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))
        counter = counter+1
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    p2 = ax1.get_position().get_points().flatten()
    # 
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)


    print('MIN1838_cat percentiles:', percentiles, file=Stats)

    #-colorbar
    ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.02])
    cbar = fig.colorbar(img, cax=ax_cbar, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['10', '1', '0.1', '0.01']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)


    #fig.savefig(dir+filename+'.png', dpi=300,transparent=False)        
    #plt.close()
    Stats.close()

    return fig
    
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_categoryPF_altfilter(PF_all, select, vkey):

    import netCDF4 as nc
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/ETOPO1_Bed_c_gmt4.grd_0.1deg_SA.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['x'][:]
    topo_lon = ds.variables['y'][:]
    topo_dat = ds.variables['z'][:]/1e3
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
def plot_PCT_percentiles_Ku(dir, filename, Kurpf, selectKurpf):

    import netCDF4 as nc
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/ETOPO1_Bed_c_gmt4.grd_0.1deg_SA.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['x'][:]
    topo_lon = ds.variables['y'][:]   
    topo_dat = ds.variables['z'][:]/1e3
    lons_topo, lats_topo = np.meshgrid(topo_lon,topo_lat)

    import seaborn as sns

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12

    prov = genfromtxt("/home/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/home/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    # replace highest temperatures with gray
    cmap1 =  plt.cm.get_cmap('tab20c')
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)



    #------------------------- Figure 
    fig = plt.figure(figsize=(6,5))     
    gs1 = gridspec.GridSpec(1, 1)
    #------ MAXHT40
    ax1 = plt.subplot(gs1[0,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('KuRPF MAXHT40T intensity category')
    MAXHT40_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'MAXHT40')
    # here mask latlat and lonlon above 2.4 km altitude
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in reversed(percentiles):
        LON  = lonlon[np.where( (MAXHT40_cat > i) & (sat_alt < 2.4) )]        
        LAT = latlat[np.where(  (MAXHT40_cat > i) & (sat_alt < 2.4) )]     
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    p2 = ax1.get_position().get_points().flatten()
    # 
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    #-colorbar
    ax_cbar = fig.add_axes([p2[0]-0.08, 0.01, p2[2], 0.02])
    cbar = fig.colorbar(img, cax=ax_cbar, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['90', '99', '99.9', '99.99']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)
    plt.close()
    fig.savefig(dir+filename+'onlymaxht40.png', dpi=300,transparent=False)        


    Stats = open(dir+filename+'_info.txt', 'w')
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(2, 2)
    #------ MAXHT20
    ax1 = plt.subplot(gs1[0,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('KuRPF MAXHT20 intensity category')
    MAXHT20_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'MAXHT20')
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in reversed(percentiles):
        LON  = lonlon[np.where ((MAXHT20_cat > i) & (sat_alt < 2.4) )]        
        LAT = latlat[np.where( (MAXHT20_cat > i) & (sat_alt < 2.4) )]     
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    print('MAXHT20_cat percentiles:', percentiles, file=Stats)
    
    #------ MAXHT30
    ax1 = plt.subplot(gs1[0,1])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('KuRPF MAXHT40 intensity category')
    MAXHT40_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'MAXHT40')
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in reversed(percentiles):
        LON  = lonlon[np.where( (MAXHT40_cat > i)  & (sat_alt < 2.4) )]     
        LAT = latlat[np.where( (MAXHT40_cat > i)  & (sat_alt < 2.4) )]     
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)

    print('MAXHT40_cat percentiles:', percentiles, file=Stats)
    
    #------ VOLRAIN_KU
    ax1 = plt.subplot(gs1[1,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('KuRPF VOLRAIN_KU intensity category')
    VOLRAIN_KU_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'VOLRAIN_KU') 
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')    
    counter = 0
    for i in reversed(percentiles):
        LON  = lonlon[np.where((VOLRAIN_KU_cat > i) & (sat_alt < 2.4) )]       
        LAT = latlat[np.where((VOLRAIN_KU_cat > i) & (sat_alt < 2.4) )]      
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    p1 = ax1.get_position().get_points().flatten()
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    print('VOLRAIN_KU_cat percentiles:', percentiles, file=Stats)

    #------ MAXNSZ
    ax1 = plt.subplot(gs1[1,1])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('KuRPF MAXNSZ intensity category')
    MAXNSZ_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'MAXNSZ')
    #MAXNSZ_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXNSZ') 
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                       (lonlon,latlat), method='nearest')
    counter = 0
    for i in reversed(percentiles):
        LON  = lonlon[np.where((MAXNSZ_cat > i) & (sat_alt < 2.4) )]       
        LAT = latlat[np.where((MAXNSZ_cat > i) & (sat_alt < 2.4) )]     
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))
        counter = counter+1
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-70,-45])
    ax1.set_ylim([-45,-15])
    p2 = ax1.get_position().get_points().flatten()
    # 
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)

    print('MAXNSZ_cat percentiles:', percentiles, file=Stats)

    
    #-colorbar
    ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.02])
    cbar = fig.colorbar(img, cax=ax_cbar, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['90', '99', '99.9', '99.99']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)
    
    #fig.savefig(dir+filename, dpi=300,transparent=False)        
   
    Stats.close()

    return fig 

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_categoryPF_hi_altfilter(PF_all, select, vkey):
    
    import netCDF4 as nc
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/ETOPO1_Bed_c_gmt4.grd_0.1deg_SA.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['x'][:]
    topo_lon = ds.variables['y'][:]   
    topo_dat = ds.variables['z'][:]/1e3
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
def plot_MIN1838_distrib(dir, filename, Kurpf, selectKurpf, PFtype):

    # Get altitude
    import netCDF4 as nc
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/ETOPO1_Bed_c_gmt4.grd_0.1deg_SA.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['x'][:]
    topo_lon = ds.variables['y'][:]   
    topo_dat = ds.variables['z'][:]/1e3
    lons_topo, lats_topo = np.meshgrid(topo_lon,topo_lat)
    
    import seaborn as sns

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
    
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(1, 1)
    #------ MIN37PCT
    ax1 = plt.subplot(gs1[0,0])
    plt.title(PFtype+' MIN1838 intensity distribution')
    MIN37PCT_cat, _, _, _  = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN37PCT')
    MIN85PCT_cat, _, _, _ = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN85PCT')
    MIN1838_cat, latlat, lonlon, percentiles = get_categoryPF_altfilter(Kurpf, selectKurpf, 'MIN1838')
    # here mask latlat and lonlon above 2.4 km altitude  
    counter = 0
    for i in percentiles:
        x_min37  = MIN37PCT_cat[( np.where( (MIN1838_cat < i) ))]          
        y_min85  = MIN85PCT_cat[( np.where( (MIN1838_cat < i)  ))]          
        if counter < 1:
            plt.scatter(x_min37, y_min85, s=15, marker='o', c = cmap_f(counter))
        elif counter < 3:
            plt.scatter(x_min37, y_min85, s=30, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(x_min37, y_min85, s=50, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.xlabel(PFtype+' MIN37PCT (K)')
    plt.ylabel(PFtype+' MIN85PCT (K)')
    plt.scatter(np.nan, np.nan, s=15, marker='o', c = cmap_f(0), label='class < 10%')        
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(1), label='class < 1%')             
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(2), label='class < 0.1%')        
    plt.scatter(np.nan, np.nan, s=50, marker='o', c = cmap_f(3), label='class < 0.11%')        
    plt.legend()    
    plt.grid()  
    #fig.savefig(dir+filename, dpi=300,transparent=False)        
    #plt.close()

    return



#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def plot_MAXHT40_distrib(dir, filename, Kurpf, MWRPF, selectKurpf, selectMWRPF, PFtype):

    import seaborn as sns

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
    
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(1, 1)
    #------ MAXHT40
    ax1 = plt.subplot(gs1[0,0])
    plt.title('PF MAXHT40 intensity distribution')
    MIN37PCT_cat, _, _, _ = get_categoryPF_altfilter(MWRPF, selectMWRPF, 'MIN37PCT')
    MAXNSZ_cat, _, _, _ = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'MAXNSZ')
    MAXHT40_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'MAXHT40')
    counter = 0
    for i in reversed(percentiles):
        x_min37   = MIN37PCT_cat[np.where(MAXHT40_cat > i)]   
        y_MAXNSZ  = MAXNSZ_cat[np.where(MAXHT40_cat > i)]
        if counter < 1:
            plt.scatter(x_min37, y_MAXNSZ, s=15, marker='o', c = cmap_f(counter))
        elif counter < 3:
            plt.scatter(x_min37, y_MAXNSZ, s=30, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(x_min37, y_MAXNSZ, s=50, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.xlabel(PFtype+' MIN37PCT (K)')
    plt.ylabel('KuRPF MAXNSZ (dBZ)')
    plt.scatter(np.nan, np.nan, s=15, marker='o', c = cmap_f(0), label='class > 90%')        
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(1), label='class > 99%')             
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(2), label='class > 99.9%')        
    plt.scatter(np.nan, np.nan, s=50, marker='o', c = cmap_f(3), label='class > 99.99%')        
    plt.legend()    
    plt.grid()
    
    #fig.savefig(dir+filename, dpi=300,transparent=False)        
    
    
    return


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def plot_volrain_Ku_distrib(dir, filename, Kurpf, MWRPF, selectKurpf, selectMWRPF, PFtype1, PFtype_area):

    import seaborn as sns
    
    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12
    # Some matplotlib figure definitions

    prov = genfromtxt("/home/victoria.galligani/Work/Tools/provincias.txt", delimiter='')
    samerica = genfromtxt("/home/victoria.galligani/Work/Tools/samerica.txt", delimiter='')

    # replace highest temperatures with gray
    cmap1 =  plt.cm.get_cmap('tab20c')
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)
    
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(1, 1)
    #------ VOLRAIN_KU
    ax1 = plt.subplot(gs1[0,0])
    plt.title('KuRPF VOLRAIN_KU intensity distribution')
    MIN85PCT_cat, _, _, _ = get_categoryPF_altfilter(MWRPF, selectMWRPF, 'MIN85PCT')
    VOLRAIN_KU_cat, latlat, lonlon, percentiles = get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'VOLRAIN_KU')
    #precipitation area IS estimated by the number of pixels associated with each PF.
    if PFtype_area == 'KuRPF':
        NPIXELS_cat, _, _, _= get_categoryPF_hi_altfilter(Kurpf, selectKurpf, 'NPIXELS')
    elif PFtype_area == 'GPCTF':
        NPIXELS_cat, _, _, _ = get_categoryPF_hi_altfilter(MWRPF, selectMWRPF, 'NPIXELS_GMI')

    npixels = NPIXELS_cat.copy()
    npixels = npixels.astype(np.float32)
    area    = npixels*5.*5.
    
    counter = 0
    for i in reversed(percentiles):
        x_min85   = MIN85PCT_cat[np.where(VOLRAIN_KU_cat > i)]   
        y_area    = area[np.where(VOLRAIN_KU_cat > i)]
        if counter < 1:
            plt.scatter(x_min85, y_area, s=15, marker='o', c = cmap_f(counter))
        elif counter < 3:
            plt.scatter(x_min85, y_area, s=30, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(x_min85, y_area, s=50, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.xlabel(PFtype1+' MIN85PCT (K)')
    plt.ylabel(PFtype_area+r' area (km$^2$)')
    ax1.set_yscale('log')
    plt.scatter(np.nan, np.nan, s=15, marker='o', c = cmap_f(0), label='class > 90%')        
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(1), label='class > 99%')             
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(2), label='class > 99.9%')        
    plt.scatter(np.nan, np.nan, s=50, marker='o', c = cmap_f(3), label='class > 99.99%')        
    plt.legend()    
    plt.grid()
    
    #fig.savefig(dir+filename, dpi=300,transparent=False)        
    
    
    return


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_orbits_extreme(Kurpf, selectKurpf, vkey):
    
    var         = Kurpf[vkey][selectKurpf].copy()
    percentiles = np.percentile(var, [10, 1, 0.1, 0.01])
    
    latlat = Kurpf['LAT'][selectKurpf].copy()
    lonlon = Kurpf['LON'][selectKurpf].copy()
    
    orbit  = Kurpf['ORBIT'][selectKurpf].copy()
    month  = Kurpf['MONTH'][selectKurpf].copy()
    day    = Kurpf['DAY'][selectKurpf].copy()
    hour   = Kurpf['HOUR'][selectKurpf].copy()
    
    LON = lonlon[np.where(var < percentiles[3])]   
    LAT = latlat[np.where(var < percentiles[3])]

    ORB = orbit[np.where(var < percentiles[3])]
    MM  = month[np.where(var < percentiles[3])]
    DD  = day[np.where(var < percentiles[3])]
    HH  = hour[np.where(var < percentiles[3])]
    
    info = []
    for i in range(len(ORB)):
        info.append('orbit Nr: '+str(ORB[i])+' '+str(DD[i])+'/'+str(MM[i])+'/2018 H'+str(HH[i])+' UTC')    
        # # Plot map? 
        # fig = plt.figure(figsize=(12,12))     
        # gs1 = gridspec.GridSpec(1, 1)
        # ax1 = plt.subplot(gs1[0,0])
        # plt.scatter(LON, LAT, c=tb_s1_gmi[inside,7], s=10, cmap=PlottingGMITools.cmaps['turbo_r']) 
        # ax1.set_xlim([-70,-45])
        # ax1.set_ylim([-45,-15])
        # plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
        # plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
        # plt.title('Location of PF centers after domain-location filter')
        # plt.text(-55, -17, 'Nr. PFs:'+str(Kurpf['LAT'][selectKurpf].shape[0]))
    
    
    return(info)  

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_orbits_extreme_hi(Kurpf, selectKurpf, vkey):
    
    var         = Kurpf[vkey][selectKurpf].copy()
    percentiles = np.percentile(var, [90, 99, 99.9, 99.99] )
    
    latlat = Kurpf['LAT'][selectKurpf].copy()
    lonlon = Kurpf['LON'][selectKurpf].copy()
    
    orbit  = Kurpf['ORBIT'][selectKurpf].copy()
    month  = Kurpf['MONTH'][selectKurpf].copy()
    day    = Kurpf['DAY'][selectKurpf].copy()
    hour   = Kurpf['HOUR'][selectKurpf].copy()
    
    LON = lonlon[np.where(var > percentiles[3])]   
    LAT = latlat[np.where(var > percentiles[3])]

    ORB = orbit[np.where(var > percentiles[3])]
    MM  = month[np.where(var > percentiles[3])]
    DD  = day[np.where(var > percentiles[3])]
    HH  = hour[np.where(var > percentiles[3])]
    
    info = []
    for i in range(len(ORB)):
        info.append('orbit Nr: '+str(ORB[i])+' '+str(DD[i])+'/'+str(MM[i])+'/2018 H'+str(HH[i])+' UTC')
    
    return(info) 


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def get_hourly_normbins(PF, select, vkey):

    # e.g. PCT_i_binned[0] is between 00UTC - 01UTC for percentile i 
    # normalize by the total numer of elements. 

    hour_bins = np.linspace(0, 24, 25)
    
    PCT_cat, latlat, lonlon, percentiles, hour, surfFlag = get_categoryPF(PF, select, vkey)
    normelems = np.zeros((len(hour_bins)-1, 4)); normelems[:]=np.nan
    counter = 0

    # NEED TO REMOVE SEA! KEEP ONLY LAND!!
    # ['LANDOCEAN'] --->    0: over ocean, 1: over land
    
    for i in percentiles:
        HOUR    = hour[np.where( (PCT_cat < i) & (surfFlag == 1))]   
        PCT_i   = PCT_cat[np.where( (PCT_cat < i) & (surfFlag == 1))]  
        totalelems_i = len(HOUR)
        PCT_i_binned = [PCT_i[np.where((HOUR > low) & (HOUR <= high))] for low, high in zip(hour_bins[:-1], hour_bins[1:])]
        for ih in range(len(hour_bins)-1):
            normelems[ih, counter] = len(PCT_i_binned[ih].data)/totalelems_i
        counter = counter+1
        del HOUR, PCT_i, totalelems_i, PCT_i_binned  
        
    return normelems, percentiles


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_categoryPF(PF_all, select, vkey):
    
    var        = PF_all[vkey][select].copy()
    percentiles = np.percentile(var, [10, 1, 0.1, 0.01])
    
    latlat = PF_all['LAT'][select].copy()
    lonlon = PF_all['LON'][select].copy()
    hour   = PF_all['HOUR'][select].copy()
    surfFlag = PF_all['LANDOCEAN'][select].copy()
    
    return var, latlat, lonlon, percentiles, hour, surfFlag




#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_categoryPF_hi(PF_all, select, vkey):
    
    var        = PF_all[vkey][select].copy()
    percentiles = np.percentile(var, [99.99, 99.9, 99, 90])
    
    latlat = PF_all['LAT'][select].copy()
    lonlon = PF_all['LON'][select].copy()
    hour   = PF_all['HOUR'][select].copy()
    surfFlag = PF_all['LANDOCEAN'][select].copy()
    
    return var, latlat, lonlon, percentiles, hour, surfFlag


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def plot_regrid_map(lonbins, latbins, zi_37, zi_85, zi_max40ht, filename, main_title, LON, LAT, MIN37PCT):

    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12

    xbins, ybins = len(lonbins), len(latbins) #number of bins in each dimension 

    prov = genfromtxt("/home/victoria.galligani/Work/Tools/provincias.txt", delimiter='')
    samerica = genfromtxt("/home/victoria.galligani/Work/Tools/samerica.txt", delimiter='')


    fig = plt.figure(figsize=(12,12))  
    gs1 = gridspec.GridSpec(2, 2)   
    ax1 = plt.subplot(gs1[0,0])
    pc = ax1.pcolor(lonbins, latbins, zi_37, vmin=50, vmax=300)
    plt.colorbar(pc)        
    ax1.set_title('MIN37PCT < 1% percentile')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
      
    ax1 = plt.subplot(gs1[0,1])
    pc = ax1.pcolor(lonbins, latbins, zi_85, vmin=50, vmax=300)
    plt.colorbar(pc)    
    ax1.set_title('MIN85PCT < 1% percentile')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    
    ax1 = plt.subplot(gs1[1,0])
    pc = ax1.pcolor(lonbins, latbins, zi_max40ht, vmin=0, vmax=20)
    plt.colorbar(pc)    
    ax1.set_title('MAXHT40 > 99% percentile')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])   
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.suptitle(main_title,y=0.93)
    
    #fig.savefig(filename+'.png', dpi=300,transparent=False)        

    return


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def plot_spatial_distrib_altfilter(Kurpf, selectKurpf, filename, main_title):

    import netCDF4 as nc
    fn = '/home/victoria.galligani/Dropbox/Hail_MW/Tools/ETOPO1_Bed_c_gmt4.grd_0.1deg_SA.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['x'][:]
    topo_lon = ds.variables['y'][:]
    topo_dat = ds.variables['z'][:]/1e3
    lons_topo, lats_topo = np.meshgrid(topo_lon,topo_lat)

    lonbins = np.arange(-70, -40, 2) 
    latbins = np.arange(-50, -10, 2)
    xbins, ybins = len(lonbins), len(latbins) #number of bins in each dimension
    
    prov = genfromtxt("/home/victoria.galligani/Work/Tools/provincias.txt", delimiter='')
    samerica = genfromtxt("/home/victoria.galligani/Work/Tools/samerica.txt", delimiter='')

    fig = plt.figure(figsize=(6,5))  
    gs1 = gridspec.GridSpec(1, 1)   
    ax1 = plt.subplot(gs1[0,0])
    max40ht_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                        (lonlon,latlat), method='nearest')
    LON         = lonlon[np.where( (max40ht_cat > percentiles[1]) & (sat_alt < 2.4) )]       
    LAT         = latlat[np.where( (max40ht_cat > percentiles[1]) & (sat_alt < 2.4) )]    
    PCT_max40ht = max40ht_cat[np.where( (max40ht_cat > percentiles[1]) & (sat_alt < 2.4) )]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT_max40ht, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)        
    cbar = plt.colorbar(pc,label='Freq. distribution')    
    #plt.plot(LON,LAT, 'xr')
    ax1.set_title('MAXHT40 > 99% percentile ('+str(np.round(percentiles[1], 2))+' km)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])   
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #cbar.set_label('# of elements in 2x2 grid')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude') 
    #fig.savefig(filename+'MAXHT40only.png', dpi=300,transparent=False)        



    fig = plt.figure(figsize=(6,5))  
    gs1 = gridspec.GridSpec(1, 1)   
    ax1 = plt.subplot(gs1[0,0])
    MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                        (lonlon,latlat), method='nearest')
    LON         = lonlon[np.where( (MIN37PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]    
    LAT         = latlat[np.where( (MIN37PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]    
    PCT37       = MIN37PCT_cat[np.where( (MIN37PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]     
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT37, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T, vmax=20)
    plt.colorbar(pc, label='Freq. distribution')        
    ax1.set_title('MIN37PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);     
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.title(main_title)
    #fig.savefig(filename+'MIN37PCTonly.png', dpi=300,transparent=False)        

    fig = plt.figure(figsize=(12,12))  
    gs1 = gridspec.GridSpec(2, 2)   
    
    ax1 = plt.subplot(gs1[0,0])
    MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                        (lonlon,latlat), method='nearest')
    LON         = lonlon[np.where( (MIN37PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]       
    LAT         = latlat[np.where( (MIN37PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]    
    PCT37       = MIN37PCT_cat[np.where( (MIN37PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]       
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT37, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T, vmax=20)
    #plt.plot(LON,LAT, 'xr')
    plt.colorbar(pc)        
    ax1.set_title('MIN37PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
      
    ax1 = plt.subplot(gs1[0,1])
    MIN85PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                        (lonlon,latlat), method='nearest')
    LON         = lonlon[np.where( (MIN85PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]      
    LAT         = latlat[np.where( (MIN85PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]    
    PCT85       = MIN85PCT_cat[np.where( (MIN85PCT_cat < percentiles[1]) & (sat_alt < 2.4) )]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT85, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)    
    #plt.plot(LON,LAT, 'xr')
    plt.colorbar(pc)    
    ax1.set_title('MIN85PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    
    ax1 = plt.subplot(gs1[1,0])
    max40ht_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    sat_alt = griddata((np.ravel(lons_topo),np.ravel(lats_topo)), np.ravel(topo_dat),
                        (lonlon,latlat), method='nearest')
    LON         = lonlon[np.where( (max40ht_cat > percentiles[1]) & (sat_alt < 2.4) )]    
    LAT         = latlat[np.where( (max40ht_cat > percentiles[1]) & (sat_alt < 2.4) )]    
    PCT_max40ht = max40ht_cat[np.where( (max40ht_cat > percentiles[1]) & (sat_alt < 2.4) )]       
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT_max40ht, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)        
    cbar = plt.colorbar(pc)    
    #plt.plot(LON,LAT, 'xr')
    ax1.set_title('MAXHT40 > 99% percentile ('+str(np.round(percentiles[1], 2))+' km)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])   
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar.set_label('# of elements in 2x2 grid')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.suptitle(main_title, y=0.93)
    
    #fig.savefig(filename+'.png', dpi=300,transparent=False)        

    return

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def plot_spatial_distrib(Kurpf, selectKurpf, filename, main_title):

    lonbins = np.arange(-70, -40, 2) 
    latbins = np.arange(-50, -10, 2)
    xbins, ybins = len(lonbins), len(latbins) #number of bins in each dimension
    
    prov = genfromtxt("/home/victoria.galligani/Work/Tools/provincias.txt", delimiter='')
    samerica = genfromtxt("/home/victoria.galligani/Work/Tools/samerica.txt", delimiter='')


    fig = plt.figure(figsize=(6,5))  
    gs1 = gridspec.GridSpec(1, 1)   
    ax1 = plt.subplot(gs1[0,0])
    max40ht_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    LON         = lonlon[np.where(max40ht_cat > percentiles[1])]   
    LAT         = latlat[np.where(max40ht_cat > percentiles[1])]
    PCT_max40ht = max40ht_cat[np.where(max40ht_cat > percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT_max40ht, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)        
    cbar = plt.colorbar(pc,label='Freq. distribution')    
    #plt.plot(LON,LAT, 'xr')
    ax1.set_title('MAXHT40 > 99% percentile ('+str(np.round(percentiles[1], 2))+' km)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])   
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #cbar.set_label('# of elements in 2x2 grid')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude') 
    #fig.savefig(filename+'MAXHT40only.png', dpi=300,transparent=False)        



    fig = plt.figure(figsize=(6,5))  
    gs1 = gridspec.GridSpec(1, 1)   
    ax1 = plt.subplot(gs1[0,0])
    MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    LON         = lonlon[np.where(MIN37PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN37PCT_cat < percentiles[1])]
    PCT37       = MIN37PCT_cat[np.where(MIN37PCT_cat < percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT37, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T, vmax=20)
    plt.colorbar(pc, label='Freq. distribution')        
    ax1.set_title('MIN37PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);     
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.title(main_title)
    #fig.savefig(filename+'MIN37PCTonly.png', dpi=300,transparent=False)        

    fig = plt.figure(figsize=(12,12))  
    gs1 = gridspec.GridSpec(2, 2)   
    
    ax1 = plt.subplot(gs1[0,0])
    MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    LON         = lonlon[np.where(MIN37PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN37PCT_cat < percentiles[1])]
    PCT37       = MIN37PCT_cat[np.where(MIN37PCT_cat < percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT37, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T, vmax=20)
    #plt.plot(LON,LAT, 'xr')
    plt.colorbar(pc)        
    ax1.set_title('MIN37PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
      
    ax1 = plt.subplot(gs1[0,1])
    MIN85PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
    LON         = lonlon[np.where(MIN85PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN85PCT_cat < percentiles[1])]
    PCT85       = MIN85PCT_cat[np.where(MIN85PCT_cat < percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT85, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)    
    #plt.plot(LON,LAT, 'xr')
    plt.colorbar(pc)    
    ax1.set_title('MIN85PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    
    ax1 = plt.subplot(gs1[1,0])
    max40ht_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    LON         = lonlon[np.where(max40ht_cat > percentiles[1])]   
    LAT         = latlat[np.where(max40ht_cat > percentiles[1])]
    PCT_max40ht = max40ht_cat[np.where(max40ht_cat > percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT_max40ht, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)        
    cbar = plt.colorbar(pc)    
    #plt.plot(LON,LAT, 'xr')
    ax1.set_title('MAXHT40 > 99% percentile ('+str(np.round(percentiles[1], 2))+' km)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])   
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar.set_label('# of elements in 2x2 grid')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.suptitle(main_title, y=0.93)
    
    #fig.savefig(filename+'.png', dpi=300,transparent=False)        

    return


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------



from scipy.interpolate import griddata


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
    fn = '/home/victoria.galligani/Work/Tools/etopo1_bedrock.nc'
    ds = nc.Dataset(fn)
    topo_lat = ds.variables['lat'][:]
    topo_lon = ds.variables['lon'][:]   
    topo_dat = ds.variables['Band1'][:]/1e3
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
    
    fig.savefig(dir+filename, dpi=300,transparent=False)        
   
    Stats.close()

    return fig 

  
  
  



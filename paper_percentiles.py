
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
    axes[0].set_title(PFtype+' MIN37PCT intensity category')
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
    #img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    #img.set_visible(False)
    axes[0].set_xlabel('Longtiude')
    # 
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    #-colorbar
    cbar = fig.colorbar(img, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['10', '1', '0.1', '0.01']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)
    
    

    #------ pixels
    axes[1].plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    axes[1].plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    axes[1].set_title(PFtype+' NPIXELS GMI intensity category')
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
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    #-colorbar
    cbar = fig.colorbar(img, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['90', '99', '99.9', '99.99']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)
    
  
    #------ max45ht
    axes[2].plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    axes[2].plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    axes[2].set_title('KuRPF MAXHT40T intensity category')
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

    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    cbar = fig.colorbar(img, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['90', '99', '99.9', '99.99']
    loc = np.arange(0, 4 , 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)
    axes[2].set_xlim([xlim1,xlim2])
    axes[2].set_ylim([ylim1,ylim2])    
    
    return fig
    

xlim_min = -68; # (actually used -70 in search)
xlim_max = -50; 
ylim_min = -40; 
ylim_max = -19; 

opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
        'ylim_min': ylim_min, 'ylim_max': ylim_max}


Kurpf_path = '/home/victoria.galligani/Work/Studies/Hail_MW/GPM.PF/KURPF/'
Kurpf_data = merge_GPCTF_dicts_keys(Kurpf_path)
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










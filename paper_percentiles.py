
def plot_PCT_percentiles_paper(dir, filename, Kurpf, selectKurpf, PFtype):

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
    #img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    #img.set_visible(False)
    
  
    #------ max45ht
    axes[2].plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    axes[2].plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    axes[2].SET_title('PF MAXHT40 intensity distribution')

    
    
    axes[2].set_xlim([xlim1,xlim2])
    axes[2].set_ylim([ylim1,ylim2])    
    
    return fig
    




return

Kurpf_path = '/home/victoria.galligani/Work/Studies/Hail_MW/GPM.PF/GPCTF/'
Kurpf_data = merge_GPCTF_dicts_keys(Kurpf_path)
# So far this generates e.g. Kurpf_data['LON'][0:37]. To to join ... 
GPCTF = {}
for key in Kurpf_data.keys():
    GPCTF[key] =  np.concatenate(Kurpf_data[key][:])
del Kurpf_data

filename = 'MIN37PCT_GMI_parameters.png'
plot_PCT_percentiles_paper(dir_name, filename, GPCTF, selectGPCTF, 'GPCTF')

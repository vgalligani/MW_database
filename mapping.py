from matplotlib import cm;



#------------------------------------------------------------------------------  
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#------------------------------------------------------------------------------  
def plot_Zhppi_wGMIcontour(radar, lat_pf, lon_pf, general_title, fname, nlev, options, era5_file, icoi):
    
    ERA5_field = xr.load_dataset(era5_file, engine="cfgrib")
    elemj      = find_nearest(ERA5_field['latitude'], lat_pf)
    elemk      = find_nearest(ERA5_field['longitude'], lon_pf)
    tfield_ref = ERA5_field['t'][:,elemj,elemk] - 273 # convert to C
    geoph_ref  = (ERA5_field['z'][:,elemj,elemk])/9.80665
    # Covert to geop. height (https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
    Re         = 6371*1e3
    alt_ref    = (Re*geoph_ref)/(Re-geoph_ref)
    freezing_lev = np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3) 
	
    # read file
    f = h5py.File( fname, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    
    for j in range(lon_gmi.shape[1]):
      #tb_s1_gmi[np.where(lon_gmi[:,j] >=  options['xlim_max']+10),:] = np.nan
      #tb_s1_gmi[np.where(lon_gmi[:,j] <=  options['xlim_min']-10),:] = np.nan
      tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
      tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan   
      lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
      lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  
      lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
      lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  	
	
    PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 
    
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    #------ 
    # Test plot figure: 
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[14,12])
    #-- Zh: 
    if 'TH' in radar.fields.keys():  
        TH = radar.fields['TH']['data'][start_index:end_index]
    elif 'DBZH' in radar.fields.keys():
        TH = radar.fields['DBZH']['data'][start_index:end_index]
    elif 'reflectivity' in radar.fields.keys(): 
        TH = radar.fields['reflectivity']['data'][start_index:end_index]
    elif 'attenuation_corrected_reflectivity_h' in radar.fields.keys(): 
        TH = radar.fields['attenuation_corrected_reflectivity_h']['data'][start_index:end_index]
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes.pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes.grid(True)
    for iPF in range(len(lat_pf)): 
        axes.plot(lon_pf[iPF], lat_pf[iPF], marker='*', markersize=20, markerfacecolor="None",
            markeredgecolor='black', markeredgewidth=2, label='GMI(PF) center') 
    axes.legend(loc='upper left')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    contorno89 = plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    #plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200, 240], colors=(['k','k']), linewidths=1.5);
    for item in contorno89.collections:
        for i in item.get_paths():
            v = i.vertices
            x = v[:, 0]
            y = v[:, 1]            
    # Get vertices of these polygon type shapes
    for ii in range(len(icoi)): 
	X1 = []; Y1 = []; vertices = []
    	for ik in range(len(contorno89.collections[0].get_paths()[int(icoi[ii])].vertices)): 
		X1.append(contorno89.collections[0].get_paths()[icoi[ii]].vertices[ik][0])
        	Y1.append(contorno89.collections[0].get_paths()[icoi[ii]].vertices[ik][1])
        	vertices.append([contorno89.collections[0].get_paths()[icoi[ii]].vertices[ik][0], 
                                        contorno89.collections[0].get_paths()[icoi[ii]].vertices[ik][1]])
    	convexhull = ConvexHull(vertices)
    	array_points = np.array(vertices)
    	##--- Run hull_paths and intersec
    	hull_path   = Path( array_points[convexhull.vertices] )
    	datapts = np.column_stack((np.ravel(lon_gmi[1:,:]),np.ravel(lat_gmi[1:,:])))
	if ii==0:
		inds1 = hull_path.contains_points(datapts)
	if ii==1:
		inds2 = hull_path.contains_points(datapts)
	if ii==3:
		inds3 = hull_path.contains_points(datapts)

    plt.xlim([options['xlim_min'], options['xlim_max']])
    plt.ylim([options['ylim_min'], options['ylim_max']])

    plt.suptitle(general_title, fontsize=14)

    # data converted to spherical coordinates to the Cartesian gridding using an azimuthal-equidistant projection with 1-km 
    # vertical and horizontal resolution. The gridding is performed using a Barnes weighting function with a radius of influence 
    # that increases with range from the radar. The minimum value for the radius of influence is 500 m. [ i.e., the maximum 
    # distance that a data point can have to a grid point to have an impact on it. In the vertical the radius of influence 
    # depends on the range, as the beamwidth increases with the range, due to the beam broadening of about 1degree.
    # This is an established radius of influence for interpolation of radar data, for example, used within various analyses
    # with the French operational radar network (see, e.g., Bousquet and Tabary 2014; Beck et al. 2014).]
    grided = pyart.map.grid_from_radars(radar, grid_shape=(20, 470, 470), 
                                       grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                    (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                       roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')
    
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True,
                        figsize=[14,6])
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    axes[0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('original radar resolution')
    axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[1].pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:], 
                  grided.fields['TH']['data'][0,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes[1].set_title('1 km gridded BARNES2')
    CS = axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
    axes[1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    # Add labels:
    labels = ["200 K","240 K"] 
    for i in range(len(labels)):
	CS.collections[i].set_label(labels[i])
    axes[1].legend(loc='upper left', fontsize=12)


    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True, figsize=[14,12])
    counter = 0
    for i in range(3):
        for j in range(3):
            axes[i,j].pcolormesh(grided.point_longitude['data'][counter,:,:], grided.point_latitude['data'][counter,:,:], 
                  grided.fields['TH']['data'][counter,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
            counter=counter+1;
            axes[i,j].set_title('horizontal slice at altitude: '+str(grided.z['data'][counter]/1e3))
            axes[i,j].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
            axes[i,j].set_xlim([options['xlim_min'], options['xlim_max']])
            axes[i,j].set_ylim([options['ylim_min'], options['ylim_max']])
		
    # Same as above but plt lowest level y closest to freezing level! 
    print('ERA5 freezing level at: (km)'+str(freezing_lev))
    frezlev = find_nearest(grided.z['data']/1e3, freezing_lev) 
    print('Freezing level at level:'+str(frezlev)+'i.e., at'+str(grided.z['data'][frezlev]/1e3))

    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True,
                        figsize=[14,6])
    axes[0].pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:], 
                  grided.fields['TH']['data'][0,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes[0].set_title('Ground Level')
    axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[1].pcolormesh(grided.point_longitude['data'][frezlev,:,:], grided.point_latitude['data'][frezlev,:,:], 
                  grided.fields['TH']['data'][frezlev,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes[1].set_title('Freezing level ('+str(grided.z['data'][frezlev]/1e3)+' km)')
    axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
    axes[1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    # Add labeled contours of cois of interest! 
    axes[0].plot(np.ravel(lon_gmi[1:,:])[inds_1[0]], np.ravel(lat_gmi[1:,:])[inds_1[0]], '*', markersize=40)
    axes[1].plot(np.ravel(lon_gmi[1:,:])[inds_1[0]], np.ravel(lat_gmi[1:,:])[inds_1[0]], '*', markersize=40)
    if ii == 2:
	axes[0].plot(np.ravel(lon_gmi[1:,:])[inds_2[0]], np.ravel(lat_gmi[1:,:])[inds_2[0]], 'k*', markersize=40)
	axes[1].plot(np.ravel(lon_gmi[1:,:])[inds_2[0]], np.ravel(lat_gmi[1:,:])[inds_2[0]], 'k*', markersize=40)
    if ii == 3:
	axes[0].plot(np.ravel(lon_gmi[1:,:])[inds_2[0]], np.ravel(lat_gmi[1:,:])[inds_2[0]], '*', color='darkblue', markersize=40)
	axes[0].plot(np.ravel(lon_gmi[1:,:])[inds_3[0]], np.ravel(lat_gmi[1:,:])[inds_3[0]], '*', markersize=40)
	axes[1].plot(np.ravel(lon_gmi[1:,:])[inds_2[0]], np.ravel(lat_gmi[1:,:])[inds_2[0]], '*', markersize=40)
	axes[1].plot(np.ravel(lon_gmi[1:,:])[inds_3[0]], np.ravel(lat_gmi[1:,:])[inds_3[0]], '*', markersize=40)
	
    #axes[0].plot(np.nan, np.nan, '*', markersize=40)

    return

#------------------------------------------------------------------------------  
def plot_3D_Zhppi(radar, lat_pf, lon_pf, general_title, fname, nlev, options):
    
    nws_ref_colors_transparent =([ [ 1.0/255.0, 159.0/255.0, 244.0/255.0, 0.3],
                    [  3.0/255.0,   0.0/255.0, 244.0/255.0, 0.3],
                    [  2.0/255.0, 253.0/255.0,   2.0/255.0, 0.3],
                    [  1.0/255.0, 197.0/255.0,   1.0/255.0, 0.3],
                    [  0.0/255.0, 142.0/255.0,   0.0/255.0, 0.3],
                    [253.0/255.0, 248.0/255.0,   2.0/255.0, 0.3],
                    [229.0/255.0, 188.0/255.0,   0.0/255.0, 1],
                    [253.0/255.0, 149.0/255.0,   0.0/255.0, 1],
                    [253.0/255.0,   0.0/255.0,   0.0/255.0, 1],
                    #[212.0/255.0,   0.0/255.0,   0.0/255.0, 0.4],
                    [188.0/255.0,   0.0/255.0,   0.0/255.0, 1],
                    [248.0/255.0,   0.0/255.0, 253.0/255.0, 1],
                    [152.0/255.0,  84.0/255.0, 198.0/255.0, 1]
                    ])
        
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
        
    TH_map = np.zeros((lons.shape[0], lons.shape[1], len(radar.fixed_angle['data'])))
    LAT_map = np.zeros((lons.shape[0], lons.shape[1], len(radar.fixed_angle['data'])))
    LON_map = np.zeros((lons.shape[0], lons.shape[1], len(radar.fixed_angle['data'])))
        
    del start_index, end_index, lats, lons
    
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ALTITUDE = [0,0,0,1000,0,2000]
    loops = [0, 3, 5]
    #for nlev in range(len(radar.fixed_angle['data'])):
    for nlev in loops:
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats  = radar.gate_latitude['data'][start_index:end_index]
        lons  = radar.gate_longitude['data'][start_index:end_index]
    
        #-- Zh 
        if 'TH' in radar.fields.keys():  
            TH = radar.fields['TH']['data'][start_index:end_index]
        elif 'DBZH' in radar.fields.keys():
            TH = radar.fields['DBZH']['data'][start_index:end_index]
        elif 'reflectivity' in radar.fields.keys(): 
            TH = radar.fields['reflectivity']['data'][start_index:end_index]
            
        # Create flat surface.
        Z = np.zeros((lons.shape[0], lons.shape[1]))
        TH_nan = TH.copy()
        TH_nan[np.where(TH_nan<0)] = np.nan
        TH[np.where(TH<0)]     = 0
    
        # Normalize in [0, 1] the DataFrame V that defines the color of the surface.
        # TH_normalized = (TH_map[:,:,nlev] - TH_map[:,:,nlev].min().min())
        # TH_normalized = TH_normalized / TH_normalized.max().max()
        TH_normalized = TH_nan / np.nanmax( TH_nan )
    
        # Plot (me falta remplazar cm.jet! por cmap)  !!!! <<<< -----------------------------
        ax.plot_surface(lons[:,:], lats[:,:], Z+ALTITUDE[nlev], facecolors=plt.cm.jet(TH_normalized))
           
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(TH_map[:,:,nlev])
    plt.colorbar(m, ax=ax, shrink=1, ticks=np.arange(vmin,max, intt))
    
    # OJO me falta ponerle el cmap custom      
    
    # ------- NSWEEP 0 TEST W/ HEIGHT:
    
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
        
    ranges  = radar.range['data']
    azimuth = radar.azimuth['data']
    elev    = radar.elevation['data']
    
    gate_range   = np.zeros([np.int(len(azimuth)/len(radar.fixed_angle['data']))-1, len(ranges)])
    gate_azimuth = np.zeros([np.int(len(azimuth)/len(radar.fixed_angle['data']))-1, len(ranges)])
    elev_angle   = np.zeros([np.int(len(azimuth)/len(radar.fixed_angle['data']))-1, len(ranges)])
    
    for irange in range(len(ranges)):
        azimuth_counter = 0
        AZILEN       = np.int( len(azimuth) / len(radar.fixed_angle['data']) )-1
        gate_range[:,irange]   = ranges[irange]
        gate_azimuth[:,irange] = azimuth[azimuth_counter:azimuth_counter+AZILEN]
        elev_angle[:,irange]   = elev[azimuth_counter]
    
    [x,y,z] = pyart.core.transforms.antenna_to_cartesian(gate_range/1e3,
                gate_azimuth, elev_angle)
    [lonlon,latlat] = pyart.core.transforms.cartesian_to_geographic_aeqd(x, y,
                radar.longitude['data'], radar.latitude['data'], R=6370997.);

    #= PLOT FIGURE
    if 'TH' in radar.fields.keys():  
        TH = radar.fields['TH']['data'][start_index:end_index]
    elif 'DBZH' in radar.fields.keys():
        TH = radar.fields['DBZH']['data'][start_index:end_index]
    elif 'reflectivity' in radar.fields.keys(): 
        TH = radar.fields['reflectivity']['data'][start_index:end_index]
    
    fig=plt.pcolormesh(x/1e3, y/1e3, TH, 
            cmap=plt.cm.jet, vmin=vmin, vmax=vmax); plt.colorbar()
        
    # Create flat surface.
    TH_nan = TH.copy()
    TH_nan[np.where(TH_nan<0)] = np.nan
    TH[np.where(TH<0)]     = 0
    
    # Normalize in [0, 1] the DataFrame V that defines the color of the surface.
    # TH_normalized = (TH_map[:,:,nlev] - TH_map[:,:,nlev].min().min())
    # TH_normalized = TH_normalized / TH_normalized.max().max()
    TH_normalized = TH_nan / np.nanmax( TH_nan )
    
    # Plot x/y (me falta remplazar cm.jet! por cmap)  !!!! <<<< -----------------------------
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x[:,:]/1e3, y[:,:]/1e3, z[:,:]/1E3, facecolors=plt.cm.jet(TH_normalized))
    
    # Plot lat/lon
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(lons, lats, z[:,:]/1E3, facecolors=plt.cm.jet(TH_normalized))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax.set_zlabel('Altitude (km)')
    plt.title('nsweep 0')
    ax.view_init(15, 230)
    
    #plot_rhi_RMA(rfile, '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures', 
    #             '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/',
    #             'RMA1', 0, 300, 220, 0, np.nan)	

    return


    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 







    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'

    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO SUPERCELDA: 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    lon_pfs = [-64.80]
    lat_pfs = [-31.83]
    time_pfs = ['2058']
    phail   = [0.534]
    MIN85PCT = [131.1081]
    MIN37PCT = [207.4052]
    #
    rfile     = 'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
    rfile_1   = 'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
    rfile_2   = 'cfrad.20180208_205455.0000_to_20180208_205739.0000_RMA1_0201_02.nc'
    gfile     = '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5'
    #
    opts = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5}
    era5_file = '20180208_21_RMA1.grib'	
    #-------------------------- DONT CHANGE
    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile)
    radar_1 = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile_1)
    radar_2 = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile_2)
    plot_Zhppi_wGMIcontour(radar, lat_pfs, lon_pfs, 'radar at '+rfile[15:19]+' UTC and PF at '+time_pfs[0]+' UTC', 
                           gmi_dir+gfile, 0, opts, era5_dir+era5_file, icoi=3)
    #-------------------------- 
    # en el gridded se pueden hacer vertical slices? 
    grided = pyart.map.grid_from_radars(radar, grid_shape=(20, 470, 470), 
                                       grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                    (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                       roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')
    # otra opcion es mirar en superficie, y en freezing level !
    # adentro de los contornos? 
	
ARREGLAR EL TEMA DE LOS INDS EN LA FUNCION! 

    # que pasa si promedio entre dos tiempos de radar? ver perfiles dentro de contornos especificos? 
    for i in range(470):
	for j in range(470):
		plt.plot(grided.fields['TH']['data'][:,i,j], grided.point_z['data'][:,i,j]/1E3, '-k')	

		

    
    
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    axes[0].set_title('original radar resolution')
    axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    
    
    
    
    diff_x = np.zeros(( grided.nx-1 )); diff_x[:] = np.nan
    diff_y = np.zeros(( grided.ny-1 )); diff_y[:] = np.nan
    for i in range(len(grided.x['data'])-1):
        diff_x[i] = (grided.x['data'][i+1] - grided.x['data'][i]) /1E3
    for i in range(len(grided.y['data'])-1):
        diff_y[i] = (grided.y['data'][i+1] - grided.y['data'][i]) /1E3
        
    
    
    
    
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO 2018/11/11 1250 UTC que tiene tambien CSPR2
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ----  
    lon_pfs = [-64.53]
    lat_pfs = [-31.83]
    time_pfs = '1250'
    phail   = [0.653]
    MIN85PCT = [100.5397]
    MIN37PCT = [190.0287]
    #
    cspr2_RHI_file = 'corcsapr2cfrhsrhiqcM1.b1.20181214.125600.nc'
    rfile     = 'cfrad.20181111_124509.0000_to_20181111_125150.0000_RMA1_0301_01.nc'
    gfile     = '1B.GPM.GMI.TB2016.20181111-S113214-E130446.026724.V05A.HDF5'
    cspr2_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/CSPR2_data/'
    cspr2_file = 'corcsapr2cfrppiM1.a1.20181111.130003.nc' #'corcsapr2cfrppiM1.a1.20181111.124503.nc'
    #
    opts = {'xlim_min': -66, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -31}
    #-------------------------- DONT CHANGE
    radar = pyart.io.read(cspr2_dir+cspr2_file)
    plot_Zhppi_wGMIcontour(radar, lat_pfs, lon_pfs, 'radar at '+cspr2_file[30:34]+' UTC and PF at '+time_pfs+' UTC', 
                           gmi_dir+gfile, 0, opts)
    
    
   

	
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO 2019/03/08 02 UTC que tiene tambien CSPR2
    # 2019	03	08	02	04	 -30.75	 -63.74		0.895	 62.1525	147.7273
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    lon_pfs = [-63.74] 
    lat_pfs = [-30.75]  
    time_pfs = '0204'
    phail   = [0.895]
    MIN85PCT = [62.1525]
    MIN37PCT = [147.7273] 
    #
    rfile     = 'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc'
    gfile     = '1B.GPM.GMI.TB2016.20190308-S004613-E021846.028537.V05A.HDF5'
    era5_file = '20190308_02_RMA1.grib'
    #
    opts = {'xlim_min': -66, 'xlim_max': -62, 'ylim_min': -33, 'ylim_max': -30}
    #-------------------------- DONT CHANGE
    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile)
    plot_Zhppi_wGMIcontour(radar, lat_pfs, lon_pfs, 'radar at '+rfile[15:19]+' UTC and PF at '+time_pfs+' UTC', 
                           gmi_dir+gfile, 0, opts)
    





    # Plot for only one PF as it's to select the contours of interest 
    plot_gmi('/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'+gfile, opts,
                                  '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/', rfile, [lon_pfs[0]], [lat_pfs[0]], reference_satLOS=200)
    plt.title('GMI'+gfile[18:26]+' Phail='+str(phail[0])+' MIN85PCT: '+str(MIN85PCT[0])+' MIN37PCT:'+str(MIN37PCT[0]))
    #--------------------------
    # En base a plot_gmi, elijo los contornos que me interan 
    plot_gmi_wCOIs('/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'+gfile, opts,
                                  '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/', rfile, [lon_pfs[0]], [lat_pfs[0]], coi=[4], 
                  reference_satLOS=200) 
    # Inside radar PCTs (que en principio son PFs). look at TB distribution w/ MIN85PCT and MIN37PCT.  
    [lon_inside, lat_inside, lon_inside2, lat_inside2, tb_s1_cont_2, tb_s2_cont_2] = return_gmi_inside_contours(
        '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'+gfile, opts,
                                   '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/', rfile, [lon_pfs[0]], [lat_pfs[0]], 4,  reference_satLOS=200)
    # Test minPCTs: 
    PCT37 = np.min( (2.15 * tb_s1_cont_2[:,5]) - (1.15 * tb_s1_cont_2[:,6])) # == 125.57155   NOT 147.7 K perhaps use optimal Table 3 Cecil(2018)? 
    PCT89 = np.min( 1.7  * tb_s1_cont_2[:,7] - 0.7  * tb_s1_cont_2[:,8] )    # == 61.99839    ok 
    #- 
    # Plot RHIs w/ corrected ZDR, first calculate freezing level:
    ERA5_field = xr.load_dataset(era5_dir+era5_file, engine="cfgrib")
    elemj      = find_nearest(ERA5_field['latitude'], lon_pfs)
    elemk      = find_nearest(ERA5_field['longitude'], lat_pfs)
    tfield_ref = ERA5_field['t'][:,elemj,elemk] - 273 # convert to C
    geoph_ref  = (ERA5_field['z'][:,elemj,elemk])/9.80665
    # Covert to geop. height (https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
    Re         = 6371*1e3
    alt_ref    = (Re*geoph_ref)/(Re-geoph_ref)
    freezing_lev = np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3) 
    #    
    check_transec(radar, 55, lon_pfs, lat_pfs)     
    #    
    plot_rhi_RMA(rfile, '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures', '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/',
                 'RMA1', 0, 200, 55, 0.5, freezing_lev)


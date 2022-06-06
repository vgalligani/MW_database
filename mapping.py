from matplotlib import cm;

#------------------------------------------------------------------------------  
def plot_Zhppi_wGMIcontour(radar, lat_pf, lon_pf, general_title, fname, nlev, options):
    
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
    plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('k'), linewidths=1.5);
    
    plt.xlim([options['xlim_min'], options['xlim_max']])
    plt.ylim([options['ylim_min'], options['ylim_max']])

    plt.suptitle(general_title, fontsize=14)

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
    
        #-- Zh: 
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
    
    
    me falta ponerle el cmap custom
    arreglar altitude labels
    
    
    

fig = plt.figure()
ax = plt.axes(projection='3d')
im = ax.plot_surface(LON_map[:,:,0], LAT_map[:,:,0], Z, cstride=1, facecolors=cm.coolwarm(TH_map[:,:,0]), 
                     cmap=matplotlib.colors.ListedColormap(nws_ref_colors_transparent)); plt.colorbar(im)






    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contourf(lons,lats,TH_map[:,:,nlev],levels=0)

    
    yplane=0.8
ind = Y[:,0] < yplane
ax.contourf(X[ind,:], Y[ind,:], Z[ind,:], zdir='z', offset=0.3)
ax.contourf(X, Z, Y, zdir='y', offset=yplane)
    
    for nlev in range(len(radar.fixed_angle['data'])):
        im = ax.pcolor(lons,lats,TH_map[:,:,nlev]),alpha=0.1)
        ax.add_collection3d(im,zs=xslices[i],zdir='y')

plt.show()


    pcm1 = ax.pcolor(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
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
    plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('k'), linewidths=1.5);
    
    plt.xlim([options['xlim_min'], options['xlim_max']])
    plt.ylim([options['ylim_min'], options['ylim_max']])

    plt.suptitle(general_title, fontsize=14)

    return






fig = plt.figure()
ax = plt.axes(projection='3d')








    gmi_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'


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
    rfile     = 'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc'; 
              # 'cfrad.20180208_205455.0000_to_20180208_205739.0000_RMA1_0201_02.nc'
    gfile     = '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5'
    #
    opts = {'xlim_min': -66, 'xlim_max': -62, 'ylim_min': -33, 'ylim_max': -30}
    #-------------------------- DONT CHANGE
    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile)
    plot_Zhppi_wGMIcontour(radar, lat_pfs, lon_pfs, 'radar at '+rfile[15:19]+' UTC and PF at '+time_pfs[0]+' UTC', 
                           gmi_dir+gfile, nlev, opts)

    # corregir  
    
    
    
    
    
    
    
    
    
    
    sys_phase  = pyart.correct.phase_proc.det_sys_phase(radar, ncp_lev=0.8, rhohv_lev=0.8, ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP')
    corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], 0, 280)
    plot_test_ppi(radar , corr_phidp, lat_pfs, lon_pfs, 'radar at '+rfile[15:19]+' UTC and PF at '+time_pfs[0]+' UTC')







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


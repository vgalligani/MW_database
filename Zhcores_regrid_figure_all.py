#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:36:54 2023

@author: vito.galligani
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import alphashape
import shapely
from matplotlib.path import Path
import matplotlib
import pyart 
import gc 
import copy
from pandas import to_datetime
from scipy.interpolate import interp2d

#------------------------------------------------------------------------------

def points_in_polygon(pts, polygon):
    """
    Returns if the points are inside the given polygon,

    Implemented with angle accumulation.
    see: 
https://en.wikipedia.org/wiki/Point_in_polygon#Winding_number_algorithm

    :param np.ndarray pts: 2d points
    :param np.ndarray polygon: 2d polygon
    :return: Returns if the points are inside the given polygon, array[i] == True means pts[i] is inside the polygon.
    """
    polygon = np.vstack((polygon, polygon[0, :]))  # close the polygon (if already closed shouldn't hurt)
    sum_angles = np.zeros([len(pts), ])
    for i in range(len(polygon) - 1):
        v1 = polygon[i, :] - pts
        norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True)
        norm_v1[norm_v1 == 0.0] = 1.0  # prevent divide-by-zero nans
        v1 = v1 / norm_v1
        v2 = polygon[i + 1, :] - pts
        norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True)
        norm_v2[norm_v2 == 0.0] = 1.0  # prevent divide-by-zero nans
        v2 = v2 / norm_v2
        dot_prods = np.sum(v1 * v2, axis=1)
        cross_prods = np.cross(v1, v2)
        angs = np.arccos(np.clip(dot_prods, -1, 1))
        angs = np.sign(cross_prods) * angs
        sum_angles += angs

    sum_degrees = np.rad2deg(sum_angles)
    # In most cases abs(sum_degrees) should be close to 360 (inside) or to 0 (outside).
    # However, in end cases, points that are on the polygon can be less than 360, so I allow a generous margin..
    return abs(sum_degrees) > 90.0

#------------------------------------------------------------------------------
def set_plot_settings(var): 
    """
    -------------------------------------------------------------
    
    -------------------------------------------------------------
    OUT    [units, cmap, vmin, vmax, mac, int, under] todas de importancia
                    para colormpas de las variables del radar.      
    IN     var      Zhh, Zdr, Kdp, Ah, phidp, rhohv, Doppler Vel. 
    ------
    notes: mas colormaps en https://github.com/jjhelmus/pyart_colormaps_and_limits 
    """
    if var == 'Zhh':
        units = 'Reflectivity [dBZ]'
        cmap = colormaps('ref')
        vmin = 0
        vmax = 60
        max = 60.01
        intt = 5
        under = 'white'
        over = 'white'
    elif var == 'Zhh_2':
        units = 'Zh [dBZ]'
        cmap = colormaps('ref2')
        vmin = 0
        vmax = 60
        max = 60.01
        intt = 5
        under = 'white'
        over = 'black'
    elif var == 'Zdr':
        units = 'ZDR [dBZ]'
        vmin = -2
        vmax = 10
        max = 10.01
        intt = 1
        N = (vmax-vmin)/intt
        cmap = discrete_cmap(int(N), 'jet') # colormaps('zdr')
        under = 'white'
        over = 'white'
    elif var == 'Kdp':
        units = 'KDP [deg/km]'
        vmin = -0.1
        vmax = 4
        max = 4.1
        intt = 0.1
        N = (vmax-vmin)/intt
        cmap = pyart.graph.cm.Theodore16 #discrete_cmap(10, 'jet')
        under = 'white'
        over = 'black'
    elif var == 'Ah':
        units = '[dB/km]'
        vmin = 0
        vmax = 0.5
        max = 0.51
        intt = 0.05
        N = (vmax-vmin)/intt
        cmap = discrete_cmap(N, 'brg_r')
        under = 'black'
        over = 'white'
    elif var == 'phidp':
        units = 'PHIDP [deg]'
        vmin = 0
        vmax = 360
        max = 360.1
        intt = 10
        N = (vmax-vmin)/intt
        print(N)
        cmap = pyart.graph.cm.Wild25  #discrete_cmap(int(N), 'jet')
        under = 'white'
        over = 'white'
    elif var == 'rhohv':
        units = r'$\rho_{hv}$'
        vmin = 0.5
        vmax = 1.
        max = 1.01
        intt = 0.1
        N = round((vmax-vmin)/intt)
        cmap = pyart.graph.cm.RefDiff #fun.discrete_cmap(N, 'jet')
        under = 'white'
        over = 'white'
    else: #Doppler por ejemplo
        units = r'V$_r$ [m/s]'
        cmap = pyart.graph.cm.BuDRd18
        vmin = -10
        vmax = 10
        max = 10.01
        intt = 2
        under = 'white'
        over = 'white'    
        
    return units, cmap, vmin, vmax, max, intt, under, over

#------------------------------------------------------------------------------  
def colormaps(variable):
    """
    Choose colormap for a radar variable
    variable : str
       Radar variable to define which colormap to use. (e.g. ref,
       dv, zdr..) More to come
    returns : matplotlib ColorList
       A Color List for the specified *variable*.
    """
    import matplotlib.colors as colors

    # Definicion de las disntitas paletas de colores:
    nws_ref_colors =([ [ 1.0/255.0, 159.0/255.0, 244.0/255.0],
                    [  3.0/255.0,   0.0/255.0, 244.0/255.0],
                    [  2.0/255.0, 253.0/255.0,   2.0/255.0],
                    [  1.0/255.0, 197.0/255.0,   1.0/255.0],
                    [  0.0/255.0, 142.0/255.0,   0.0/255.0],
                    [253.0/255.0, 248.0/255.0,   2.0/255.0],
                    [229.0/255.0, 188.0/255.0,   0.0/255.0],
                    [253.0/255.0, 149.0/255.0,   0.0/255.0],
                    [253.0/255.0,   0.0/255.0,   0.0/255.0],
                    #[212.0/255.0,   0.0/255.0,   0.0/255.0, 0.4],
                    [188.0/255.0,   0.0/255.0,   0.0/255.0],
                    [248.0/255.0,   0.0/255.0, 253.0/255.0],
                    [152.0/255.0,  84.0/255.0, 198.0/255.0]
                    ])
    # Definicion de las disntitas paletas de colores:
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
    
    nws_zdr_colors = ([ [  1.0/255.0, 159.0/255.0, 244.0/255.0],
                    [  3.0/255.0,   0.0/255.0, 244.0/255.0],
                    [  2.0/255.0, 253.0/255.0,   2.0/255.0],
                    [  1.0/255.0, 197.0/255.0,   1.0/255.0],
                    [  0.0/255.0, 142.0/255.0,   0.0/255.0],
                    [253.0/255.0, 248.0/255.0,   2.0/255.0],
                    [229.0/255.0, 188.0/255.0,   2.0/255.0],
                    [253.0/255.0, 149.0/255.0,   0.0/255.0],
                    [253.0/255.0,   0.0/255.0,   0.0/255.0],
                    [188.0/255.0,   0.0/255.0,   0.0/255.0],
                    [152.0/255.0,  84.0/255.0, 198.0/255.0]
                    ])

    nws_dv_colors = ([  [0,  1,  1],
                    [0,  0.966666638851166,  1],
                    [0,  0.933333337306976,  1],
                    [0,  0.899999976158142,  1],
                    [0,  0.866666674613953,  1],
                    [0,  0.833333313465118,  1],
                    [0,  0.800000011920929,  1],
                    [0,  0.766666650772095,  1],
                    [0,  0.733333349227905,  1],
                    [0,  0.699999988079071,  1],
                    [0,  0.666666686534882,  1],
                    [0,  0.633333325386047,  1],
                    [0,  0.600000023841858,  1],
                    [0,  0.566666662693024,  1],
                    [0,  0.533333361148834,  1],
                    [0,  0.5,  1],
                    [0,  0.466666668653488,  1],
                    [0,  0.433333337306976,  1],
                    [0,  0.400000005960464,  1],
                    [0,  0.366666674613953,  1],
                    [0,  0.333333343267441,  1],
                    [0,  0.300000011920929,  1],
                    [0,  0.266666680574417,  1],
                    [0,  0.233333334326744,  1],
                    [0,  0.200000002980232,  1],
                    [0,  0.16666667163372,   1],
                    [0,  0.133333340287209,  1],
                    [0,  0.100000001490116,  1],
                    [0,  0.0666666701436043, 1],
                    [0,  0.0333333350718021, 1],
                    [0,  0,  1],
                    [0,  0,  0],
                    [0,  0,  0],
                    [0,  0,  0],
                    [0,  0,  0],
                    [1,  0,  0],
                    [1,  0.0322580635547638, 0],
                    [1,  0.0645161271095276, 0],
                    [1,  0.0967741906642914, 0],
                    [1,  0.129032254219055,  0],
                    [1,  0.161290317773819,  0],
                    [1,  0.193548381328583,  0],
                    [1,  0.225806444883347,  0],
                    [1,  0.25806450843811,   0],
                    [1,  0.290322571992874,  0],
                    [1,  0.322580635547638,  0],
                    [1,  0.354838699102402,  0],
                    [1,  0.387096762657166,  0],
                    [1,  0.419354826211929,  0],
                    [1,  0.451612889766693,  0],
                    [1,  0.483870953321457,  0],
                    [1,  0.516129016876221,  0],
                    [1,  0.548387110233307,  0],
                    [1,  0.580645143985748,  0],
                    [1,  0.612903237342834,  0],
                    [1,  0.645161271095276,  0],
                    [1,  0.677419364452362,  0],
                    [1,  0.709677398204803,  0],
                    [1,  0.74193549156189,   0],
                    [1,  0.774193525314331,  0],
                    [1,  0.806451618671417,  0],
                    [1,  0.838709652423859,  0],
                    [1,  0.870967745780945,  0],
                    [1,  0.903225779533386,  0],
                    [1,  0.935483872890472,  0],
                    [1,  0.967741906642914,  0],
                    [1,  1,  0]   ])


    cmap_nws_ref = colors.ListedColormap(nws_ref_colors)
    cmap_nws_zdr = colors.ListedColormap(nws_zdr_colors)
    cmap_nws_dv = colors.ListedColormap(nws_dv_colors)
    cmap_nws_ref_trans = colors.ListedColormap(nws_ref_colors_transparent)
    

    if variable == 'ref':
       return cmap_nws_ref
    if variable == 'ref2':
       return cmap_nws_ref_trans
        

    if variable == 'zdr':
       return cmap_nws_zdr

    if variable == 'dv':
       return cmap_nws_dv
   
#------------------------------------------------------------------------------
def calc_PCTs(TB_s1):
    """
    -------------------------------------------------------------
    PCTs(89 GHz) <= 200 K following Cecil et al., 2018
    PCT(10) = 2.5TBV  - 1.5TBH
    PCT(19) = 2.4TBV  - 1.4TBH
    PCT(37) = 2.15TBV - 1.15TBH
    PCT(89) = 1.7TBV  - 0.7TBH
    -------------------------------------------------------------
    OUT    PCT10   
           PCT19
           PCT37
           PCT89
    IN     TB_s1         
    -------------------------------------------------------------
    """
    PCT10 = 2.5  * TB_s1[:,:,0] - 1.5  * TB_s1[:,:,1] 
    PCT19 = 2.4  * TB_s1[:,:,2] - 1.4  * TB_s1[:,:,3] 
    PCT37 = 2.15 * TB_s1[:,:,5] - 1.15 * TB_s1[:,:,6] 
    PCT89 = 1.7  * TB_s1[:,:,7] - 0.7  * TB_s1[:,:,8] 
    
    return PCT10, PCT19, PCT37, PCT89

#------------------------------------------------------------------------------
def stack_ppis(radar, files_list, options):
    

    #- HERE MAKE PPIS SIMILAR TO RMA1S ... ? to achive the gridded field ...
    #- Radar sweep
    lats0        = radar.gate_latitude['data']
    lons0        = radar.gate_longitude['data']
    azimuths     = radar.azimuth['data']
    #
    Ze     = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); Ze[:]=np.nan
    ZDR    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); ZDR[:]=np.nan
    lon    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); lon[:]=np.nan
    lat    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); lat[:]=np.nan
    RHO    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); RHO[:]=np.nan
    PHIDP  = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); PHIDP[:]=np.nan
    HID    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); HID[:]=np.nan
    KDP    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); KDP[:]=np.nan
    approx_altitude = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); approx_altitude[:]=np.nan
    #gate_range      = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); gate_range[:]=np.nan
    alt_43aproox    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); alt_43aproox[:]=np.nan
    #
    gate_range      = np.zeros( [len(files_list), lats0.shape[1] ]); gate_range[:]=np.nan
    azy   = np.zeros( [len(files_list), lats0.shape[0] ]); azy[:]=np.nan
    fixed_angle     = np.zeros( [len(files_list)] ); fixed_angle[:]=np.nan
    #
    nlev = 0
    for file in files_list:
        print(file)
        if 'low_v176' in file:
            radar   = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/DOW7/'+file)
            fixed_angle[nlev] = radar.fixed_angle['data'].data[0]
            azy[nlev,:]  = radar.azimuth['data']
            ZHZH    = radar.fields['DBZHCC']['data']
            TV      = radar.fields['DBZVCC']['data']
            ZDRZDR  = radar.fields['ZDRC']['data']
            RHORHO  = radar.fields['RHOHV']['data']
            KDPKDP  = radar.fields['KDP']['data']
            PHIPHI  = radar.fields['PHIDP']['data']        
            gateZ    = radar.gate_z['data']
            gateX    = radar.gate_x['data']
            gateY    = radar.gate_y['data']
            gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
            #
            lats        = radar.gate_latitude['data']
            lons        = radar.gate_longitude['data']

            #------- aca hay que stack correctamente por azimuths?
            for TransectNo in range(lats0.shape[0]):
                [xgate, ygate, zgate]    = pyart.core.antenna_to_cartesian(gates_range[TransectNo,:]/1e3, azimuths[TransectNo], np.double(    file[41:45]) );
                Ze[nlev,TransectNo,:]    = ZHZH[TransectNo,:]
                ZDR[nlev,TransectNo,:]   = ZDRZDR[TransectNo,:]
                RHO[nlev,TransectNo,:]   = RHORHO[TransectNo,:]
                PHIDP[nlev,TransectNo,:] = PHIPHI[TransectNo,:]
                KDP[nlev,TransectNo,:]   = KDPKDP[TransectNo,:]
                lon[nlev,TransectNo,:]   = lons[TransectNo,:]
                lat[nlev,TransectNo,:]   = lats[TransectNo,:]

            gc.collect()
            nlev = nlev + 1


    # From https://arm-doe.github.io/pyart/notebooks/basic_ingest_using_test_radar_object.html
    radar_stack = pyart.testing.make_empty_ppi_radar(lat.shape[2], lat.shape[1], lat.shape[0])
    # Start filling the radar attributes with variables in the dataset.
    radar_stack.latitude['data']    = np.array([radar.latitude['data'][0]])
    radar_stack.longitude['data']   = np.array([radar.longitude['data'][0]])
    radar_stack.range['data']       = np.array( radar.range['data'][:] )
    radar_stack.fixed_angle['data'] = np.array( fixed_angle )
    azi_all = []
    rays_per_sweep = []
    raye_per_sweep = []
    rayN = 0
    elev = np.zeros( [azy.shape[0]*azy.shape[1]] ); elev[:]=np.nan
    for i in range(azy.shape[0]):
        rays_per_sweep.append(rayN)
        rayN = rayN + azy.shape[1]
        raye_per_sweep.append(rayN-1)
        for j in range(azy.shape[1]):
            azi_all.append(  azy[i,j] )
    ii = 0
    for i in range(azy.shape[0]):
        for j in range(azy.shape[1]):
            elev[ii] = fixed_angle[i]
            ii=ii+1
    radar_stack.azimuth['data'] = np.array( azi_all )
    radar_stack.sweep_number['data'] = np.array(  np.arange(0,nlev,1) )
    radar_stack.sweep_start_ray_index['data'] = np.array( rays_per_sweep )
    radar_stack.sweep_end_ray_index['data']   = np.array( raye_per_sweep )
    radar_stack.altitude['data']        = [ radar.altitude['data'][0] ]
    # elevation is theta too.
    radar_stack.elevation['data'] = elev
    radar_stack.init_gate_altitude()
    radar_stack.init_gate_longitude_latitude()

    #plt.plot(radar_stack.gate_longitude['data'], radar_stack.gate_latitude['data'], 'ok')
    # Let's work on the field data, we will just do reflectivity for now, but any of the
    # other fields can be done the same way and added as a key pair in the fields dict.
    from pyart.config import get_metadata
    Ze_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); Ze_all[:]=np.nan
    ZDR_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); ZDR_all[:]=np.nan
    RHOHV_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); RHOHV_all[:]=np.nan
    PHIDP_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); PHIDP_all[:]=np.nan
    KDP_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); KDP_all[:]=np.nan
    ii = 0
    for i in range(azy.shape[0]):
        for j in range(Ze.shape[1]):
            Ze_all[ii,:]    = Ze[i,j,:]
            ZDR_all[ii,:]   = ZDR[i,j,:]
            RHOHV_all[ii,:] = RHO[i,j,:]
            PHIDP_all[ii,:] = PHIDP[i,j,:]
            KDP_all[ii,:] = KDP[i,j,:]
            ii=ii+1

    #- REFLECTIVITY
    ref_dict_ZH = get_metadata('DBZHCC')
    ref_dict_ZH['data'] = np.array(Ze_all)

    #- ZDR
    ref_dict_ZDR = get_metadata('ZDRC')
    ref_dict_ZDR['data'] = np.array(ZDR_all)

    #- RHOHV
    ref_dict_RHOHV = get_metadata('RHOHV')
    ref_dict_RHOHV['data'] = np.array(RHOHV_all)

    #- PHIDP
    ref_dict_PHIDP = get_metadata('PHIDP')
    ref_dict_PHIDP['data'] = np.array(PHIDP_all)

    #- KDP
    ref_dict_KDP = get_metadata('KDP')
    ref_dict_KDP['data'] = np.array(KDP_all)


    radar_stack.fields = {'DBZHCC': ref_dict_ZH,
              'ZDRC':   ref_dict_ZDR,
              'RHOHV':  ref_dict_RHOHV,
              'PHIDP':  ref_dict_PHIDP,
              'KDP':    ref_dict_KDP}

    return radar_stack

#------------------------------------------------------------------------------
def get_colmax(radar, field, gatefilter):
    
    # Determine the lowest sweep (used for metadata and such)
    minimum_sweep = np.min(radar.sweep_number["data"])
    
    # loop over all measured sweeps
    for sweep in sorted(radar.sweep_number["data"]):
        # get start and stop index numbers
        sweep_slice = radar.get_slice(sweep)

        # grab radar data
        z = radar.get_field(sweep, field)
        z_dtype = z.dtype

        # Use gatefilter
        if gatefilter is not None:
           mask_sweep = gatefilter.gate_excluded[sweep_slice, :]
           z = np.ma.masked_array(z, mask_sweep)

        # extract lat lons
        lon = radar.gate_longitude["data"][sweep_slice, :]
        lat = radar.gate_latitude["data"][sweep_slice, :]   
        
        # get the range and time
        ranges = radar.range["data"]
        time = radar.time["data"]

        # get azimuth
        az = radar.azimuth["data"][sweep_slice]
        # get order of azimuths
        az_ids = np.argsort(az)

        # reorder azs so they are in order
        az = az[az_ids]
        z = z[az_ids]
        lon = lon[az_ids]
        lat = lat[az_ids]
        time = time[az_ids]       
        
        # if the first sweep, store re-ordered lons/lats
        if sweep == minimum_sweep:
            azimuth_final = az
            time_final = time
            lon_0 = copy.deepcopy(lon)
            lon_0[-1, :] = lon_0[0, :]
            lat_0 = copy.deepcopy(lat)
            lat_0[-1, :] = lat_0[0, :]
    
        else:
            # Configure the intperpolator
            z_interpolator = interp2d(ranges, az, z, kind="linear")

            # Apply the interpolation
            z = z_interpolator(ranges, azimuth_final)
            
        # if first sweep, create new dim, otherwise concat them up
        if sweep == minimum_sweep:
            z_stack = copy.deepcopy(z[np.newaxis, :, :])
        else:
            z_stack = np.concatenate([z_stack, z[np.newaxis, :, :]])

    
    # now that the stack is made, take max across vertical
    out_compz = np.nanmax(z_stack.data, axis=0) #.astype(z_dtype)
    
    return lon_0,lat_0,out_compz

#------------------------------------------------------------------------------
def GET_contours(options, icois, fname, radar, title_in):

    home_dir = '/Users/vito.galligani/'  	
    gmi_dir  = '/Users/vito.galligani/Work/Studies/Hail_MW/GMI_data/'
	
    changui = 1
    # ojo que aca agarro los verdaderos PCTMIN, no los que me pasó Sarah B. que estan 
    # ajustados a TMI footprints. 
    # read file
    
    colores_in = ['k','darkgreen','darkred','darkblue','cyan']     

    f = h5py.File( fname, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()

    S1_sub_lat  = lat_gmi.copy()
    S1_sub_lon  = lon_gmi.copy()
    S1_sub_tb = tb_s1_gmi.copy()

    idx1       = (lat_gmi>=options['ylim_min']-changui) & (lat_gmi<=options['ylim_max']+changui) & (lon_gmi>=options['xlim_min']-changui) & (lon_gmi<=options['xlim_max']+changui)
    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	

    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)

    nlev = 1

    # Configure a gatefilter to filter out copolar correlation coefficient values > 0.9
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    
    
    #-----------------------------------
    # radares:
    if options['radar_name'] == 'RMA1':
        reflectivity_name = 'TH'   
        RHOHVname = 'RHOHV'
        nx = 940
        ny = 940
        gatefilter.exclude_equal(RHOHVname, 0.9)
        # Calculate composite reflectivity, or the maximum reflectivity across all elevation levels
     
    if options['radar_name'] == 'RMA5':
        reflectivity_name = 'DBZH'   
        RHOHVname = 'RHOHV'
        nx = 570
        ny = 570
        gatefilter.exclude_equal(RHOHVname, 0.9)
        # Calculate composite reflectivity, or the maximum reflectivity across all elevation levels
        
    #-----------------------------------  
    # REGRID
    grided  = pyart.map.grid_from_radars(radar, grid_shape=(20, nx, ny), 
            grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
           (-np.max(radar.range['data']), np.max(radar.range['data']))),
            roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')   
    gc.collect()
    diff_nx = grided.point_x['data'][0,0,1]-grided.point_x['data'][0,0,0]
    print('dx resolution of re-grid is: ' + str(diff_nx))    
    
    # Get colmax
    grid_colmax = np.nanmax(grided.fields[reflectivity_name]['data'], axis=0) 
     
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')

    # check 
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[14,12])
    pcm1 = plt.pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:],  
                          grided.fields[reflectivity_name]['data'][0,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, shrink=1, label=units, ticks = np.arange(vmin,max,intt))         
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    plt.grid(True)    
    plt.contour(grided.point_longitude['data'][0,:,:], 
                      grided.point_latitude['data'][0,:,:], grid_colmax, [45], colors='darkblue', linewidths=4)              
    axes.set_xlim([options['xlim_min'], options['xlim_max']]) 
    axes.set_ylim([options['ylim_min'], options['ylim_max']])

    # check:
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[14,12])
    contorno89 = plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200] , colors=(['r']), linewidths=1.5);
    contorno89_FIX = plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:] , [200], colors=(['k']), linewidths=1.5);
    axes.set_xlim([options['xlim_min'], options['xlim_max']]) 
    axes.set_ylim([options['ylim_min'], options['ylim_max']])
    contorno45dBZ = plt.contour(grided.point_longitude['data'][0,:,:], 
                          grided.point_latitude['data'][0,:,:], grid_colmax, [45], colors='darkblue', linewidths=4)   
    plt.grid(True)                 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    plt.title('General summary: gridded contours')
           
    # Ordenar por los contornos mas grandes asi es mas facil identificar 
    area_col = []
    contour_col = contorno45dBZ.collections[0]
    for i in range(len(contour_col.get_paths())):
        vs = contour_col.get_paths()[i].vertices
        x = vs[:,0]
        y = vs[:,1]
        area = 0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
        area=np.abs(area)
        area_col.append(area)
    area_col = np.array(area_col)
    index_list = area_col.argsort().tolist()[::-1]
        
    #Now plot the first 3 largest
    for ii in range(len(icois)): 
        i = index_list[icois[ii]]
        print(i)
        print(area_col[i])
        vs = contour_col.get_paths()[i].vertices
        x = vs[:,0]
        y = vs[:,1]
        plt.plot(x,y,'ok')
        del x, y, vs

    # GET X and Y points that make the contorno for each contorno: 
    # using https://stackoverflow.com/questions/70701788/find-points-that-lie-in-a-concave-hull-of-a-point-cloud
    area_45dBZ   = []
    max45dBZ_hgt = []
        
    lons_nlev0 = grided.point_longitude['data'][0,:,:]
    lats_nlev0 = grided.point_latitude['data'][0,:,:]
    for ii in range(len(icois)): 
        X1 = []; Y1 = []; vertices = []
        i = index_list[icois[ii]]
        for ik in range(len(contorno45dBZ.collections[0].get_paths()[i].vertices)):
            X1.append(contorno45dBZ.collections[0].get_paths()[i].vertices[ik][0])
            Y1.append(contorno45dBZ.collections[0].get_paths()[i].vertices[ik][1])            
        points = np.vstack([X1, Y1]).T
        
        # DATAPOINTS that I wish to know if they are inside the contour of interest 
        datapts = np.column_stack(( np.ravel(lons_nlev0), np.ravel(lats_nlev0) ))
        cond = points_in_polygon(datapts, points)
        plt.plot(np.ravel(lons_nlev0)[cond] , np.ravel(lats_nlev0)[cond], 'or', markersize=10 )    
        
                    
        gate_areas    = 500*500 # this is the new grid? 
        Ngates_inside = len(np.ravel(lons_nlev0)[cond])
        area_45dBZ.append( round( np.sum(gate_areas*Ngates_inside)*1e-6,1) ) # Array containing the area (km2) for each coi
        
        # Calculate the max height at which there is a > 45dBZ
        # Re-calculate contour inside the grid for each nlev
        maxhgt_pernlev = []
        for i in range(grided.point_longitude['data'].shape[0]): 
            fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[14,12])
            pcm1 = plt.pcolormesh(grided.point_longitude['data'][i,:,:],
                                  grided.point_latitude['data'][i,:,:],
                                  grided.fields[reflectivity_name]['data'][i,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
            # Vuelvo a graficar el contorno de colmax
            plt.contour(grided.point_longitude['data'][0,:,:], 
                          grided.point_latitude['data'][0,:,:], grid_colmax, [45], colors='darkblue', linewidths=4)   
            plt.grid(True)               
            cbar = plt.colorbar(pcm1, shrink=1, label=units, ticks = np.arange(vmin,max,intt))         
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            axes.set_xlim([options['xlim_min'], options['xlim_max']]) 
            axes.set_ylim([options['ylim_min'], options['ylim_max']])
            # grafico en otro color el contorno de ese nivel:
            plt.contour(grided.point_longitude['data'][i,:,:], 
                          grided.point_latitude['data'][i,:,:], grided.fields[reflectivity_name]['data'][i,:,:], 
                          [45], colors='darkred', linewidths=4)   
            # define datapts for each lev
            nlon = grided.point_longitude['data'][i,:,:]
            nlat = grided.point_latitude['data'][i,:,:]
            nZh  = grided.fields[reflectivity_name]['data'][i,:,:]
            nalt = grided.point_altitude['data'][i,:,:]
            datapts = np.column_stack(( np.ravel(nlon), np.ravel(nlat) ))           
            # Now find points inside the colmaxDbZ because points == colmax contour 
            cond = points_in_polygon(datapts, points)
            plt.title('nsweep Nr: '+str(i))

            # For each sweep get the points within contour of interest check if maxdbZ > 45. 
            ind = np.where(np.ravel(nZh)[cond] > 45) 
            print('-------- nsweep Nr: '+str(i))
            if len(ind[0]>0):
                maxhgt_pernlev.append(np.nanmax( np.ravel(nalt)[cond][ind]))    
                print(np.nanmax( np.ravel(nalt)[cond][ind]))   
            else:
                print('no ind available')
                
        max45dBZ_hgt.append( round(np.nanmax(maxhgt_pernlev)/1e3,1) )
                
                
    return area_45dBZ, max45dBZ_hgt
#------------------------------------------------------------------------------  

def reproject(latitude, longitude):
    """Returns the x & y coordinates in meters using a sinusoidal projection"""
    from math import pi, cos, radians
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0

    y = [lat * lat_dist for lat in latitude]
    x = [long * lat_dist * cos(radians(lat)) 
                for lat, long in zip(latitude, longitude)]
    return x, y


#------------------------------------------------------------------------------  
def area_of_polygon(x, y):
    """Calculates the area of an arbitrary polygon given its verticies"""
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0

#------------------------------------------------------------------------------  
def pyplot_rings(lat_radar,lon_radar,radius):
    """
    Calculate lat-lon of the maximum range ring
    lat_radar : float32
       Radar latitude. Positive is north.
    lon_radar : float32
       Radar longitude. Positive is east.
    radius : float32
       Radar range in kilometers.
    returns : numpy.array
       A 2d array containing the 'radius' range latitudes (lat) and longitudes (lon)
    """
    import numpy as np

    R=12742./2.
    m=2.*np.pi*R/360.
    alfa=np.arange(-np.pi,np.pi,0.0001)

    nazim  = 360.0
    nbins  = 480.0
    binres = 0.5

    #azimuth = np.transpose(np.tile(np.arange(0,nazim,1), (int(nbins),1)))
    #rangos  = np.tile(np.arange(0,nbins,1)*binres, (int(nazim),1))
    #lats    = lat_radar + (rangos/m)*np.cos((azimuth)*np.pi/180.0)
    #lons    = lon_radar + (rangos/m)*np.sin((azimuth)*np.pi/180.0)/np.cos(lats*np.pi/180)

    lat_radius = lat_radar + (radius/m)*np.sin(alfa)
    lon_radius = lon_radar + ((radius/m)*np.cos(alfa)/np.cos(lat_radius*np.pi/180))

    return lat_radius, lon_radius


#------------------------------------------------------------------------------

plt.matplotlib.rc('font', family='serif', size = 12)
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12  
    
tbvbin = 15
TBV_bin  = np.arange(50,300,tbvbin)

gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
r_dir    = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'

        



do_this = 0
if do_this == 1: 
            
    #----------------------------------------------------------------------------------------------     
    # RMA1 - 8/2/2018
    # con contornos de 250 K, usamos coi=3 y coi=4, donde solo coi=4 tiene Phail > 50% 
    gfile    = '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5'  #21UTC
    rfile    = 'RMA1/cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
    radar = pyart.io.read(r_dir+rfile)

    opts           = {'xlim_min': -65.5, 'xlim_max': -63, 
                      'ylim_min': -33, 'ylim_max': -30.5, 'radar_name': 'RMA1'}

    # para 45 dBZ. con contorno
    coi_45 = [0]

    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar, '08/02/2018')
    print('------------------------------------------------------------------------------')
    for i in range(len(coi_45)): 
        print('08/02/2018    ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) )

    # #----------------------------------------------------------------------------------------------     
    # # # RMA1 - 03/08/2019
    gfile = '1B.GPM.GMI.TB2016.20190308-S004613-E021846.028537.V05A.HDF5'
    rfile = 'RMA1/cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc'
    radar = pyart.io.read(r_dir+rfile)
     
    opts = {'xlim_min': -65.2, 'xlim_max': -62, 'ylim_min': -33, 'ylim_max': -30, 'radar_name':'RMA1'}
    
    # para 45 dBZ. con contorno
    coi_45 = [0]
    
    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar, '03/08/2019')
    print('------------------------------------------------------------------------------')
    for i in range(len(coi_45)): 
       print('03/08/2019    ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) )    
       
       
do_this = 1
if do_this == 1: 
    
    # #----------------------------------------------------------------------------------------------     
    # # # # RMA5 - 15/08/2020
    gfile    = '1B.GPM.GMI.TB2016.20200815-S015947-E033219.036720.V05A.HDF5'
    rfile = 'RMA5/cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc' 
    radar = pyart.io.read(r_dir+rfile)
    
    opts = {'xlim_min': -55.0, 'xlim_max': -52.0, 'ylim_min': -27.5, 'ylim_max': -25.0, 'radar_name': 'RMA5'}
    
    # para 45 dBZ. con contorno
    coi_45 = [0]
    
    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar, '15/08/2020')
    print('------------------------------------------------------------------------------')
    for i in range(len(coi_45)): 
         print('15/08/2020   ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) )
         

do_this = 0
if do_this == 1: 
    

     
    # # #----------------------------------------------------------------------------------------------     
    # # # # RMA4 - 09/02/2018
    gfile    = '1B.GPM.GMI.TB2016.20180209-S184820-E202054.022451.V05A.HDF5' 
    rfile = 'RMA4/cfrad.20180209_200449.0000_to_20180209_201043.0000_RMA4_0200_01.nc' 
    radar = pyart.io.read(r_dir+rfile)
       
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 'radar_name': 'RMA4'}
    
    # para 45 dBZ. con contorno 
    coi_45 = [0] # tambien son 1,2,3
    
    print('------------------------------------------------------------------------------')
    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar,'09/02/2018')
    for i in range(len(coi_45)): 
        print('09/02/2018   ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) )
    

    # #----------------------------------------------------------------------------------------------       
    # # # # RMA4 - 31/10/2018
    gfile    = '1B.GPM.GMI.TB2016.20181031-S005717-E022950.026546.V05A.HDF5' 
    rfile    = 'RMA4/cfrad.20181031_010936.0000_to_20181031_011525.0000_RMA4_0200_01.nc' 
    radar = pyart.io.read(r_dir+rfile)
      
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -25.5, 'radar_name': 'RMA4'}
     
    # para 45 dBZ. con contorno 
    coi_45 = [6,7,8,9] 
    
    print('------------------------------------------------------------------------------')
    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar,'31/10/2018')
    for i in range(len(coi_45)): 
        print('31/10/2018   ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) )
        
    # #----------------------------------------------------------------------------------------------   
    # # # RMA3 - 05/03/2019
    gfile     = '1B.GPM.GMI.TB2016.20190305-S123614-E140847.028498.V05A.HDF5'
    rfile     = 'RMA3/cfrad.20190305_124638.0000_to_20190305_125231.0000_RMA3_0200_01.nc'
    radar = pyart.io.read(r_dir+rfile)
     
    opts = {'xlim_min': -63, 'xlim_max': -58, 'ylim_min': -27, 'ylim_max': -23, 'radar_name':'RMA3'}
     
    # para 45 dBZ. con contorno 
    coi_45 = [0,4] # 4 y 11
    
    print('------------------------------------------------------------------------------')
    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar,'05/03/2019')
    for i in range(len(coi_45)): 
        print('05/03/2019   ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) )


    # #----------------------------------------------------------------------------------------------  
    # # # RMA4 - 09/02/2019
    gfile    = '1B.GPM.GMI.TB2016.20190209-S191744-E205018.028129.V05A.HDF5'
    rfile    = 'RMA4/cfrad.20190209_192724.0000_to_20190209_193317.0000_RMA4_0200_01.nc' 
    radar = pyart.io.read(r_dir+rfile)
      
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 'radar_name':'RMA4'}
    
    # para 45 dBZ. con contorno 
    coi_45 = [4]
    
    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar, '09/02/2019')
    print('------------------------------------------------------------------------------')
    for i in range(len(coi_45)): 
        print('09/02/2019   ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) ) 

    # #----------------------------------------------------------------------------------------------     
    # # # CSPR2 - 11/11/2018    
    gfile     = '1B.GPM.GMI.TB2016.20181111-S113214-E130446.026724.V05A.HDF5'
    rfile     = 'CSPR2_data/corcsapr2cmacppiM1.c1.20181111.130003.nc' #'corcsapr2cfrppiM1.a1.20181111.130003.nc'
    radar = pyart.io.read(r_dir+rfile)

    opts = {'xlim_min': -66, 'xlim_max': -63.6, 'ylim_min': -32.6, 'ylim_max': -31.5, 'radar_name': 'CSPR2'}

    # para 45 dBZ. con contorno == 17
    coi_45 = [14]

    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar, '11/11/2018')
    for i in range(len(coi_45)): 
        print('11/11/2018    ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) )

    # #----------------------------------------------------------------------------------------------
    # # # DOW7 - 14/12/2018 
    gfile     = '1B.GPM.GMI.TB2016.20181214-S015009-E032242.027231.V05A.HDF5'
    rfile     = 'DOW7/cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc'
    radar = pyart.io.read(r_dir+rfile)
     
    opts = {'xlim_min': -65.0, 'xlim_max': -63.50, 'ylim_min': -32.2, 'ylim_max': -31.2, 'radar_name':'DOW7'}

    # para 45 dBZ. con contorno
    coi_45 = [0, 2]

    [area_45dBZ, hgt45dBZ] = GET_contours(opts, coi_45, gmi_dir+gfile, radar, '14/12/2018')
    for i in range(len(coi_45)): 
        print('14/12/2018    ------    area: '+str(area_45dBZ[i])+'   hgt: '+str(hgt45dBZ[i]) )
        
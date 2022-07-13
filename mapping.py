from matplotlib import cm;
from os import listdir
import pyart
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
import matplotlib
import matplotlib.gridspec as gridspec
import platform
from matplotlib.colors import ListedColormap
from os.path import isfile, join
import pandas as pd
from pyart.correct import phase_proc
import xarray as xr
from pyart.core.transforms import antenna_to_cartesian
from copy import deepcopy
import matplotlib.colors as colors
import wradlib as wrl    
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.path import Path
from cycler import cycler
#import seaborn as sns

import copy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import alphashape
from descartes import PolygonPatch
import cartopy.feature as cfeature

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import h5py
from csu_radartools import csu_fhc

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
#------------------------------------------------------------------------------  
def make_densityPlot(radarTH, radarZDR, RN_COI1, RN_COI2, RN_COI3):	
    # FIGURE  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]] 
    xedges = np.arange(-10, 70, 1)
    yedges = np.arange(-10, 11, 0.5)
    xx, yy = np.meshgrid(yedges[0:-1],xedges[0:-1])
    H1, xedges, yedges = np.histogram2d(np.ravel(radarTH)[RN_COI1], y=np.ravel(radarZDR)[RN_COI1]-opts['ZDRoffset'], 
				bins=(xedges, yedges), normed=True)
    areas1 = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
    H2, xedges, yedges = np.histogram2d(np.ravel(radarTH)[RN_COI2], y=np.ravel(radarZDR)[RN_COI2]-opts['ZDRoffset'], 
			bins=(xedges, yedges), normed=True)
    areas2 = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))	
    H3, xedges, yedges = np.histogram2d(np.ravel(radarTH)[RN_COI3], y=np.ravel(radarZDR)[RN_COI3]-opts['ZDRoffset'], 
			bins=(xedges, yedges), normed=True)
    areas3 = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
    vmax1 = np.nanmax([H1*areas1][0])
    vmax2 = np.nanmax([H2*areas2][0])
    vmax3 = np.nanmax([H3*areas3][0])
    VMAXX = np.round(np.nanmax([vmax1, vmax2, vmax3]),2)
    fig = plt.figure(figsize=(12,12)) 
    gs1 = gridspec.GridSpec(1, 3)
    ax1 = plt.subplot(gs1[0,0])
    im1 = ax1.pcolormesh(yy, xx, [H1*areas1][0], vmin=0, vmax=VMAXX)
    ax1.set_title('coi=1')
    ax2 = plt.subplot(gs1[0,1])
    im2 = ax2.pcolormesh(yy, xx, [H2*areas2][0], vmin=0, vmax=VMAXX)
    ax2.set_title('coi=3 [Phail = 0.534]')
    ax3 = plt.subplot(gs1[0,2])   
    im3 = ax3.pcolormesh(yy, xx, [H3*areas3][0], vmin=0, vmax=VMAXX)
    #plt.colorbar(im3, ax=ax3, shrink=0.5)
    ax3.set_title('coi=4')
    # common 
    ax1.set_ylim([-10, 10])
    ax2.set_ylim([-10, 10])
    ax3.set_ylim([-10, 10])
    ax1.set_xlim([0, 65])
    ax2.set_xlim([0, 65])
    ax3.set_xlim([0, 65])	
    ax1.set_ylabel('ZDR')
    ax1.set_xlabel('ZH')        ; ax1.grid('True')
    ax2.grid('True')
    ax3.grid('True') 
    ax2.set_yticklabels([])    
    ax3.set_yticklabels([])
    # common colobar                   
    p1 = ax1.get_position().get_points().flatten()
    p2 = ax2.get_position().get_points().flatten();
    p3 = ax3.get_position().get_points().flatten(); 
    ax_cbar = fig.add_axes([p3[0]+(p3[0]-p2[0]), 0.15, 0.04, 0.7])   #ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
    cbar = fig.colorbar(im3, cax=ax_cbar, shrink=0.9, ticks=np.arange(0,np.round(VMAXX,2)+0.01,0.01)); 
    return

#------------------------------------------------------------------------------  
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    import matplotlib.pyplot as plt
    import numpy as np

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    
    return base.from_list(cmap_name, color_list, N)

#------------------------------------------------------------------------------  
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def plot_Zhppi_wGMIcontour(radar, lat_pf, lon_pf, general_title, fname, nlev, options, era5_file, icoi, use_freezingLev):
    fontize = 20

    plt.matplotlib.rc('font', family='serif', size = 20)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['font.serif'] = ['Helvetica']

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

    # keep domain of interest only by keeping those where the center nadir obs is inside domain
    inside_s1   = np.logical_and(np.logical_and(lon_gmi >= options['xlim_min'], lon_gmi <=  options['xlim_max']), 
                              np.logical_and(lat_gmi >= options['ylim_min'], lat_gmi <= options['ylim_max']))
    inside_s2   = np.logical_and(np.logical_and(lon_s2_gmi >= options['xlim_min'], lon_s2_gmi <=  options['xlim_max']), 
                                         np.logical_and(lat_s2_gmi >= options['ylim_min'], lat_s2_gmi <= options['ylim_max']))    
    lon_gmi_inside   = lon_gmi[inside_s1] 
    lat_gmi_inside   = lat_gmi[inside_s1] 	
    lon_gmi2_inside  = lon_s2_gmi[inside_s2] 	
    lat_gmi2_inside  = lat_s2_gmi[inside_s2] 	
    tb_s1_gmi_inside = tb_s1_gmi[inside_s1, :]

    PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 

    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    #------ 
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
    grided2 = pyart.map.grid_from_radars(radar, grid_shape=(20, 94, 94), 
                                   grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                   roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')
    grided3 = pyart.map.grid_from_radars(radar, grid_shape=(20, 48, 48), 
                               grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                            (-np.max(radar.range['data']), np.max(radar.range['data']))),
                               roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')
#------ 
    print('ERA5 freezing level at: (km)'+str(freezing_lev))
    frezlev = find_nearest(grided.z['data']/1e3, freezing_lev) 
    print('Freezing level at level:'+str(frezlev)+'i.e., at'+str(grided.z['data'][frezlev]/1e3))
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
	
        #------- testing from https://stackoverflow.com/questions/57260352/python-concave-hull-polygon-of-a-set-of-lines 
        alpha = 0.95 * alphashape.optimizealpha(array_points)
        hull_pts_CONCAVE = alphashape.alphashape(array_points, alpha)
        #axes.add_patch(PolygonPatch(hull_pts_CONCAVE, fill=False, color='m'))
        hull_coors_CONCAVE = hull_pts_CONCAVE.exterior.coords.xy
        check_points = np.vstack((hull_coors_CONCAVE)).T
        concave_path = Path(check_points)
        #-------
        ##--- Run hull_paths and intersec
        hull_path   = Path( array_points[convexhull.vertices] )
        datapts = np.column_stack((lon_gmi_inside,lat_gmi_inside))
        #datapts = np.column_stack((np.ravel(lon_gmi[1:,:]),np.ravel(lat_gmi[1:,:])))
        # Ahora como agarro los Zh, ZDR, etc inside the contour ... 
        datapts_RADAR_NATIVE = np.column_stack((np.ravel(lons),np.ravel(lats)))
    	# Ahora agarro los Zh, ZDR, etc inside countour pero con el GRIDED BARNES2 at FREEZELEVL O GROUND LEVEL? 
        if use_freezingLev == 1: 
           datapts_RADAR_BARNES = np.column_stack((np.ravel(grided.point_longitude['data'][frezlev,:,:]),
							np.ravel(grided.point_latitude['data'][frezlev,:,:])))
       	else: 
            datapts_RADAR_BARNES = np.column_stack((np.ravel(grided.point_longitude['data'][0,:,:]),
							np.ravel(grided.point_latitude['data'][0,:,:])))
        if ii==0:
            inds_1  = concave_path.contains_points(datapts)
       	   #inds_1 = hull_path.contains_points(datapts)
            inds_RN1 = hull_path.contains_points(datapts_RADAR_NATIVE)
            inds_RB1 = hull_path.contains_points(datapts_RADAR_BARNES)
        if ii==1:
            inds_2  = concave_path.contains_points(datapts)
            #inds_2 = hull_path.contains_points(datapts)
            inds_RN2 = hull_path.contains_points(datapts_RADAR_NATIVE)
            inds_RB2 = hull_path.contains_points(datapts_RADAR_BARNES)
        if ii==2:
            inds_3 = hull_path.contains_points(datapts)
            inds_RN3 = hull_path.contains_points(datapts_RADAR_NATIVE)
            inds_RB3 = hull_path.contains_points(datapts_RADAR_BARNES)


    plt.xlim([options['xlim_min'], options['xlim_max']])
    plt.ylim([options['ylim_min'], options['ylim_max']])

    plt.suptitle(general_title, fontsize=fontize)

    #---- NEW FIGURE 
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    fig, axes = plt.subplots(nrows=1, ncols=4, constrained_layout=True,
                        figsize=[25,6])
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    axes[0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('original radar resolution')
    axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[1].pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:], 
                  grided.fields['TH']['data'][0,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes[1].set_title('1 km gridded BARNES2')
    CS = axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    axes[1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1].set_ylim([options['ylim_min'], options['ylim_max']])
    #
    axes[2].pcolormesh(grided2.point_longitude['data'][0,:,:], grided2.point_latitude['data'][0,:,:], 
              grided2.fields['TH']['data'][0,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes[2].set_title('5 km gridded BARNES2')
    CS = axes[2].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    axes[2].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[2].set_ylim([options['ylim_min'], options['ylim_max']])
    #
    axes[3].pcolormesh(grided3.point_longitude['data'][0,:,:], grided3.point_latitude['data'][0,:,:], 
          grided3.fields['TH']['data'][0,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes[3].set_title('10 km gridded BARNES2')
    CS = axes[3].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    axes[3].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[3].set_ylim([options['ylim_min'], options['ylim_max']])
    #
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[3].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[3].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[3].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    # Add labels:
    labels = ["200 K"] 
    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])
    axes[3].legend(loc='upper left', fontsize=fontize)

    # New figure for resolutions
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                    figsize=[14,12])
    axes.pcolormesh(grided3.point_longitude['data'][0,:,:], grided3.point_latitude['data'][0,:,:], 
      grided3.fields['TH']['data'][0,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes.set_title('10 km grided radar')
    axes.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    axes.set_xlim([options['xlim_min'], options['xlim_max']])
    axes.set_ylim([options['ylim_min'], options['ylim_max_zoom']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    # Add labels:
    labels = ["200 K"] 
    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])
    axes.legend(loc='upper left', fontsize=fontize)
    # Add resolutions? 
    #axes.plot(lon_gmi_inside[100:110], lat_gmi_inside[inds_1][100:110],'ok', markersize=20, markerfacecolor='none')
    plt.scatter(lon_gmi_inside[1030:1040], lat_gmi_inside[1030:1040], marker='o', color='k', s=100)
    print('length of lon_gmi_inside is'+str(lon_gmi_inside.shape))
    plt.close()

    # NEW FIGURE 
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True, figsize=[14,12])
    counter = 0
    for i in range(3):
        for j in range(3):
            axes[i,j].pcolormesh(grided.point_longitude['data'][counter,:,:], grided.point_latitude['data'][counter,:,:], 
                  grided.fields['TH']['data'][counter,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
            counter=counter+1;
            axes[i,j].set_title('horiz. cut at '+str(grided.z['data'][counter]/1e3))
            axes[i,j].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
            axes[i,j].set_xlim([options['xlim_min'], options['xlim_max']])
            axes[i,j].set_ylim([options['ylim_min'], options['ylim_max']])

    # Same as above but plt lowest level y closest to freezing level! 
    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True,
                        figsize=[14,6])
    axes[0].pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:], 
                  grided.fields['TH']['data'][0,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes[0].set_title('Ground Level')
    axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[1].pcolormesh(grided.point_longitude['data'][frezlev,:,:], grided.point_latitude['data'][frezlev,:,:], 
                  grided.fields['TH']['data'][frezlev,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
    axes[1].set_title('Freezing level ('+str(grided.z['data'][frezlev]/1e3)+' km)')
    CS1 = axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
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
    axes[0].plot(lon_gmi_inside[inds_1], lat_gmi_inside[inds_1], 'o', markersize=5, markerfacecolor='black')
    axes[1].plot(lon_gmi_inside[inds_1], lat_gmi_inside[inds_1], 'o', markersize=5, markerfacecolor='black')
    if ii == 1:
        axes[0].plot(lon_gmi_inside[inds_2], lat_gmi_inside[inds_2], 'o', markersize=5, markerfacecolor='darkblue')
        axes[1].plot(lon_gmi_inside[inds_2], lat_gmi_inside[inds_2], 'o', markersize=5, markerfacecolor='darkblue')
    if ii == 2:
        axes[0].plot(lon_gmi_inside[inds_2], lat_gmi_inside[inds_2], 'o', markersize=5, markerfacecolor='darkblue')
        axes[0].plot(lon_gmi_inside[inds_3], lat_gmi_inside[inds_3], 'o', markersize=5, markerfacecolor='darkred')
        axes[1].plot(lon_gmi_inside[inds_2], lat_gmi_inside[inds_2], 'o', markersize=5, markerfacecolor='darkblue')
        axes[1].plot(lon_gmi_inside[inds_3], lat_gmi_inside[inds_3], 'o', markersize=5, markerfacecolor='darkred')

    # Addlabels to icois! 
    dummy = axes[0].plot(np.nan, np.nan, 'o', markersize=20, markerfacecolor='black', label='icoi:'+str(icoi[0]))
    axes[0].legend(dummy, str(icoi[0]))
    if ii == 1:
        dummy = [axes[0].plot([], [], 'o', markersize=20, markerfacecolor=cc)[0] for cc in ['black','darkblue']]
        axes[0].legend(dummy, ['icoi:'+str(icoi[0]), 'icoi:'+str(icoi[1])])
    if ii == 2:
        dummy = [axes[0].plot([], [], 'o', markersize=20, markerfacecolor=cc)[0] for cc in ['black','darkblue','darkred']]
        axes[0].legend(dummy, ['icoi:'+str(icoi[0]), 'icoi:'+str(icoi[1]), 'icoi:'+str(icoi[2])])	

    # Add labels:
    labels = ["200 K"] 
    for i in range(len(labels)):
        CS1.collections[i].set_label(labels[i])
    axes[1].legend(loc='upper left', fontsize=fontize)

    # faltaria agregar radar inside countours! 	
    freezing_lev = np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3) 		
    #---- HID FIGURES! (ojo sin corregir ZH!!!!) 
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    #- Add height field for 4/3 propagation
    radar_height = get_z_from_radar(radar)
    radar = add_field_to_radar_object(radar_height, radar, field_name = 'height')    
    iso0 = np.ma.mean(radar.fields['height']['data'][np.where(np.abs(radar.fields['sounding_temperature']['data']) < 0)])
    radar.fields['height_over_iso0'] = deepcopy(radar.fields['height'])
    radar.fields['height_over_iso0']['data'] -= iso0   
    dzh_  = radar.fields['TH']['data']
    dzv_  = radar.fields['TV']['data']
    drho_ = radar.fields['RHOHV']['data']
    dkdp_ = radar.fields['KDP']['data']
    # Filters
    dzh_[np.where(drho_<0.7)] = np.nan	
    dzv_[np.where(drho_<0.7)] = np.nan	
    drho_[np.where(drho_<0.7)] = np.nan	
    dkdp_[np.where(drho_<0.7)] = np.nan	
    scores          = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=(dzh_-dzv_) - opts['ZDRoffset'], rho=drho_, kdp=dkdp_, 
                                             use_temp=True, band='C', T=radar_T)
    HID             = np.argmax(scores, axis=0) + 1

    #--- need to add tempertaure to grided data ... 
    radar_z = grided.point_z['data']
    shape   = np.shape(radar_z)
    rad_T1d = np.interp(radar_z.ravel(), alt_ref, tfield_ref)   # interpolate_sounding_to_radar(snd_T, snd_z, radar)
    radargrid_TT = np.reshape(rad_T1d, shape)
    grided = add_field_to_radar_object(np.reshape(rad_T1d, shape), grided, field_name='sounding_temperature')      
    #- Add height field for 4/3 propagation
    grided = add_field_to_radar_object( grided.point_z['data'], grided, field_name = 'height')    
    iso0 = np.ma.mean(grided.fields['height']['data'][np.where(np.abs(grided.fields['sounding_temperature']['data']) < 0)])
    grided.fields['height_over_iso0'] = deepcopy(grided.fields['height'])
    grided.fields['height_over_iso0']['data'] -= iso0 
    # Filters
    dzh_grid  = grided.fields['TH']['data']
    dzv_grid  = grided.fields['TV']['data']
    drho_grid = grided.fields['RHOHV']['data']
    dkdp_grid = grided.fields['KDP']['data']
    dzh_grid[np.where(drho_grid<0.7)] = np.nan	
    dzv_grid[np.where(drho_grid<0.7)] = np.nan	
    drho_grid[np.where(drho_grid<0.7)] = np.nan	
    dkdp_grid[np.where(drho_grid<0.7)] = np.nan	
    scores          = csu_fhc.csu_fhc_summer(dz=dzh_grid, zdr=(dzh_grid-dzv_grid) - opts['ZDRoffset'], rho=drho_grid, kdp=dkdp_grid, 
                                            use_temp=True, band='C', T=radargrid_TT)
    GRIDDED_HID = np.argmax(scores, axis=0) + 1 
    print(GRIDDED_HID.shape)
    print(grided.point_latitude['data'].shape)

    #---- plot hid ppi  
    hid_colors = ['MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)

    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True,
                        figsize=[14,4])
    pcm1 = axes[0].pcolormesh(lons, lats, HID[start_index:end_index], cmap=cmaphid, vmin=1.8, vmax=10.4)
    axes[0].set_title('HID radar nlev 0 PPI')
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    pcm1 = axes[1].pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:], GRIDDED_HID[0,:,:], cmap=cmaphid, vmin=1.8, vmax=10.4)
    axes[1].set_title('HID GRIDDED 0 km')
    axes[1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);		
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    pcm1 = axes[2].pcolormesh(grided.point_longitude['data'][frezlev,:,:], grided.point_latitude['data'][frezlev,:,:], GRIDDED_HID[frezlev,:,:], cmap=cmaphid, vmin=1.8, vmax=10.4)
    axes[2].set_title(r'HID GRIDDED 0$^o$'+str(round(grided.z['data'][frezlev]/1e3,2))+' km)')
    axes[2].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[2].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[2].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);			
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)

    p1 = axes[0].get_position().get_points().flatten()
    p2 = axes[1].get_position().get_points().flatten();
    p3 = axes[2].get_position().get_points().flatten(); 
    ax_cbar = fig.add_axes([p3[0]+(p3[0]-p2[0])+0.04, 0.1, 0.02, 0.8])   #ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
    cbar = fig.colorbar(pcm1, cax=ax_cbar, shrink=0.9, label='CSU HID')
    cbar = adjust_fhc_colorbar_for_pyart(cbar)
    cbar.cmap.set_under('white')

    if ii == 0:
        return grided, frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1],tb_s1_gmi_inside[inds_1,:], inds_RN1, inds_RB1
    if ii == 1:
        return grided, frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1],tb_s1_gmi_inside[inds_1,:], inds_RN1, inds_RB1, lon_gmi_inside[inds_2], lat_gmi_inside[inds_2],tb_s1_gmi_inside[inds_2,:], inds_RN2, inds_RB2
    if ii == 2:
        return grided, frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1],tb_s1_gmi_inside[inds_1,:], inds_RN1, inds_RB1, lon_gmi_inside[inds_2], lat_gmi_inside[inds_2],tb_s1_gmi_inside[inds_2,:], inds_RN2, inds_RB2, lon_gmi_inside[inds_3], lat_gmi_inside[inds_3],tb_s1_gmi_inside[inds_3,:], inds_RN3, inds_RB3
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
    def return_altitude(ele, az, rng):
		
    	theta_e = ele * np.pi / 180.0  # elevation angle in radians.
    	theta_a = az * np.pi / 180.0  # azimuth angle in radians.
    	R = 6371.0 * 1000.0 * 4.0 / 3.0  # effective radius of earth in meters.
    	r = rng * 1000.0  # distances to gates in meters.
    	z = (r ** 2 + R ** 2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    	s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    	x = s * np.sin(theta_a)
    	y = s * np.cos(theta_a)

    return  z

# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
def plot_sweep0(radar, opts, fname):  

	# read file
	f = h5py.File( fname, 'r')
	tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
	lon_gmi = f[u'/S1/Longitude'][:,:] 
	lat_gmi = f[u'/S1/Latitude'][:,:]
	tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
	lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
	lat_s2_gmi = f[u'/S2/Latitude'][:,:]
	f.close()
	# PCT
	PCT10, PCT19, PCT37, PCT89 = calc_PCTs(tb_s1_gmi) 
	#
	nlev = 0 
	start_index = radar.sweep_start_ray_index['data'][nlev]
	end_index   = radar.sweep_end_ray_index['data'][nlev]
	lats     = radar.gate_latitude['data'][start_index:end_index]
	lons     = radar.gate_longitude['data'][start_index:end_index]
	radarTH  = radar.fields['TH']['data'][start_index:end_index]
	#-------------------------- ZH y contornos y RHO
	fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=[21,6])
	[units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
	pcm1 = axes[0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
	plt.colorbar(pcm1, ax=axes[0])
	axes[0].set_xlim([opts['xlim_min'], opts['xlim_max']])	
	axes[0].set_ylim(opts['ylim_min'], opts['ylim_max'])
	axes[0].set_title('ZH (w/ 45dBZ contour)')
	axes[0].contour(lons[:], lats[:], radar.fields['TH']['data'][start_index:end_index][:], [45], colors=(['k']), linewidths=1.5);	
	#-------------------------- RHOHV
	[units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
	pcm1 = axes[1].pcolormesh(lons, lats, radar.fields['RHOHV']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
	plt.colorbar(pcm1, ax=axes[1])
	axes[1].set_xlim([opts['xlim_min'], opts['xlim_max']])	
	axes[1].set_ylim(opts['ylim_min'], opts['ylim_max'])
	axes[1].set_title('RHOHV (w/ 45dBZ contour)')
	axes[1].contour(lons[:], lats[:], radar.fields['TH']['data'][start_index:end_index][:], [45], colors=(['k']), linewidths=1.5);	
	axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['m']), linewidths=1.5);
	axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
	#-------------------------- DOPPLER FOR OVERVIEW - SC
	VEL = radar.fields['VRAD']['data'][start_index:end_index]
	vel_texture = pyart.retrieve.calculate_velocity_texture(radar, vel_field='VRAD', nyq=39.9)
	radar.add_field('velocity_texture', vel_texture, replace_existing=True)
	velocity_dealiased = pyart.correct.dealias_region_based(radar, vel_field='VRAD', nyquist_vel=39.9,centered=True)
	radar.add_field('corrected_velocity', velocity_dealiased, replace_existing=True)
	VEL_cor = radar.fields['corrected_velocity']['data'][start_index:end_index]
	[units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('doppler')
	pcm1 = axes[2].pcolormesh(lons, lats, VEL_cor, cmap=cmap, vmin=vmin, vmax=vmax)
	cbar = plt.colorbar(pcm1, ax=axes[2], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
	cbar.cmap.set_under(under)
	cbar.cmap.set_over(over)
	axes[2].set_xlim([opts['xlim_min'], opts['xlim_max']])	
	axes[2].set_ylim(opts['ylim_min'], opts['ylim_max'])
	axes[2].set_title('Vr corrected (w/ 45dBZ contour)')
	axes[2].contour(lons[:], lats[:], radar.fields['TH']['data'][start_index:end_index][:], [45], colors=(['k']), linewidths=1.5);	
	axes[2].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);

	return

# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 

def queesesto(): 

    nlev = 0 
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    azimuths    = radar.azimuth['data'][start_index:end_index]

    fig, axes = plt.subplots(nrows=4, ncols=3, constrained_layout=True, figsize=[13,12])

    for iz in range(len(azi_oi)):
        target_azimuth = azimuths[azi_oi[iz]]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()	
        grid_lon   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_lon[:]   = np.nan
        grid_lat   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_lat[:]   = np.nan
        grid_THTH  = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_THTH[:]  = np.nan
        grid_TVTV  = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_TVTV[:]  = np.nan
        grid_alt   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_alt[:]   = np.nan
        grid_range = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_range[:] = np.nan
        grid_RHO   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_RHO[:]   = np.nan
        grid_HID   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_HID[:]   = np.nan
        grid_KDP   = np.zeros((gridded_radar.fields['KDP']['data'].shape[0], lons[filas,:].shape[2])); grid_KDP[:]   = np.nan

        # need to find x/y pair for each gate at the surface 
        for i in range(lons[filas,:].shape[2]):	
            # First, find the index of the grid point nearest a specific lat/lon.   
            abslat = np.abs(gridded_radar.point_latitude['data'][0,:,:]  - lats[filas,i])
            abslon = np.abs(gridded_radar.point_longitude['data'][0,:,:] - lons[filas,i])
            c = np.maximum(abslon, abslat)
            ([xloc], [yloc]) = np.where(c == np.min(c))	
            grid_lon[:,i]   = gridded_radar.point_longitude['data'][:,xloc,yloc]
            grid_lat[:,i]   = gridded_radar.point_latitude['data'][:,xloc,yloc]
            grid_TVTV[:,i]  = gridded_radar.fields['TV']['data'][:,xloc,yloc]
            grid_THTH[:,i]  = gridded_radar.fields['TH']['data'][:,xloc,yloc]
            grid_RHO[:,i]   = gridded_radar.fields['RHOHV']['data'][:,xloc,yloc]
            grid_alt[:,i]   = gridded_radar.z['data'][:]
            x               = gridded_radar.point_x['data'][:,xloc,yloc]
            y               = gridded_radar.point_y['data'][:,xloc,yloc]
            z               = gridded_radar.point_z['data'][:,xloc,yloc]
            grid_range[:,i] = ( x**2 + y**2 + z**2 ) ** 0.5
            grid_KDP[:,i]   = gridded_radar.fields['KDP']['data'][:,xloc,yloc]


        # Filters
        grid_TVTV[np.where(grid_RHO<0.7)] = np.nan	
        grid_THTH[np.where(grid_RHO<0.7)] = np.nan	
        grid_RHO[np.where(grid_RHO<0.7)] = np.nan	
        grid_KDP[np.where(grid_RHO<0.7)] = np.nan	
    
        #--- need to add tempertaure to grided data ... 
        radar_z = gridded_radar.point_z['data']
        shape   = np.shape(radar_z)
        rad_T1d = np.interp(radar_z.ravel(), alt_ref, tfield_ref)   # interpolate_sounding_to_radar(snd_T, snd_z, radar)
        radargrid_TT = np.reshape(rad_T1d, shape)
        gridded_radar = add_field_to_radar_object(np.reshape(rad_T1d, shape), gridded_radar, field_name='sounding_temperature')      
        #- Add height field for 4/3 propagation
        gridded_radar = add_field_to_radar_object( gridded_radar.point_z['data'], gridded_radar, field_name = 'height')    	
        iso0 = np.ma.mean(gridded_radar.fields['height']['data'][np.where(np.abs(gridded_radar.fields['sounding_temperature']['data']) < 0)])
        gridded_radar.fields['height_over_iso0'] = deepcopy(gridded_radar.fields['height'])
        gridded_radar.fields['height_over_iso0']['data'] -= iso0 
        #
        for i in range(lons[filas,:].shape[2]):	
            scores          = csu_fhc.csu_fhc_summer(dz=grid_THTH[:,i], zdr=(grid_TVTV[:,i]-grid_THTH[:,i]) - opts['ZDRoffset'], 
							 rho=grid_RHO[:,i], kdp=grid_KDP[:,i], 
                                            use_temp=True, band='C', T=radargrid_TT)
            grid_HID[:,i] = np.argmax(scores, axis=0) + 1 

        grid_HID[np.where(grid_RHO<0.7)] = np.nan

        #---- plot hid ppi  
        hid_colors = ['MediumBlue', 'DarkOrange', 'LightPink',
                'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
        cmaphid = colors.ListedColormap(hid_colors)
        cmaphid.set_bad('white')
        cmaphid.set_under('white')
        cmaphid.set_over('white')
        # Figure
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        im_TH  = axes[0,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_THTH, cmap=cmap, vmin=vmin, vmax=vmax)
        im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, (grid_THTH-grid_TVTV)-opts['ZDRoffset'], cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)
        im_RHO = axes[2,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_RHO, cmap=pyart.graph.cm.RefDiff , vmin=0.7, vmax=1.)
        im_HID = axes[3,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=1.8, vmax=10.4)
        
        axes[0,iz].set_title('coi='+titlecois[iz])
        if iz == 1:
            axes[0,iz].set_xlim([0,xlims_xlims[1]])
            axes[1,iz].set_xlim([0,xlims_xlims[1]])
            axes[2,iz].set_xlim([0,xlims_xlims[1]])
            axes[3,iz].set_xlim([0,xlims_xlims[1]])
        if iz == 2:
            axes[0,iz].set_xlim([0,xlims_xlims[2]])
            axes[1,iz].set_xlim([0,xlims_xlims[2]])
            axes[2,iz].set_xlim([0,xlims_xlims[2]])
            axes[3,iz].set_xlim([0,xlims_xlims[2]])
        if iz == 3:
            axes[0,iz].set_xlim([0,xlims_xlims[2]])
            axes[1,iz].set_xlim([0,xlims_xlims[2]])
            axes[2,iz].set_xlim([0,xlims_xlims[2]])
            axes[3,iz].set_xlim([0,xlims_xlims[2]])
    
        if iz == 0:
            axes[0,0].set_ylabel('Altitude (km)')
            axes[1,0].set_ylabel('Altitude (km)')
            axes[2,0].set_ylabel('Altitude (km)')
            axes[3,0].set_ylabel('Altitude (km)')
            axes[0,iz].set_xlim([0,xlims_xlims[0]])
            axes[1,iz].set_xlim([0,xlims_xlims[0]])
            axes[2,iz].set_xlim([0,xlims_xlims[0]])
            axes[3,iz].set_xlim([0,xlims_xlims[0]])
            axes[3,0].set_xlabel('Range (km)')
        if iz == len(azi_oi)-1: 
            # Add colorbars #ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
            pm1    = axes[0,iz-1].get_position().get_points().flatten()
            p_last = axes[0,iz].get_position().get_points().flatten(); 
    
            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.76, 0.02, 0.2])  
            cbar    = fig.colorbar(im_TH,  cax=ax_cbar, shrink=0.9, label='ZH')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            pm2    = axes[1,iz-1].get_position().get_points().flatten()
            
            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.55, 0.02, 0.2])  
            cbar    = fig.colorbar(im_ZDR, cax=ax_cbar, shrink=0.9, label='ZDR')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            
            pm3   = axes[2,iz-1].get_position().get_points().flatten()
            
            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.28, 0.02, 0.2])  
            cbar    = fig.colorbar(im_RHO, cax=ax_cbar, shrink=0.9, label='RHO')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.03, 0.02, 0.2])  
            cbar    = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            cbar = adjust_fhc_colorbar_for_pyart(cbar)
            cbar.cmap.set_under('white')

            pm2    = axes[3,iz-1].get_position().get_points().flatten()

    return

# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
def figure_transect_gridded(lons, radarTH, filas, gridded):
	# gridLev to check 0 km and Freezing Level
	#-------- what about gridded
    gridLev = 0
    grid_lon = []
    grid_lat = []
    grid_TH  = []	
    grid_TV  = []	
    # need to find x/y pair for each gate? 
    for i in range(lons[filas,:].shape[2]):	
        # First, find the index of the grid point nearest a specific lat/lon.   
        abslat = np.abs(gridded.point_latitude['data'][gridLev,:,:]  - lats[filas,i])
        abslon = np.abs(gridded.point_longitude['data'][gridLev,:,:] - lons[filas,i])
        c = np.maximum(abslon, abslat)
        ([xloc], [yloc]) = np.where(c == np.min(c))	
        grid_lon.append(gridded.point_longitude['data'][gridLev,xloc,yloc])
        grid_lat.append(gridded.point_latitude['data'][gridLev,xloc,yloc])
        grid_TH.append(gridded.fields['TH']['data'][gridLev,xloc,yloc])
        grid_TV.append(gridded.fields['TV']['data'][gridLev,xloc,yloc])

    gridLev = frezlev
    grid_lon_frezlev = []
    grid_lat_frezlev = []
    grid_TH_frezlev  = []	
    grid_TV_frezlev  = []	
    # need to find x/y pair for each gate? 
    for i in range(lons[filas,:].shape[2]):	
        # First, find the index of the grid point nearest a specific lat/lon.   
        abslat = np.abs(gridded.point_latitude['data'][gridLev,:,:]  - lats[filas,i])

        abslon = np.abs(gridded.point_longitude['data'][gridLev,:,:] - lons[filas,i])
        c = np.maximum(abslon, abslat)
        ([xloc], [yloc]) = np.where(c == np.min(c))	
        grid_lon_frezlev.append(gridded.point_longitude['data'][gridLev,xloc,yloc])
        grid_lat_frezlev.append(gridded.point_latitude['data'][gridLev,xloc,yloc])
        grid_TH_frezlev.append(gridded.fields['TH']['data'][gridLev,xloc,yloc])
        grid_TV_frezlev.append(gridded.fields['TV']['data'][gridLev,xloc,yloc])

    # Figure ploting the azimuth
    fig,ax = plt.subplots()
    plt.plot(np.ravel(lons[filas,:]), np.ravel(radarTH[filas, :]),'-k', label='radar nlev=0' )
    plt.plot(grid_lon,  grid_TH,'-r', label='gridded 0 km' )
    plt.plot(grid_lon_frezlev,  grid_TH_frezlev,'-',color='darkred', label='gridded frezlev '+str(np.ndarray.round(np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3),2))+' km' )
    plt.xlim([-65.2, -64.2])
    plt.ylabel('ZH')
    plt.xlabel('Longitude')		     
    plt.legend()
    plt.grid(True) 
    ax2=ax.twinx()
    ax2.plot(np.ravel(grid_lon), np.ravel(radar_gateZ)/1e3, '--', color="black") 
    ax2.plot([grid_lon[0],grid_lon[-1]], [0, 0], '--', color="red")
    ax2.plot([grid_lon[0],grid_lon[-1]], [np.ndarray.round(np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3),2),np.ndarray.round(np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3),2)], '--', color="darkred")
    ax2.set_ylabel("Altitude",color="gray")
    return
# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
def interpolate_sounding_to_radar(snd_T, snd_z, radar):
    """Takes sounding data and interpolates it to every radar gate."""
    radar_z = get_z_from_radar(radar)
    radar_T = None
    shape   = np.shape(radar_z)
    rad_z1d = radar_z.ravel()
    rad_T1d = np.interp(rad_z1d, snd_z, snd_T)
    return np.reshape(rad_T1d, shape), radar_z
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def add_field_to_radar_object(field, radar, field_name='FH', units='unitless', 
                              long_name='Hydrometeor ID', standard_name='Hydrometeor ID'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    if 'TH' in radar.fields.keys():  
        dz_field='TH'
    elif 'DBZH' in radar.fields.keys():
        dz_field='DBZH'   
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def get_z_from_radar(radar):
    """Input radar object, return z from radar (km, 2D)"""
    azimuth_1D = radar.azimuth['data']
    elevation_1D = radar.elevation['data']
    srange_1D = radar.range['data']
    sr_2d, az_2d = np.meshgrid(srange_1D, azimuth_1D)
    el_2d = np.meshgrid(srange_1D, elevation_1D)[1]
    xx, yy, zz = antenna_to_cartesian(sr_2d/1000.0, az_2d, el_2d) # Cartesian coordinates in meters from the radar.

    return zz + radar.altitude['data'][0]	

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  

def adjust_fhc_colorbar_for_pyart(cb):
    
    # HID types:           Species #:  
    # -------------------------------
    # Drizzle                  1    
    # Rain                     2    
    # Ice Crystals             3
    # Aggregates               4
    # Wet Snow                 5
    # Vertical Ice             6
    # Low-Density Graupel      7
    # High-Density Graupel     8
    # Hail                     9
    # Big Drops                10
    
    
    cb.set_ticks(np.arange(2.4, 10, 0.9))
    cb.ax.set_yticklabels(['Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def plot_gmi(fname, options, radardat_dir, radar_file, lon_pfs, lat_pfs,reference_satLOS):

    # coi       len(contorno89.collections[0].get_paths()) = cantidad de contornos.
    #           me interesa tomar los vertices del contorno de interes. 
    #           Entonces abajo hago este loop   
    #               for ii in range(len(coi)):
    #                   i = coi[ii]
    #                   for ik in range(len(contorno89.collections[0].get_paths()[i].vertices)):  
    # Como saber cual es el de interes? lon_pfs y lat_pfs estan adentro de ese contorn. entonces podria
    # borrar coi del input y que loopee por todos ... 
    plt.matplotlib.rc('font', family='serif', size = 20)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['font.serif'] = ['Helvetica']

    radar = pyart.io.read(radardat_dir+radar_file) 
    reflectivity_name = 'TH'   

    user = platform.system()
    if   user == 'Linux':
        home_dir = '/home/victoria.galligani/'  
    elif user == 'Darwin':
        home_dir = '/Users/victoria.galligani'

    # Shapefiles for cartopy 
    geo_reg_shp = home_dir+'Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
    geo_reg = shpreader.Reader(geo_reg_shp)

    countries = shpreader.Reader(home_dir+'Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',
        edgecolor='black')
    rivers = cfeature.NaturalEarthFeature(
        category='physical',
        name='rivers_lake_centerlines',
        scale='10m',
        facecolor='none',
        edgecolor='black')


    cmaps = GMI_colormap() 

    # read file
    f = h5py.File( fname, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()

    # keep domain of interest only by keeping those where the center nadir obs is inside domain
    inside_s1   = np.logical_and(np.logical_and(lon_gmi >= options['xlim_min'], lon_gmi <=  options['xlim_max']), 
                              np.logical_and(lat_gmi >= options['ylim_min'], lat_gmi <= options['ylim_max']))
    inside_s2   = np.logical_and(np.logical_and(lon_s2_gmi >= options['xlim_min'], lon_s2_gmi <=  options['xlim_max']), 
                                         np.logical_and(lat_s2_gmi >= options['ylim_min'], lat_s2_gmi <= options['ylim_max']))    

    lon_gmi_inside   =  lon_gmi[inside_s1] 
    lat_gmi_inside   =  lat_gmi[inside_s1] 
    lon_gmi2_inside  =  lon_gmi[inside_s2]
    lat_gmi2_inside  =  lat_gmi[inside_s2]
    tb_s1_gmi_inside =  tb_s1_gmi[inside_s1,:]
    tb_s2_gmi_inside =  tb_s2_gmi[inside_s2,:]

    #Esto si el centro no corre por el medio es un problema! Normalmente uso 110
    tb_s1_gmi[np.where(lon_gmi[:,reference_satLOS] >=  options['xlim_max']+1),:,:] = np.nan
    tb_s1_gmi[np.where(lon_gmi[:,reference_satLOS] <=  options['xlim_min']-1),:,:] = np.nan
    tb_s1_gmi[np.where(lat_gmi[:,reference_satLOS] >=  options['ylim_max']+1),:,:] = np.nan
    tb_s1_gmi[np.where(lat_gmi[:,reference_satLOS] <=  options['ylim_min']-1),:,:] = np.nan

    # CALCULATE PCTs
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(tb_s1_gmi) 

    fig = plt.figure(figsize=(24,12)) 
    gs1 = gridspec.GridSpec(1, 3)

    # BT(37)       
    ax1 = plt.subplot(gs1[0,0], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax1.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax1.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax1.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax1.add_feature(states_provinces,linewidth=0.4)
    ax1.add_feature(rivers)
    ax1.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_gmi[:], lat_gmi[:], 
           c=tb_s1_gmi[:,:,5], s=40, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 37 GHz')
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,1), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, '-k', linewidth=1)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=10, markerfacecolor="none",
            markeredgecolor='magenta', markeredgewidth=1.5) 
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('m'), linewidths=1.5);
    plt.plot(np.nan, np.nan, '-m', label='PCT89 200 K ')
    plt.legend(loc='upper left')


    # BT(89)              
    ax2 = plt.subplot(gs1[0,1], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax2.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax2.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax2.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax2.add_feature(states_provinces,linewidth=0.4)
    ax2.add_feature(rivers)
    ax2.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_gmi[inside_s1], lat_gmi[inside_s1], 
           c=tb_s1_gmi[inside_s1,7], s=40, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 89 GHz')
    ax2.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax2.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,1), crs=crs_latlon)
    ax2.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=10, markerfacecolor="none",
            markeredgecolor='magenta', markeredgewidth=1.5)   
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    contorno89 = plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('m'), linewidths=1.5);            

    # So, interested in paths: 1, 2, 3
    # Get vertices of these polygon type shapes
    for ii in range(len(contorno89.collections[0].get_paths())):                 #range(len(coi)):
        #i = coi[ii]
        i = ii
        X1 = []; Y1 = []; vertices = []
        for ik in range(len(contorno89.collections[0].get_paths()[i].vertices)): 
            X1.append(contorno89.collections[0].get_paths()[i].vertices[ik][0])
            Y1.append(contorno89.collections[0].get_paths()[i].vertices[ik][1])
            vertices.append([contorno89.collections[0].get_paths()[i].vertices[ik][0], 
                                        contorno89.collections[0].get_paths()[i].vertices[ik][1]])
        convexhull = ConvexHull(vertices)
        array_points = np.array(vertices)
        ##--- Run hull_paths and intersec
        hull_path   = Path( array_points[convexhull.vertices] )
        datapts = np.column_stack((lon_gmi_inside,lat_gmi_inside))
        inds = hull_path.contains_points(datapts)           
        # if lon_pfs y lat_pfs inside, continue 
        if hull_path.contains_points(np.column_stack((lon_pfs,lat_pfs))): 
            print('========= contour ii contains lat/lon pfs: '+str(ii))
            coi = ii 

    # BT(166)           
    ax3 = plt.subplot(gs1[0,2], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax3.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax3.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax3.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax3.add_feature(states_provinces,linewidth=0.4)
    ax3.add_feature(rivers)
    ax3.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_s2_gmi[inside_s2], lat_s2_gmi[inside_s2], 
           c=tb_s2_gmi[inside_s2,0], s=40, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 166 GHz')
    ax3.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax3.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,1), crs=crs_latlon)
    ax3.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=10, markerfacecolor="none",
            markeredgecolor='magenta', markeredgewidth=1.5) 
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    contorno89 = plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('m'), linewidths=1.5);
    #ax3.clabel(CONTORNO, CONTORNO.levels, inline=True, fmt=fmt, fontsize=10)

    # So, interested in paths: 1, 2, 3
    # Get vertices of these polygon type shapes
    # son los mismos contornos pq usamos TB89 200 K sobre lat_s1 y lon_s1 *por eso comento abajo
    ##for ii in range(len(coi)):
    ##    i = coi[ii]
    ##    X1 = []; Y1 = []; vertices = []
    ##    for ik in range(len(contorno89.collections[0].get_paths()[i].vertices)): 
    ##        X1.append(contorno89.collections[0].get_paths()[i].vertices[ik][0])
    ##        Y1.append(contorno89.collections[0].get_paths()[i].vertices[ik][1])
    ##        vertices.append([contorno89.collections[0].get_paths()[i].vertices[ik][0], 
    ##                                    contorno89.collections[0].get_paths()[i].vertices[ik][1]])
    ##    convexhull = ConvexHull(vertices)
    ##    array_points = np.array(vertices)
    ##    ##--- Run hull_paths and intersec
    ##    hull_path   = Path( array_points[convexhull.vertices] )
    ##    datapts = np.column_stack((lon_gmi_inside,lat_gmi_inside))
    ##    inds = hull_path.contains_points(datapts)

    p1 = ax1.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[0], 0.25, p2[2]-p1[0], 0.03])   # [left, bottom, width, height] or Bbox 
    cbar = fig.colorbar(im, cax=ax_cbar, shrink=0.8, ticks=np.arange(50,300,50), extend='both', orientation="horizontal", label='TBV (K)')   

    # antes hacia esto a mano: 
    # contour n1 (OJO ESTOS A MANO!) pero ahora no solo me interesa el que tiene el PF sino 
    # otros posibles para comparar pq no se detectaron como P_hail ... estos podrian ser a mano entonces ... 
    # tirar en dos partes. primero plot_gmi que me tira la figura con TODOS los contornos y luego la otra con los que
    # me interesan ... 


    return
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def GMI_colormap(): 
    
    _turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],
                        [0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],
                        [0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],
                        [0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],
                        [0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],
                        [0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],
                        [0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],
                        [0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],
                        [0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],
                        [0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],
                        [0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],
                        [0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],
                        [0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],
                        [0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],
                        [0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],
                        [0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],
                        [0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],
                        [0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],
                        [0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],
                        [0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],
                        [0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],
                        [0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],
                        [0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],
                        [0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],
                        [0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],
                        [0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],
                        [0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],
                        [0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],
                        [0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],
                        [0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],
                        [0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],
                        [0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],
                        [0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],
                        [0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],
                        [0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],
                        [0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],
                        [0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],
                        [0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],
                        [0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],
                        [0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],
                        [0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],
                        [0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],
                        [0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],
                        [0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],
                        [0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],
                        [0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],
                        [0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],
                        [0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],
                        [0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],
                        [0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],
                        [0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],
                        [0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],
                        [0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],
                        [0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],
                        [0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],
                        [0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],
                        [0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],
                        [0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],
                        [0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],
                        [0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],
                        [0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],
                        [0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],
                        [0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],
                        [0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],
                        [0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],
                        [0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],
                        [0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],
                        [0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],
                        [0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],
                        [0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],
                        [0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],
                        [0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],
                        [0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],
                        [0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],
                        [0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],
                        [0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],
                        [0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],
                        [0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],
                        [0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],
                        [0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],
                        [0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],
                        [0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],
                        [0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],
                        [0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],
                        [0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],
                        [0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],
                        [0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],
                        [0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],
                        [0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],
                        [0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],
                        [0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],
                        [0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],
                        [0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],
                        [0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],
                        [0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],
                        [0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],
                        [0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],
                        [0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],
                        [0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],
                        [0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],
                        [0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],
                        [0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],
                        [0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],
                        [0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],
                        [0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],
                        [0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],
                        [0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],
                        [0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],
                        [0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],
                        [0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],
                        [0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],
                        [0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],
                        [0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],
                        [0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],
                        [0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],
                        [0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],
                        [0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],
                        [0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],
                        [0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],
                        [0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],
                        [0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],
                        [0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],
                        [0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],
                        [0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],
                        [0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],
                        [0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],
                        [0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],
                        [0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]

    _turbo_colormap_data.reverse()
    cmaps = {}
    cmaps['turbo_r'] = ListedColormap(_turbo_colormap_data, name='turbo_r')
    
    return cmaps

#------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------

def check_transec(radar, test_transect, lon_pf, lat_pf):       
  nlev  = 0  
  fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[13,12])
  [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
  start_index = radar.sweep_start_ray_index['data'][nlev]
  end_index   = radar.sweep_end_ray_index['data'][nlev]
  lats = radar.gate_latitude['data'][start_index:end_index]
  lons = radar.gate_longitude['data'][start_index:end_index]
  if 'TH' in radar.fields.keys():
  	pcm1 = axes.pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
  elif 'reflectivity' in radar.fields.keys():
  	pcm1 = axes.pcolormesh(lons, lats, radar.fields['reflectivity']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
  elif 'DBZHCC' in radar.fields.keys():
  	pcm1 = axes.pcolormesh(lons, lats, radar.fields['DBZHCC']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)	
  elif 'DBZH' in radar.fields.keys():
  	pcm1 = axes.pcolormesh(lons, lats, radar.fields['DBZH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)	
  cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
  cbar.cmap.set_under(under)
  cbar.cmap.set_over(over)
  [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
  axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
  [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
  axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
  [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
  axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
  axes.grid()
  azimuths = radar.azimuth['data'][start_index:end_index]
  target_azimuth = azimuths[test_transect]
  filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
  lon_transect     = lons[filas,:]
  lat_transect     = lats[filas,:]
  plt.plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k')
  plt.title('Transecta Nr:'+ str(test_transect), Fontsize=20)
  for iPF in range(len(lat_pf)):
    plt.plot(lon_pf[iPF], lat_pf[iPF], marker='*', markersize=40, markerfacecolor="None",
            markeredgecolor='black', markeredgewidth=2, label='GMI(PF) center') 

  return 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def make_pseudoRHISfromGrid(gridded_radar, radar, azi_oi, titlecois, xlims_xlims): 

    nlev = 0 
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    azimuths    = radar.azimuth['data'][start_index:end_index]

    fig, axes = plt.subplots(nrows=4, ncols=3, constrained_layout=True, figsize=[13,12])

    for iz in range(len(azi_oi)):
        target_azimuth = azimuths[azi_oi[iz]]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()	
        grid_lon   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_lon[:]   = np.nan
        grid_lat   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_lat[:]   = np.nan
        grid_THTH  = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_THTH[:]  = np.nan
        grid_TVTV  = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_TVTV[:]  = np.nan
        grid_alt   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_alt[:]   = np.nan
        grid_range = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_range[:] = np.nan
        grid_RHO   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_RHO[:]   = np.nan
        grid_HID   = np.zeros((gridded_radar.fields['TH']['data'].shape[0], lons[filas,:].shape[2])); grid_HID[:]   = np.nan
        grid_KDP   = np.zeros((gridded_radar.fields['KDP']['data'].shape[0], lons[filas,:].shape[2])); grid_KDP[:]   = np.nan

        # need to find x/y pair for each gate at the surface 
        for i in range(lons[filas,:].shape[2]):	
            # First, find the index of the grid point nearest a specific lat/lon.   
            abslat = np.abs(gridded_radar.point_latitude['data'][0,:,:]  - lats[filas,i])
            abslon = np.abs(gridded_radar.point_longitude['data'][0,:,:] - lons[filas,i])
            c = np.maximum(abslon, abslat)
            ([xloc], [yloc]) = np.where(c == np.min(c))	
            grid_lon[:,i]   = gridded_radar.point_longitude['data'][:,xloc,yloc]
            grid_lat[:,i]   = gridded_radar.point_latitude['data'][:,xloc,yloc]
            grid_TVTV[:,i]  = gridded_radar.fields['TV']['data'][:,xloc,yloc]
            grid_THTH[:,i]  = gridded_radar.fields['TH']['data'][:,xloc,yloc]
            grid_RHO[:,i]   = gridded_radar.fields['RHOHV']['data'][:,xloc,yloc]
            grid_alt[:,i]   = gridded_radar.z['data'][:]
            x               = gridded_radar.point_x['data'][:,xloc,yloc]
            y               = gridded_radar.point_y['data'][:,xloc,yloc]
            z               = gridded_radar.point_z['data'][:,xloc,yloc]
            grid_range[:,i] = ( x**2 + y**2 + z**2 ) ** 0.5
            grid_KDP[:,i]   = gridded_radar.fields['KDP']['data'][:,xloc,yloc]


        # Filters
        #grid_TVTV[np.where(grid_RHO<0.7)] = np.nan	
        #grid_THTH[np.where(grid_RHO<0.7)] = np.nan	
        #grid_RHO[np.where(grid_RHO<0.7)] = np.nan	
        #grid_KDP[np.where(grid_RHO<0.7)] = np.nan	
    
        #--- need to add tempertaure to grided data ... 
        radar_z = gridded_radar.point_z['data']
        shape   = np.shape(radar_z)
        rad_T1d = np.interp(radar_z.ravel(), alt_ref, tfield_ref)   # interpolate_sounding_to_radar(snd_T, snd_z, radar)
        radargrid_TT = np.reshape(rad_T1d, shape)
        gridded_radar = add_field_to_radar_object(np.reshape(rad_T1d, shape), gridded_radar, field_name='sounding_temperature')      
        #- Add height field for 4/3 propagation
        gridded_radar = add_field_to_radar_object( gridded_radar.point_z['data'], gridded_radar, field_name = 'height')    	
        iso0 = np.ma.mean(gridded_radar.fields['height']['data'][np.where(np.abs(gridded_radar.fields['sounding_temperature']['data']) < 0)])
        gridded_radar.fields['height_over_iso0'] = deepcopy(gridded_radar.fields['height'])
        gridded_radar.fields['height_over_iso0']['data'] -= iso0 
        #
        for i in range(lons[filas,:].shape[2]):	
            scores          = csu_fhc.csu_fhc_summer(dz=grid_THTH[:,i], zdr=(grid_TVTV[:,i]-grid_THTH[:,i]) - opts['ZDRoffset'], 
							 rho=grid_RHO[:,i], kdp=grid_KDP[:,i], 
                                            use_temp=True, band='C', T=radargrid_TT)
            grid_HID[:,i] = np.argmax(scores, axis=0) + 1 

        grid_HID[np.where(grid_RHO<0.7)] = np.nan

        #---- plot hid ppi  
        hid_colors = ['MediumBlue', 'DarkOrange', 'LightPink',
                'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
        cmaphid = colors.ListedColormap(hid_colors)
        cmaphid.set_bad('white')
        cmaphid.set_under('white')
        # Figure
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        im_TH  = axes[0,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_THTH, cmap=cmap, vmin=vmin, vmax=vmax)
        im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, (grid_THTH-grid_TVTV)-opts['ZDRoffset'], cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)
        im_RHO = axes[2,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_RHO, cmap=pyart.graph.cm.RefDiff , vmin=0.7, vmax=1.)
        im_HID = axes[3,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=1.8, vmax=10.4)
        
        axes[0,iz].set_title('coi='+titlecois[iz])
        if iz == 1:
            axes[0,iz].set_xlim([0,xlims_xlims[1]])
            axes[1,iz].set_xlim([0,xlims_xlims[1]])
            axes[2,iz].set_xlim([0,xlims_xlims[1]])
            axes[3,iz].set_xlim([0,xlims_xlims[1]])
        if iz == 2:
            axes[0,iz].set_xlim([0,xlims_xlims[2]])
            axes[1,iz].set_xlim([0,xlims_xlims[2]])
            axes[2,iz].set_xlim([0,xlims_xlims[2]])
            axes[3,iz].set_xlim([0,xlims_xlims[2]])
        if iz == 3:
            axes[0,iz].set_xlim([0,xlims_xlims[2]])
            axes[1,iz].set_xlim([0,xlims_xlims[2]])
            axes[2,iz].set_xlim([0,xlims_xlims[2]])
            axes[3,iz].set_xlim([0,xlims_xlims[2]])
    
        if iz == 0:
            axes[0,0].set_ylabel('Altitude (km)')
            axes[1,0].set_ylabel('Altitude (km)')
            axes[2,0].set_ylabel('Altitude (km)')
            axes[3,0].set_ylabel('Altitude (km)')
            axes[0,iz].set_xlim([0,xlims_xlims[0]])
            axes[1,iz].set_xlim([0,xlims_xlims[0]])
            axes[2,iz].set_xlim([0,xlims_xlims[0]])
            axes[3,iz].set_xlim([0,xlims_xlims[0]])
            axes[3,0].set_xlabel('Range (km)')
        if iz == len(azi_oi)-1: 
            # Add colorbars #ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
            pm1    = axes[0,iz-1].get_position().get_points().flatten()
            p_last = axes[0,iz].get_position().get_points().flatten(); 
    
            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.76, 0.02, 0.2])  
            cbar    = fig.colorbar(im_TH,  cax=ax_cbar, shrink=0.9, label='ZH')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            pm2    = axes[1,iz-1].get_position().get_points().flatten()
            
            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.55, 0.02, 0.2])  
            cbar    = fig.colorbar(im_ZDR, cax=ax_cbar, shrink=0.9, label='ZDR')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            
            pm3   = axes[2,iz-1].get_position().get_points().flatten()
            
            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.28, 0.02, 0.2])  
            cbar    = fig.colorbar(im_RHO, cax=ax_cbar, shrink=0.9, label='RHO')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.03, 0.02, 0.2])  
            cbar    = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            cbar = adjust_fhc_colorbar_for_pyart(cbar)
            cbar.cmap.set_under('white')

            pm2    = axes[3,iz-1].get_position().get_points().flatten()

    return
	
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------	

def check_transec_rma_campos(dat_dir, file_PAR_all, test_transect, campo, nlev):       

  radar = pyart.io.read(dat_dir+file_PAR_all) 
  start_index = radar.sweep_start_ray_index['data'][nlev]
  end_index   = radar.sweep_end_ray_index['data'][nlev]
  lats = radar.gate_latitude['data'][start_index:end_index]
  lons = radar.gate_longitude['data'][start_index:end_index]
  
  fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[20,12]) 
  if 'ZDR' in campo:
    campotoplot = 'ZDR'
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
    if 'ZDR' in radar.fields.keys(): 
        campo_field_plot = radar.fields['ZDR']['data'][start_index:end_index]
    elif 'TH' in radar.fields.keys():
        ZHZH             = radar.fields['TH']['data'][start_index:end_index]
        TV               = radar.fields['TV']['data'][start_index:end_index]     
        campo_field_plot = ZHZH-TV
    elif 'DBZH' in radar.fields.keys():
        ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
        TV         = radar.fields['DBZV']['data'][start_index:end_index]     
        campo_field_plot = ZHZH-TV
  if 'RHO' in campo:
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    campo_field_plot  = radar.fields['RHOHV']['data'][start_index:end_index]  
    campotoplot = r'$\rho_{HV}$'
  RHOFIELD  = radar.fields['RHOHV']['data'][start_index:end_index]  
 
  if 'ZH' in campo:
    campotoplot = 'Zh'
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    if 'TH' in radar.fields.keys(): 
        campo_field_plot = radar.fields['TH']['data'][start_index:end_index]
    elif 'DBZH' in radar.fields.keys():
        campo_field_plot       = radar.fields['DBZH']['data'][start_index:end_index]      
        
  pcm1 = axes.pcolormesh(lons, lats, campo_field_plot, cmap=cmap, vmin=vmin, vmax=vmax)
  cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
  cbar.cmap.set_under(under)
  cbar.cmap.set_over(over)
  [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
  axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
  axes.grid()
  azimuths = radar.azimuth['data'][start_index:end_index]
  target_azimuth = azimuths[test_transect]
  filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
  lon_transect     = lons[filas,:]
  lat_transect     = lats[filas,:]
  plt.plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k')
  plt.title('Transecta Nr:'+ str(test_transect), Fontsize=20)

  gateZ    = radar.gate_z['data'][start_index:end_index]
  gateX    = radar.gate_x['data'][start_index:end_index]
  gateY    = radar.gate_y['data'][start_index:end_index]
  gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
        
  fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[20,12]) 
  axes.plot(np.ravel(gates_range[filas,:])/1000, np.ravel(campo_field_plot[filas,:]),'-k')
  plt.title('Lowest sweep transect of interest', fontsize=14)
  plt.xlabel('Range (km)', fontsize=14)
  plt.ylabel(str(campotoplot), fontsize=14)
  plt.grid(True)
  plt.ylim([-2,5])
  plt.xlim([0, 100])  
  ax2= axes.twinx()  # instantiate a second axes that shares the same x-axis
  ax2.plot(np.ravel(gates_range[filas,:])/1000, np.ravel(RHOFIELD[filas,:]),'-r', label='RHOhv')
  plt.ylabel(r'$RHO_{rv}$')  
  plt.xlabel('Range (km)', fontsize=14)

  return
	
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------	

def plot_rhi_RMA(file, fig_dir, dat_dir, radar_name, xlim_range1, xlim_range2, test_transect, ZDRoffset, freezing_lev): 

    radar = pyart.io.read(dat_dir+file) 
    print(radar.fields.keys())

    #- Radar sweep
    nelev       = 0
    start_index = radar.sweep_start_ray_index['data'][nelev]
    end_index   = radar.sweep_end_ray_index['data'][nelev]
    lats0        = radar.gate_latitude['data'][start_index:end_index]
    lons0        = radar.gate_longitude['data'][start_index:end_index]
    azimuths     = radar.azimuth['data'][start_index:end_index]

    Ze_transect     = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); Ze_transect[:]=np.nan
    ZDR_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); ZDR_transect[:]=np.nan
    PHI_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); PHI_transect[:]=np.nan
    lon_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); lon_transect[:]=np.nan
    lat_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); lat_transect[:]=np.nan
    RHO_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); RHO_transect[:]=np.nan
    approx_altitude = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); approx_altitude[:]=np.nan
    color           = np.full((  len(radar.sweep_start_ray_index['data']), lats0.shape[1], 4), np.nan)
    gate_range      = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); gate_range[:]=np.nan
    HID_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); HID_transect[:]=np.nan
    KDP_transect      = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); KDP_transect[:]=np.nan
    azydims = lats0.shape[1]-1

    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]       
        if radar_name == 'RMA1':
            ZHZH       = radar.fields['TH']['data'][start_index:end_index]
            TV       = radar.fields['TV']['data'][start_index:end_index]
            ZDRZDR      = (ZHZH-TV)-ZDRoffset   
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]       
            ZDRZDR[RHORHO<0.75]=np.nan
            RHORHO[RHORHO<0.75]=np.nan
        elif radar_name == 'RMA5':
            ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
            if 'DBZV' in radar.fields.keys(): 
                TV     = radar.fields['DBZV']['data'][start_index:end_index]     
                ZDRZDR = ZHZH-TV   
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]        
        elif radar_name == 'RMA4':
            if 'TH' in radar.fields.keys():
                ZHZH       = radar.fields['TH']['data'][start_index:end_index]
            elif 'DBZH' in radar.fields.keys():
                ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
            if 'TV' in radar.fields.keys(): 
                TV     = radar.fields['TV']['data'][start_index:end_index]     
                ZDRZDR = (ZHZH-TV)-ZDRoffset   
            elif  'ZDR' in radar.fields.keys(): 
                ZDRZDR     = (radar.fields['ZDR']['data'][start_index:end_index])-ZDRoffset 
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]  
        elif radar_name == 'RMA3':
            if 'TH' in radar.fields.keys():
                ZHZH       = radar.fields['TH']['data'][start_index:end_index]
            elif 'DBZH' in radar.fields.keys():
                ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
            if 'TV' in radar.fields.keys(): 
                TV     = radar.fields['TV']['data'][start_index:end_index]     
                ZDRZDR = (ZHZH-TV)-ZDRoffset   
            elif  'ZDR' in radar.fields.keys(): 
                ZDRZDR     = radar.fields['ZDR']['data'][start_index:end_index]     
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]   
        elif radar_name == 'CSPR2':
            ZHZH       = radar.fields['attenuation_corrected_reflectivity_h']['data'][start_index:end_index]
            ZDRZDR     = radar.fields['attenuation_corrected_differential_reflectivity']['data'][start_index:end_index]
            RHORHO     = radar.fields['copol_correlation_coeff']['data'][start_index:end_index]       
            ZDRZDR[RHORHO<0.75]=np.nan
            RHORHO[RHORHO<0.75]=np.nan
        KDPKDP = radar.fields['KDP']['data'][start_index:end_index]
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        # En verdad buscar azimuth no transecta ... 
        azimuths    = radar.azimuth['data'][start_index:end_index]
        target_azimuth = azimuths[test_transect]  #- target azimuth for nlev=0 test case is 301.5
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect[nlev,:]     = lons[filas,:]
        lat_transect[nlev,:]     = lats[filas,:]
        #
        gateZ    = radar.gate_z['data'][start_index:end_index]
        gateX    = radar.gate_x['data'][start_index:end_index]
        gateY    = radar.gate_y['data'][start_index:end_index]
        gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
        #
        Ze_transect[nlev,:]      = ZHZH[filas,:]
        ZDR_transect[nlev,:]     = ZDRZDR[filas,:]
        RHO_transect[nlev,:]     = RHORHO[filas,:]
        KDP_transect[nlev,:]     = KDPKDP[filas,:]	
        # 
        [xgate, ygate, zgate]   = pyart.core.antenna_to_cartesian(gates_range[filas,:]/1e3, azimuths[filas],radar.get_elevation(nlev)[0]);
        approx_altitude[nlev,:] = zgate/1e3
        gate_range[nlev,:]      = gates_range[filas,:]/1e3;
        #
        scores          = csu_fhc.csu_fhc_summer(dz=Ze_transect[nlev,:], zdr=ZDR_transect[nlev,:], 
                                             rho=RHO_transect[nlev,:], kdp=KDP_transect[nlev,:], 
                                             use_temp=True, band='C', T=radar_T)
        HID_transect[nlev,:]  = np.argmax(scores, axis=0) + 1 
    #---------------------------------------- REFLECTIVITY
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(111)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude, Ze_transect, cmap=colormaps('ref'), vmin=0, vmax=60)

    #- De esta manera me guardo el color con el que rellenar los polygons (scatter plot para sacar el color de cada pixel)
    print(len(radar.sweep_start_ray_index['data']))
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
         fig = plt.figure(figsize=[30,10])
         fig.add_subplot(221)
         sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                 s=1,c=Ze_transect[nlev,:],
                 cmap=colormaps('ref'), vmin=0, vmax=60)
         color[nlev,:,:] = sc.to_rgba(Ze_transect[nlev,:])
         plt.close()

    #- Try polygons
    fig2, axes = plt.subplots(nrows=4,ncols=1,constrained_layout=True,figsize=[8,10])  # 8,4 muy chiquito
    fig1 = plt.figure(figsize=(20,20))
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
         if nlev > 9: continue
         # Create the cone for each elevation IN TERMS OF RANGE. 
         # ===> ACA HABRIA QUE AGREGAR COMO CAMBIA LA ALTURA CON EL RANGE (?)
         ancho_haz_i0    = (np.pi/180*gate_range[nlev,0]/2)
         ancho_haz_i1099 = (np.pi/180*gate_range[nlev,azydims]/2)
         P1 = Polygon([( gate_range[nlev,0],    approx_altitude[nlev,0]-ancho_haz_i0      ),
                   ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]-ancho_haz_i1099),
                   ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]+ancho_haz_i1099),
                   ( gate_range[nlev,0],    approx_altitude[nlev,0]+ancho_haz_i0      )])
         ancho = 50/1E3
         # Location of gates? Every 100m?  
         LS = [Polygon([(gate_range[nlev,x]-ancho, 0),
                   (gate_range[nlev,x]+ancho, 0),
                   (gate_range[nlev,x]+ancho, 50),
                   (gate_range[nlev,x]-ancho, 50)]) for x in np.arange(approx_altitude.shape[1])]
         # Plot
         for i, l in enumerate(LS):
             # Get the polygon of the intersection between the cone and the space 
             #reserved for a specific point
             inter = l.intersection(P1)
             x,y = inter.exterior.xy    
             # Then plot it, filled by the color we want
             axes[0].fill(x, y, color = color[nlev,i,:], )
             x, y = P1.exterior.xy
         axes[0].set_ylim([0, 20])
         axes[0].set_ylabel('Altitude (km)')
         axes[0].grid()
         axes[0].set_xlim((xlim_range1, xlim_range2))
         norm = matplotlib.colors.Normalize(vmin=0.,vmax=60.)
         cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormaps('ref'))
         cax.set_array(Ze_transect)
         cbar_z = fig2.colorbar(cax, ax=axes[0], shrink=1.1, ticks=np.arange(0,60.01,10), label='Zh (dBZ)')
         axes[0].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)
    del mycolorbar, x, y, inter
    #---------------------------------------- ZDR
    N = (5+2)
    cmap_ZDR = discrete_cmap(int(N), 'jet') 
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                ZDR_transect,
                cmap=cmap_ZDR, vmin=-2, vmax=5.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=ZDR_transect[nlev,:],
                cmap=cmap_ZDR, vmin=-2, vmax=5.)
        color[nlev,:,:] = sc.to_rgba(ZDR_transect[nlev,:])
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        if nlev > 9: continue
        # Create the cone for each elevation IN TERMS OF RANGE. 
        ancho_haz_i0    = (np.pi/180*gate_range[nlev,0]/2)
        ancho_haz_i1099 = (np.pi/180*gate_range[nlev,azydims]/2)
        P1 = Polygon([( gate_range[nlev,0],   approx_altitude[nlev,0]-ancho_haz_i0      ),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]-ancho_haz_i1099),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]+ancho_haz_i1099),
                  ( gate_range[nlev,0],       approx_altitude[nlev,0]+ancho_haz_i0      )])
        ancho = 50/1E3
        # Location of gates? Every 100m?  
        LS = [Polygon([(gate_range[nlev,x]-ancho, 0),
                  (gate_range[nlev,x]+ancho, 0),
                  (gate_range[nlev,x]+ancho, 50),
                  (gate_range[nlev,x]-ancho, 50)]) for x in np.arange(approx_altitude.shape[1])]
        # Plot
        #ax1 = plt.gca()
        for i, l in enumerate(LS):
            inter = l.intersection(P1)
            x,y = inter.exterior.xy
            # Then plot it, filled by the color we want
            axes[1].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[1].set_ylim([0, 20])
        axes[1].set_ylabel('Altitude (km)')
        axes[1].grid()
        axes[1].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=-2.,vmax=5.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_ZDR)
        cax.set_array(ZDR_transect)
        cbar_zdr = fig2.colorbar(cax, ax=axes[1], shrink=1.1, ticks=np.arange(-2.,5.01,1.), label='ZDR')     
        axes[1].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, inter
    #---------------------------------------- RHOHV
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                RHO_transect,
                cmap = pyart.graph.cm.RefDiff, vmin=0.7, vmax=1.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=RHO_transect[nlev,:],
                cmap= pyart.graph.cm.RefDiff, vmin=0.7, vmax=1.)
        color[nlev,:,:] = sc.to_rgba(RHO_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        if nlev > 9: continue
        # Create the cone for each elevation IN TERMS OF RANGE. 
        ancho_haz_i0    = (np.pi/180*gate_range[nlev,0]/2)
        ancho_haz_i1099 = (np.pi/180*gate_range[nlev,azydims]/2)
        P1 = Polygon([( gate_range[nlev,0],   approx_altitude[nlev,0]-ancho_haz_i0      ),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]-ancho_haz_i1099),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]+ancho_haz_i1099),
                  ( gate_range[nlev,0],       approx_altitude[nlev,0]+ancho_haz_i0      )])
        ancho = 50/1E3
        # Location of gates? Every 100m?  
        LS = [Polygon([(gate_range[nlev,x]-ancho, 0),
                  (gate_range[nlev,x]+ancho, 0),
                  (gate_range[nlev,x]+ancho, 50),
                  (gate_range[nlev,x]-ancho, 50)]) for x in np.arange(approx_altitude.shape[1])]
        # Plot
        #ax1 = plt.gca()
        for i, l in enumerate(LS):
            inter = l.intersection(P1)
            x,y = inter.exterior.xy
            # Then plot it, filled by the color we want
            axes[2].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[2].set_ylim([0, 20])
        axes[2].set_ylabel('Altitude (km)')
        axes[2].grid()
        axes[2].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0.7,vmax=1.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.RefDiff)
        cax.set_array(RHO_transect)
        cbar_rho = fig2.colorbar(cax, ax=axes[2], shrink=1.1, ticks=np.arange(0.7,1.01,0.1), label=r'$\rho_{hv}$')     
        axes[2].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, inter

    #---------------------------------------- HID
    hid_colors = ['MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)
    cmaphid.set_bad('white')
    cmaphid.set_under('white')

    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                HID_transect,
                cmap = cmaphid, vmin=1.8, vmax=10.4)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=HID_transect[nlev,:],
                cmap = cmaphid, vmin=1.8, vmax=10.4)
        color[nlev,:,:] = sc.to_rgba(HID_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        if nlev > 9: continue
        # Create the cone for each elevation IN TERMS OF RANGE. 
        ancho_haz_i0    = (np.pi/180*gate_range[nlev,0]/2)
        ancho_haz_i1099 = (np.pi/180*gate_range[nlev,azydims]/2)
        P1 = Polygon([( gate_range[nlev,0],   approx_altitude[nlev,0]-ancho_haz_i0      ),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]-ancho_haz_i1099),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]+ancho_haz_i1099),
                  ( gate_range[nlev,0],       approx_altitude[nlev,0]+ancho_haz_i0      )])
        ancho = 50/1E3
        # Location of gates? Every 100m?  
        LS = [Polygon([(gate_range[nlev,x]-ancho, 0),
                  (gate_range[nlev,x]+ancho, 0),
                  (gate_range[nlev,x]+ancho, 50),
                  (gate_range[nlev,x]-ancho, 50)]) for x in np.arange(approx_altitude.shape[1])]
        # Plot
        #ax1 = plt.gca()
        for i, l in enumerate(LS):
            inter = l.intersection(P1)
            x,y = inter.exterior.xy
            # Then plot it, filled by the color we want
            axes[3].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[3].set_ylim([0, 20])
        axes[3].set_ylabel('Altitude (km)')
        axes[3].grid()
        axes[3].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=1.8,vmax=10.4)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmaphid)
        cax.set_array(HID_transect)
        cbar_HID = fig2.colorbar(cax, ax=axes[3], shrink=1.1, label=r'HID')    
        cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	
        axes[3].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)
    del mycolorbar, x, y, inter

    #- savefile
    plt.suptitle(radar_name + ': '+str(file[0:12]) ,fontweight='bold')

    return

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def calc_freezinglevel(era5_dir, era5_file, lat_pf, lon_pf):
	
	ERA5_field = xr.load_dataset(era5_dir+era5_file, engine="cfgrib")	
	elemj      = find_nearest(ERA5_field['latitude'], lat_pf[0])
	elemk      = find_nearest(ERA5_field['longitude'], lon_pf[0])
	tfield_ref = ERA5_field['t'][:,elemj,elemk] - 273 # convert to C
	geoph_ref  = (ERA5_field['z'][:,elemj,elemk])/9.80665
	# Covert to geop. height (https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
	Re         = 6371*1e3
	alt_ref    = (Re*geoph_ref)/(Re-geoph_ref) 
	freezing_lev = np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3) 

	return alt_ref, tfield_ref, freezing_lev

def add_43prop_field(radar):
	
	radar_height = get_z_from_radar(radar)
	radar = add_field_to_radar_object(radar_height, radar, field_name = 'height')    
	iso0 = np.ma.mean(radar.fields['height']['data'][np.where(np.abs(radar.fields['sounding_temperature']['data']) < 0)])
	radar.fields['height_over_iso0'] = deepcopy(radar.fields['height'])
	radar.fields['height_over_iso0']['data'] -= iso0 
	
	return radar 

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
def despeckle_phidp(phi, rho):
    '''
    Elimina pixeles aislados de PhiDP
    '''

    # Unmask data and despeckle
    dphi = np.copy(phi)
    
    # Descartamos pixeles donde RHO es menor que un umbral (e.g., 0.7) o no está definido (e.g., NaN)
    #dphi[np.isnan(rho)] = np.nan
    #rho_thr = 0.93
    #dphi[rho < rho_thr] = np.nan
    
    # Calculamos la textura de RHO (rhot) y descartamos todos los pixeles de PHIDP por encima
    # de un umbral de rhot (e.g., 0.25)
    rhot = wrl.dp.texture(rho)
    rhot_thr = 0.25
    dphi[rhot > rhot_thr] = np.nan
    
    # Eliminamos pixeles aislados rodeados de NaNs
    # https://docs.wradlib.org/en/stable/generated/wradlib.dp.linear_despeckle.html     
    dphi = wrl.dp.linear_despeckle(dphi, ndespeckle=5, copy=False)

    return dphi
		
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------ 
def unfold_phidp(phi, rho, diferencia):
    '''
    Unfolding
    '''
       
    # Dimensión del PPI (elevaciones, azimuth, bins)
    nb = phi.shape[1]
    nr = phi.shape[0]

    phi_cor = np.zeros((nr, nb)) #Asigno cero a la nueva variable phidp corregida
    
    v1 = np.zeros(nb)  #Vector v1
    
    # diferencia = 200 # Valor que toma la diferencia entre uno y otro pixel dentro de un mismo azimuth
    
    for m in range(0, nr):
    
        v1 = phi[m,:]
        v2 = np.zeros(nb)
    
        for l in range(0, nb):
            a = v2[l-1] - v1[l]

            if np.isnan(a):
                v2[l] = v2[l-1]

            elif a > diferencia:
                v2[l] = v1[l] + 360
                if v2[l-1] - v2[l] > 100:  # Esto es por el doble folding cuando es mayor a 700
                    v2[l] = v1[l] + v2[l-1]

            else:
                v2[l] = v1[l]
    
        phi_cor[m,:] = v2
    
    #phi_cor[phi_cor <= 0] = np.nan
    #phi_cor[rho < .7] = np.nan
    
    return phi_cor

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------    
def subtract_sys_phase(phi, sys_phase):

    nb = phi.shape[1] #GATE
    nr = phi.shape[0] #AZYMUTH
    phi_final = np.copy(phi) * 0
    phi_err=np.ones((nr, nb)) * np.nan
    
    try:
        phi_final = phi-sys_phase
    except:
        phi_final = phi_err
    
    return phi_final

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def correct_phidp(phi, rho, zh, sys_phase, diferencia):
    
    ni = phi.shape[0]
    nj = phi.shape[1]

    dphi = phi.copy() 
    for i in range(ni):
        rho_h = rho[i,:]
        zh_h = zh[i,:]
        for j in range(nj):
            if ((rho_h[j]<0.7) & (zh_h[j]<40)):
                phi[i,j]  = np.nan
                #rho[i,j]  = np.nan
		
    dphi = despeckle_phidp(phi, rho)
    uphi_i = unfold_phidp(dphi, rho, diferencia)
    
    # Reemplazo nan por sys_phase para que cuando reste esos puntos queden en cero
    uphi = uphi_i.copy()
    uphi = np.where(np.isnan(uphi), sys_phase, uphi)
    phi_cor = subtract_sys_phase(uphi, sys_phase)
    # phi_cor[rho<0.7] = np.nan
    
    return dphi, uphi_i, phi_cor, uphi

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def calc_KDP(radar):

    dbz     = radar.fields['TH']['data']
    maskphi = radar.fields['corrPHIDP']['data']

    nb = radar.ngates
    nr = radar.nrays
    
    kdp = np.zeros((nr,nb))

    for j in range(0,nr):
        phi_kdp = maskphi[j,:]
    
        for i in range(0,nb): 
            s1=max(0,i-2)
            s2=min(i+2,nb-1)
            r=[x*1. for x in range(s2-s1+1)]
            x=2.*0.5*np.array(r)
            y=phi_kdp[s1:s2+1]
            a=len(y[np.where(y>0)])
            b=np.std(y)
  
            if a==5:# and b<20:
                ajuste=np.polyfit(x,y,1)
                kdp[j,i]=ajuste[0]
            else:
                kdp[j,i]=np.nan

    #Enmascarar los datos invalidos con umbrales y con la mascara
    # kdp[kdp>15]=np.nan
    # kdp[kdp<-10]=np.nan
    kdp_ok = np.ma.masked_invalid(kdp) 
    mask_kdp=np.ma.masked_where(np.isnan(radar.fields['corrPHIDP']['data']),kdp_ok)
    aa=np.ma.filled(mask_kdp,fill_value=np.nan)
    bb = np.ma.masked_invalid(aa)

    radar.add_field_like('TH','NEW_kdp',bb, replace_existing=True)
	
    return radar 


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def check_correct_RHOHV_KDP(radar, options, nlev, azimuth_ray, diff_value):

    # Esto es para todas las elevaciones
    sys_phase  = pyart.correct.phase_proc.det_sys_phase(radar, ncp_lev=0.8, rhohv_lev=0.7, 
							ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP')
    dphi, uphi, corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], radar.fields['TH']['data'], sys_phase, diff_value)
    
	
    #-EJEMPLO de azimuth
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[azimuth_ray]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
    #-figure
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True,figsize=[14,7])
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['RHOHV']['data'][start_index:end_index][filas,:])*100, '-k', label='RHOHV')
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['TH']['data'][start_index:end_index][filas,:]), '-r', label='ZH')
    axes[0].legend()
    axes[1].plot(radar.range['data']/1e3, np.ravel(radar.fields['PHIDP']['data'][start_index:end_index][filas,:]), 'or', label='obs. phidp')
    axes[1].plot(radar.range['data']/1e3, np.ravel(dphi[start_index:end_index][filas,:]), '*b', label='despeckle phidp'); 
    axes[1].plot(radar.range['data']/1e3, np.ravel(uphi[start_index:end_index][filas,:]), color='darkgreen', label='unfolded phidp');
    axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]), color='magenta', label='phidp corrected');
    #plt.plot(radar.range['data']/1e3, np.ravel(uphidp_2[filas,:]), color='k', label='uphidp2');
    plt.legend()


    radar.add_field_like('PHIDP','corrPHIDP', corr_phidp, replace_existing=True)

    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    rhoHV = radar.fields['RHOHV']['data'][start_index:end_index]
    PHIDP = radar.fields['PHIDP']['data'][start_index:end_index]
    KDP   = radar.fields['KDP']['data'][start_index:end_index]
    ZH   = radar.fields['ZH']['data'][start_index:end_index]

    # Y CALCULAR KDP! 
    radar = calc_KDP(radar)
    calculated_KDP  = radar.fields['NEW_kdp']['data'][start_index:end_index]
	

    fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True,
                        figsize=[14,7])
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,0].pcolormesh(lons, lats, rhoHV, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,0].set_title('RHOHV radar nlev 0 PPI')
    axes[0,0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0,0].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    plt.colorbar(pcm1, ax=axes[0,0])
    axes[0,0].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
    pcm1 = axes[0,1].pcolormesh(lons, lats, PHIDP, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,1].set_title('Phidp radar nlev 0 PPI')
    axes[0,1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0,1].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    plt.colorbar(pcm1, ax=axes[0,1])
    axes[0,1].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')

	
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Kdp')
    pcm1 = axes[0,2].pcolormesh(lons, lats, KDP, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,2].set_title('KDP radar nlev 0 PPI')
    axes[0,2].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0,2].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[0,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    plt.colorbar(pcm1, ax=axes[0,2])
    axes[0,2].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes[1,0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[1,0].set_title('ZH nlev 0 PPI')
    axes[1,0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1,0].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    plt.colorbar(pcm1, ax=axes[1,0])
    axes[1,0].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
    pcm1 = axes[1,1].pcolormesh(lons, lats, radar.fields['corrPHIDP']['data'][start_index:end_index], cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[1,1].set_title('CORR Phidp radar nlev 0 PPI')
    axes[1,1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1,1].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    plt.colorbar(pcm1, ax=axes[1,1])
    axes[1,1].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Kdp')
    pcm1 = axes[1,2].pcolormesh(lons, lats, calculated_KDP[start_index:end_index], cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[1,2].set_title('Calc. KDP nlev 0 PPI')
    axes[1,2].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1,2].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    plt.colorbar(pcm1, ax=axes[1,2])
    axes[1,2].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')
	
    fig = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[14,7])
    plt.plot(radar.range['data']/1e3, np.ravel(uphi[filas,:]), color='darkgreen', label='unfolded phidp');



	
    return




#------------------------------------------------------------------------------
# ACA EMPIEZA EL MAIL! 
#------------------------------------------------------------------------------	

def main(): 

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'

   ----> automatizarestas figuras para hacerlas para todos los casos !!! ver tambien twitter como decia nesbitt. y exteneder casos luego! 

ncp = np.zeros_like(radar.fields['TH']['data'])+1
radar.add_field('normalized_coherent_power', {'data':ncp})
phidp, kdp = pyart.correct.phase_proc_lp(radar, 0.0, debug=True, refl_field='TH', phidp_field='PHIDP', 
					 rhv_field='RHOHV', ncp_field='normalized_coherent_power', LP_solver='cylp')


    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO SUPERCELDA: 

def check_this(azimuth_ray): 
	
    options = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5, 
	    'ZDRoffset': 4, 'ylim_max_zoom':-30.5}
    rfile     = 'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile)
    sys_phase  = pyart.correct.phase_proc.det_sys_phase(radar, ncp_lev=0.8, rhohv_lev=0.7, 
							ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP')
    dphi, uphi, corr_phidp, uphi2 = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], radar.fields['TH']['data'], sys_phase, 280)
    
    nlev = 0 
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]

    #-EJEMPLO de azimuth
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[azimuth_ray]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
    #-figure
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True,figsize=[14,7])
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['RHOHV']['data'][start_index:end_index][filas,:])*100, '-k', label='RHOHV')
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['TH']['data'][start_index:end_index][filas,:]), '-r', label='ZH')
    axes[0].legend()
    axes[0].set_xlim([0,120])
    axes[1].plot(radar.range['data']/1e3, np.ravel(radar.fields['PHIDP']['data'][start_index:end_index][filas,:]), 'or', label='obs. phidp')
    axes[1].plot(radar.range['data']/1e3, np.ravel(dphi[start_index:end_index][filas,:]), '*b', label='despeckle phidp'); 
    axes[1].plot(radar.range['data']/1e3, np.ravel(uphi[start_index:end_index][filas,:]), color='darkgreen', label='unfolded phidp');
    axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]), color='magenta', label='phidp corrected');
    #plt.plot(radar.range['data']/1e3, np.ravel(uphidp_2[filas,:]), color='k', label='uphidp2');
    axes[1].set_xlim([0,120])
    plt.legend()


    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    rhoHV = radar.fields['RHOHV']['data'][start_index:end_index]
    PHIDP = radar.fields['PHIDP']['data'][start_index:end_index]
    KDP   = radar.fields['KDP']['data'][start_index:end_index]
    ZH   = radar.fields['TH']['data'][start_index:end_index]

    fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True,
                        figsize=[14,7])
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,0].pcolormesh(lons, lats, rhoHV, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,0].set_title('RHOHV radar nlev 0 PPI')
    axes[0,0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0,0].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    plt.colorbar(pcm1, ax=axes[0,0])
    axes[0,0].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
    pcm1 = axes[0,1].pcolormesh(lons, lats, PHIDP, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,1].set_title('Phidp radar nlev 0 PPI')
    axes[0,1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0,1].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    plt.colorbar(pcm1, ax=axes[0,1])
    axes[0,1].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')


    return 







    
	
    #-EJEMPLO de azimuth
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[azimuth_ray]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
    #-figure
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True,figsize=[14,7])
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['RHOHV']['data'][start_index:end_index][filas,:])*100, '-k', label='RHOHV')
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['TH']['data'][start_index:end_index][filas,:]), '-r', label='ZH')
    axes[0].legend()
    axes[1].plot(radar.range['data']/1e3, np.ravel(radar.fields['PHIDP']['data'][start_index:end_index][filas,:]), 'or', label='obs. phidp')
    axes[1].plot(radar.range['data']/1e3, np.ravel(dphi[start_index:end_index][filas,:]), '*b', label='despeckle phidp'); 
    axes[1].plot(radar.range['data']/1e3, np.ravel(uphi[start_index:end_index][filas,:]), color='darkgreen', label='unfolded phidp');
    axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]), color='magenta', label='phidp corrected');
    #plt.plot(radar.range['data']/1e3, np.ravel(uphidp_2[filas,:]), color='k', label='uphidp2');
    plt.legend()




    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    lon_pfs = [-64.80]
    lat_pfs = [-31.83]
    time_pfs = ['2058']
    phail   = [0.534]
    MIN85PCT = [131.1081]
    MIN37PCT = [207.4052]
    #
    rfile     = 'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
    gfile     = '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5'
    #
    opts = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5, 
	    'ZDRoffset': 4, 'ylim_max_zoom':-30.5}
    era5_file = '20180208_21_RMA1.grib'
    # read radar file:  
    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile)
    # Frezing level:     
    alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel(era5_dir, era5_file, lat_pfs, lon_pfs) 
    #-------------------------- DONT CHANGE
    print('Freezing level at ', np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3), ' km')
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    radar = add_43prop_field(radar) 
    #- output of plot_Zhppi_wGMIcontour depends on the number of icois of interest. Here in this case we have three:
    # OJO for use_freezingLev == 0 es decir ground level! 
    [gridded, frezlev, GMI_lon_COI1, GMI_lat_COI1, GMI_tbs1_COI1, RN_inds_COI1, RB_inds_COI1, 
     GMI_lon_COI2, GMI_lat_COI2, GMI_tbs1_COI2, RN_inds_COI2, RB_inds_COI2,
     GMI_lon_COI3, GMI_lat_COI3, GMI_tbs1_COI3, RN_inds_COI3, RB_inds_COI3] = plot_Zhppi_wGMIcontour(radar, lat_pfs, lon_pfs, 'radar at '+rfile[15:19]+' UTC and PF at '+time_pfs[0]+' UTC', 
                           gmi_dir+gfile, 0, opts, era5_dir+era5_file, icoi=[1,3,4], use_freezingLev=0)
     #-------------------------- 
     # FIGURE scatter plot check
     # scatter plots the tbs y de Zh a ver si esta ok 
     fig = plt.figure(figsize=(12,7)) 
     gs1 = gridspec.GridSpec(1, 2)
     ax1 = plt.subplot(gs1[0,0])
     plt.scatter(GMI_tbs1_COI1[:,5], GMI_tbs1_COI1[:,7], s=20, marker='x', color='k', label='icoi=1')
     plt.scatter(GMI_tbs1_COI2[:,5], GMI_tbs1_COI2[:,7], s=20, marker='x', color='darkblue', label='icoi=3')
     plt.scatter(GMI_tbs1_COI3[:,5], GMI_tbs1_COI3[:,7], s=20, marker='x', color='darkred', label='icoi=4')
     plt.grid(True)
     plt.legend()
     plt.xlim([180,260])
     plt.ylim([130,240])
     plt.xlabel('TBV(37)')
     plt.ylabel('TBV(85)')
     #--- radar?  RN_inds_COI1, RN_inds_COI2, RN_inds_COI3 freezing level? 
     ax1 = plt.subplot(gs1[0,1])
     # axes[0].plot(np.ravel(lons)[inds_RN1], np.ravel(lats)[inds_RN1], 'x', markersize=10, markerfacecolor='black')
     nlev = 0 
     start_index = radar.sweep_start_ray_index['data'][nlev]
     end_index   = radar.sweep_end_ray_index['data'][nlev]
     lats     = radar.gate_latitude['data'][start_index:end_index]
     lons     = radar.gate_longitude['data'][start_index:end_index]
     radarTH  = radar.fields['TH']['data'][start_index:end_index]
     radarZDR = (radar.fields['TH']['data'][start_index:end_index])-(radar.fields['TV']['data'][start_index:end_index])-opts['ZDRoffset']
     plt.scatter(np.ravel(radarTH)[RN_inds_COI1], np.ravel(radarZDR)[RN_inds_COI1]-opts['ZDRoffset'], s=20, marker='x', color='k', label='icoi=1')
     plt.scatter(np.ravel(radarTH)[RN_inds_COI2], np.ravel(radarZDR)[RN_inds_COI2]-opts['ZDRoffset'], s=20, marker='x', color='darkblue', label='icoi=3')
     plt.scatter(np.ravel(radarTH)[RN_inds_COI3], np.ravel(radarZDR)[RN_inds_COI3]-opts['ZDRoffset'], s=20, marker='x', color='darkred', label='icoi=4')
     plt.xlabel('ZH')
     plt.ylabel('ZDR')	
		 #-------------------------- 
     # FIGURE DENSITY PLOTS
     make_densityPlot(radarTH, radarZDR, RN_inds_COI1, RN_inds_COI2, RN_inds_COI3)
     gridedTH  = gridded.fields['TH']['data'][0,:,:]
     gridedZDR = (gridded.fields['TH']['data'][0,:,:]-gridded.fields['TV']['data'][0,:,:]) - opts['ZDRoffset']
     make_densityPlot(gridedTH, gridedZDR, RB_inds_COI1, RB_inds_COI2, RB_inds_COI3)

     [gridded, frezlev, GMI_lon_COI1, GMI_lat_COI1, GMI_tbs1_COI1, RN_inds_COI1, RB_Inds_COI1, 
     GMI_lon_COI2, GMI_lat_COI2, GMI_tbs1_COI2, RN_inds_COI2, RB_Inds_COI12,
     GMI_lon_COI3, GMI_lat_COI3, GMI_tbs1_COI3, RN_inds_COI3, RB_Inds_COI3] = plot_Zhppi_wGMIcontour(radar, lat_pfs, lon_pfs, 'radar at '+rfile[15:19]+' UTC and PF at '+time_pfs[0]+' UTC', 
                           gmi_dir+gfile, 0, opts, era5_dir+era5_file, icoi=[1,3,4], use_freezingLev=1)
     gridedTH  = gridded.fields['TH']['data'][frezlev,:,:]
     gridedZDR = (gridded.fields['TH']['data'][frezlev,:,:]-gridded.fields['TV']['data'][frezlev,:,:]) - opts['ZDRoffset']
     make_densityPlot(gridedTH, gridedZDR, RB_inds_COI1, RB_inds_COI2, RB_inds_COI3)  
		#-------------------------- 
    #DIFRFERENCE BETWEEN GROUND LEVEL NATIVE GRID AND GRIDDED LEVEL 0 ? 
    azimuth_ray = 220 
    #-------- what about nlev 0 
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[azimuth_ray]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
    radar_gateZ = []
    for i in range(np.array(radar.range['data']).shape[0]):
    	radar_gateZ.append(return_altitude(radar.elevation['data'][start_index:end_index][nlev], target_azimuth, 
					   np.array(radar.range['data'])[i]/1e3))
    
		#- Figure transect as a function of range: 
		figure_transect_gridded(lons, radarTH, filas, gridded)

    # FIGURE pseudo-RHIs 
    make_pseudoRHISfromGrid(gridded, radar, [356,220,192],['1','3[Phail = 0.534]','4'], [60, 100, 140])
    	
	
	grided_5KM = pyart.map.grid_from_radars(radar, grid_shape=(20, 94, 94), 
                                   grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                   roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')
	make_pseudoRHISfromGrid(grided_5KM, radar, [356,220,192],['1','3[Phail = 0.534]','4'], [60, 100, 140])

	
	plot_rhi_RMA(rfile, '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures', 
		     '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/',
                 'RMA1', 0, 100, 220, 4 , 3)
		
		
		
    #-------------------------- HID ON GRIDDED ON NATIVE? 
    # hacer contorno de rhohv e el ZDR con zom sobre celdas sur? \
    #
    # read file
    f = h5py.File( gmi_dir+gfile, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 

	
	
		
	#-------------------------- ZH y contornos y RHO
	fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=[14,6])
		
	[units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
	pcm1 = axes[0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
	plt.colorbar(pcm1, ax=axes[0])
	axes[0].set_xlim([-65, -64])	
	axes[0].set_ylim([-33, -31.5])
	axes[0].set_title('ZH (w/ 45dBZ contour)')
	axes[0].contour(lons[:], lats[:], radar.fields['TH']['data'][start_index:end_index][:], [45], colors=(['k']), linewidths=1.5);	
	
	[units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
	pcm1 = axes[1].pcolormesh(lons, lats, radar.fields['RHOHV']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
	plt.colorbar(pcm1, ax=axes[1])
	axes[1].set_xlim([-65, -64])	
	axes[1].set_ylim([-33, -31.5])	
	axes[1].set_title('RHOHV (w/ 45dBZ contour)')
	axes[1].contour(lons[:], lats[:], radar.fields['TH']['data'][start_index:end_index][:], [45], colors=(['k']), linewidths=1.5);	
	axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
	axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);


	
	#-------------------------- DOPPLER FOR OVERVIEW - SC
	VEL = radar.fields['VRAD']['data'][start_index:end_index]
	vel_texture = pyart.retrieve.calculate_velocity_texture(radar, vel_field='VRAD', nyq=39.9)
	radar.add_field('velocity_texture', vel_texture, replace_existing=True)
	velocity_dealiased = pyart.correct.dealias_region_based(radar, vel_field='VRAD', nyquist_vel=39.9,centered=True)
	radar.add_field('corrected_velocity', velocity_dealiased, replace_existing=True)
	VEL_cor = radar.fields['corrected_velocity']['data'][start_index:end_index]
	
	fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=[14,6])
	[units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('doppler')
	pcm1 = axes[0].pcolormesh(lons, lats, VEL_cor, cmap=cmap, vmin=vmin, vmax=vmax)
	cbar = plt.colorbar(pcm1, ax=axes[0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
	cbar.cmap.set_under(under)
	cbar.cmap.set_over(over)
	axes[0].set_xlim([-65, -64])	
	axes[0].set_ylim([-33, -31.5])	
	axes[0].set_title('Vr corrected (w/ 45dBZ contour)')
	axes[0].contour(lons[:], lats[:], radar.fields['TH']['data'][start_index:end_index][:], [45], colors=(['k']), linewidths=1.5);	

		
		
		
    Nr pixels, mean values ? std. barplot, hid 
    #-------------------------- 
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 

    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO MSC RMA3 - 20190305: P(hail) = 0.737 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
	lat_pfs  = [-25.95] 
	lon_pfs  = [-60.57]
	time_pfs = ['1252'] 
	phail    = [0.737]
	MIN85PCT = [75.0826]
	MIN37PCT = [164.4755] 
    	#
	rfile     = 'cfrad.20190305_124638.0000_to_20190305_125231.0000_RMA3_0200_01.nc'
	gfile     = '1B.GPM.GMI.TB2016.20190305-S123614-E140847.028498.V05A.HDF5'
	era5_file = '20190305_13.grib'
	#
	opts = {'xlim_min': -63, 'xlim_max': -58, 'ylim_min': -27, 'ylim_max': -23, 'ylim_max_zoom': -24.5, 'ZDRoffset': 3}
	# 
	ERA5_field = xr.load_dataset(era5_dir+era5_file, engine="cfgrib")	
	elemj      = find_nearest(ERA5_field['latitude'], lat_pfs[0])
	elemk      = find_nearest(ERA5_field['longitude'], lon_pfs[0])	
	tfield_ref = ERA5_field['t'][:,elemj,elemk] - 273 # convert to C
	geoph_ref  = (ERA5_field['z'][:,elemj,elemk])/9.80665
	# Covert to geop. height (https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
	Re         = 6371*1e3
	alt_ref    = (Re*geoph_ref)/(Re-geoph_ref) 
	#-------------------------- DONT CHANGE
	radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/'+rfile)
	#
	# ADD temperature info: 
    	print('Freezing level at ', np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3), ' km')
    	freezing_lev = np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3) 
    	radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    	radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    	#- Add height field for 4/3 propagation
    	radar_height = get_z_from_radar(radar)
    	radar = add_field_to_radar_object(radar_height, radar, field_name = 'height')    
    	iso0 = np.ma.mean(radar.fields['height']['data'][np.where(np.abs(radar.fields['sounding_temperature']['data']) < 0)])
    	radar.fields['height_over_iso0'] = deepcopy(radar.fields['height'])
    	radar.fields['height_over_iso0']['data'] -= iso0 
	#----
	#- output of plot_Zhppi_wGMIcontour depends on the number of icois of interest. Here in this case we have three:
   	# OJO for use_freezingLev == 0 es decir ground level! 
    	[gridded, frezlev, GMI_lon_COI1, GMI_lat_COI1, GMI_tbs1_COI1, RN_inds_COI1, RB_inds_COI1, 
     	GMI_lon_COI2, GMI_lat_COI2, GMI_tbs1_COI2, RN_inds_COI2, RB_inds_COI2] = plot_Zhppi_wGMIcontour(radar, lat_pfs, lon_pfs, 'radar at '+rfile[15:19]+' UTC and PF at '+time_pfs[0]+' UTC', 
                           gmi_dir+gfile, 0, opts, era5_dir+era5_file, icoi=[6,7], use_freezingLev=0)
	# SWEEP 0
	plot_sweep0(radar, opts, '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'+gfile)
	plot_gmi('/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'+gfile, opts,
                                  '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/', rfile, [lon_pfs[0]], [lat_pfs[0]], 
			 	reference_satLOS=100)
	# CHECK TRANSECTS
	check_transec(radar, 176, lon_pfs, lat_pfs)       
	check_transec(radar, 210, lon_pfs, lat_pfs)      
	check_transec(radar, 30, lon_pfs, lat_pfs)      
	# CHECK BB -----------------------------------------------------------------------------------------
	rfileBB='cfrad.20190212_092509.0000_to_20190212_092757.0000_RMA3_0200_02.nc'
	radarBB = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/find_BB/'+rfileBB)
	plot_sweep_nlev(radarBB, opts, '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'+gfile, 3)
	check_transec_rma_campos('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/find_BB/', 
				 fileBB, 200, 'ZDR', 3)       
	check_transec(radarBB, 200, lon_pfs, lat_pfs)       
	# ---------------------------------------------------------------------------------------------------
	#- CHECK 1,5,10 KM RESOLUTION	
	grided_05KM = pyart.map.grid_from_radars(radar, grid_shape=(20, 950, 950), 
                                       grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                    (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                       roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')    
	grided_1KM = pyart.map.grid_from_radars(radar, grid_shape=(20, 470, 470), 
                                       grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                    (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                       roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')    
    	grided_5KM = pyart.map.grid_from_radars(radar, grid_shape=(20, 94, 94), 
                                   grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                   roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')
    	grided_10KM = pyart.map.grid_from_radars(radar, grid_shape=(20, 48, 48), 
                               grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                            (-np.max(radar.range['data']), np.max(radar.range['data']))),
                               roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')
	#- and run pseudo RHIS
	make_pseudoRHISfromGrid(grided_1KM, radar, [176,210,30],['6[Phail = 0.534]','7',''], [100, 100, 100])
	make_pseudoRHISfromGrid(grided_5KM, radar, [176,210,30],['6[Phail = 0.534]','7',''], [100, 100, 100])
	make_pseudoRHISfromGrid(grided_10KM, radar, [176,210,30],['6[Phail = 0.534]','7',''], [100, 100, 100])
	#- run with radar grid
	plot_rhi_RMA(rfile, '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures', 
		     '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/', 'RMA3', 0, 150, 176, 3, 5)
	plot_rhi_RMA(rfile, '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures', 
		     '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/', 'RMA3', 0, 200, 210, 3, 5)
	plot_rhi_RMA(rfile, '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures', 
		     '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/', 'RMA3', 0, 150, 30, 3, 5)		
		

		

	
		


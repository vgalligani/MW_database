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
from copy import deepcopy
import matplotlib.colors as colors
import wradlib as wrl    
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.path import Path
from cycler import cycler
#import seaborn as sns
import cartopy.io.shapereader as shpreader
import copy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import gc
import math
from pyart.core.transforms import antenna_to_cartesian

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
    Re           = 6371*1e3
    alt_ref      = (Re*geoph_ref)/(Re-geoph_ref)
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
    grided  = pyart.map.grid_from_radars(radar, grid_shape=(20, 470, 470), 
                                       grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                    (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                       roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')    
    grided2 = pyart.map.grid_from_radars(radar, grid_shape=(20, 94, 94), 
                                   grid_limits=((0.,20000,), (-np.max(radar.range['data']), np.max(radar.range['data'])),
                                                (-np.max(radar.range['data']), np.max(radar.range['data']))),
                                   roi_func='dist_beam', min_radius=500.0, weighting_function='BARNES2')

    #------ 	
    frezlev      = find_nearest(grided.z['data']/1e3, freezing_lev) 
    #------

    #----------------------------------------------------------------------------------------
    # Test plot figure: General figure with Zh and the countours identified 
    #----------------------------------------------------------------------------------------
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
    #----------------------------------------------------------------------------------------



    #----------------------------------------------------------------------------------------
    #---- NEW FIGURE: different gridded resolutions (original radar, 1km and 5 km BARNES) 
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True,
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

    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    # Add labels:
    labels = ["200 K"] 
    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])
    axes[2].legend(loc='upper left', fontsize=fontize)
    #----------------------------------------------------------------------------------------



    #----------------------------------------------------------------------------------------
    # NEW FIGURE: different horizontal cuts
    #----------------------------------------------------------------------------------------
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True, figsize=[14,12])
    counter = 0
    for i in range(3):
        for j in range(3):
            axes[i,j].pcolormesh(grided.point_longitude['data'][counter,:,:], grided.point_latitude['data'][counter,:,:], 
                  grided.fields['TH']['data'][counter,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
            counter=counter+1;
            axes[i,j].set_title('horiz. cut at '+str( round(grided.z['data'][counter]/1e3,2)))
            axes[i,j].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
            axes[i,j].set_xlim([options['xlim_min'], options['xlim_max']])
            axes[i,j].set_ylim([options['ylim_min'], options['ylim_max']])


    #----------------------------------------------------------------------------------------
    # NEW FIGURE. solo dos paneles: Same as above but plt lowest level y closest to freezing level!
    #----------------------------------------------------------------------------------------
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
    axes[1].set_title('Freezing level ('+str( round(grided.z['data'][frezlev]/1e3,2) )+' km)')
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
    axes[0].plot(lon_gmi_inside[inds_1], lat_gmi_inside[inds_1], 'o', markersize=10, markerfacecolor='black')
    axes[1].plot(lon_gmi_inside[inds_1], lat_gmi_inside[inds_1], 'o', markersize=10, markerfacecolor='black')
    if ii == 1:
        axes[0].plot(lon_gmi_inside[inds_2], lat_gmi_inside[inds_2], 'o', markersize=10, markerfacecolor='darkblue')
        axes[1].plot(lon_gmi_inside[inds_2], lat_gmi_inside[inds_2], 'o', markersize=10, markerfacecolor='darkblue')
    if ii == 2:
        axes[0].plot(lon_gmi_inside[inds_2], lat_gmi_inside[inds_2], 'o', markersize=10, markerfacecolor='darkblue')
        axes[0].plot(lon_gmi_inside[inds_3], lat_gmi_inside[inds_3], 'o', markersize=10, markerfacecolor='darkred')
        axes[1].plot(lon_gmi_inside[inds_2], lat_gmi_inside[inds_2], 'o', markersize=10, markerfacecolor='darkblue')
        axes[1].plot(lon_gmi_inside[inds_3], lat_gmi_inside[inds_3], 'o', markersize=10, markerfacecolor='darkred')

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
    #----------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------	
    #---- HID FIGURES! (ojo sin corregir ZH!!!!) 
    #----------------------------------------------------------------------------------------
    #radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    #radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    #dzh_  = radar.fields['TH']['data']
    #dzv_  = radar.fields['TV']['data']
    #drho_ = radar.fields['RHOHV']['data']
    #dkdp_ = radar.fields['corrKDP']['data']
    # Filters
    #ni = dzh_.shape[0]
    #nj = dzh_.shape[1]
    #for i in range(ni):
    #    rho_h = drho_[i,:]
    #    zh_h = dzh_[i,:]
    #    for j in range(nj):
    #        if (rho_h[j]<0.7) or (zh_h[j]<30):
    #            dzh_[i,j]  = np.nan
    #            dzv_[i,j]  = np.nan
    #            drho_[i,j]  = np.nan
    #            dkdp_[i,j]  = np.nan

    #scores          = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=(dzh_-dzv_) - opts['ZDRoffset'], rho=drho_, kdp=dkdp_, 
    #                                         use_temp=True, band='C', T=radar_T)
    #HID             = np.argmax(scores, axis=0) + 1

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
    dkdp_grid = grided.fields['corrKDP']['data']

    # Filters
    ni = dzh_grid.shape[0]
    nj = dzh_grid.shape[1]
    nk = dzh_grid.shape[2]
    for i in range(ni):
        rho_hh = drho_grid[i,:,:]
        zh_hh = dzh_grid[i,:,:]
        for j in range(nj):
            for k in range(nk):
                if (rho_hh[j,k]<0.7) or (zh_hh[j,k]<30):
                		dzh_grid[i,j,k]  = np.nan
                		dzv_grid[i,j,k]  = np.nan
                		drho_grid[i,j,k]  = np.nan
                		dkdp_grid[i,j,k]  = np.nan

    scores          = csu_fhc.csu_fhc_summer(dz=dzh_grid, zdr=(dzh_grid-dzv_grid) - opts['ZDRoffset'], rho=drho_grid, kdp=dkdp_grid, 
                                            use_temp=True, band='C', T=radargrid_TT)
    GRIDDED_HID = np.argmax(scores, axis=0) + 1 
    print(GRIDDED_HID.shape)
    print(grided.point_latitude['data'].shape)

    #---- plot hid ppi  
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)

    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True,
                        figsize=[14,4])
    pcm1 = axes[0].pcolormesh(lons, lats, radar.fields['HID']['data'][start_index:end_index], cmap=cmaphid, vmin=0.2, vmax=10)
    axes[0].set_title('HID radar nlev 0 PPI')
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    
    pcm1 = axes[1].pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:], GRIDDED_HID[0,:,:], cmap=cmaphid, vmin=0.2, vmax=10)
    axes[1].set_title('HID GRIDDED 0 km')
    axes[1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);		
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    
    pcm1 = axes[2].pcolormesh(grided.point_longitude['data'][frezlev,:,:], grided.point_latitude['data'][frezlev,:,:], GRIDDED_HID[frezlev,:,:], cmap=cmaphid, vmin=0.2, vmax=10)
    axes[2].set_title(r'HID GRIDDED at '+str(round(grided.z['data'][frezlev]/1e3,2))+' km)')
    axes[2].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[2].set_ylim([options['ylim_min'], options['ylim_max']])
    axes[2].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);			
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)

    #axes[0].contour(lons, lats, radar.fields['TH']['data'][start_index:end_index], [30], colors=(['r']), linewidths=1.5);	
    #axes[1].contour(lons, lats, radar.fields['TH']['data'][start_index:end_index], [30], colors=(['r']), linewidths=1.5);	
    #axes[2].contour(lons, lats, radar.fields['TH']['data'][start_index:end_index], [30], colors=(['r']), linewidths=1.5);	

    p1 = axes[0].get_position().get_points().flatten()
    p2 = axes[1].get_position().get_points().flatten();
    p3 = axes[2].get_position().get_points().flatten(); 
    ax_cbar = fig.add_axes([p3[0]+(p3[0]-p2[0])+0.04, 0.1, 0.02, 0.8])   #ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
    cbar = fig.colorbar(pcm1, cax=ax_cbar, shrink=0.9, label='CSU HID')
    cbar = adjust_fhc_colorbar_for_pyart(cbar)
    #cbar.cmap.set_under('white')

    inds_1.shape
    # no sacar los gridded
    #if ii == 0:
    #    return grided, frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1],tb_s1_gmi_inside[inds_1,:], inds_RN1, inds_RB1
    #if ii == 1:
    #    return grided, frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1],tb_s1_gmi_inside[inds_1,:], inds_RN1, inds_RB1, lon_gmi_inside[inds_2], lat_gmi_inside[inds_2],tb_s1_gmi_inside[inds_2,:], inds_RN2, inds_RB2
    #if ii == 2:
    #    return grided, frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1],tb_s1_gmi_inside[inds_1,:], inds_RN1, inds_RB1, lon_gmi_inside[inds_2], lat_gmi_inside[inds_2],tb_s1_gmi_inside[inds_2,:], inds_RN2, inds_RB2, lon_gmi_inside[inds_3], lat_gmi_inside[inds_3],tb_s1_gmi_inside[inds_3,:], inds_RN3, inds_RB3
    if ii == 0:
        return frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1], tb_s1_gmi_inside[inds_1,:], inds_RN1
    if ii == 1:
        return frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1], tb_s1_gmi_inside[inds_1,:],  inds_RN1, lon_gmi_inside[inds_2], lat_gmi_inside[inds_2],tb_s1_gmi_inside[inds_2,:], inds_RN2
    if ii == 2:
        return frezlev, lon_gmi_inside[inds_1], lat_gmi_inside[inds_1], tb_s1_gmi_inside[inds_1,:],  inds_RN1, lon_gmi_inside[inds_2], lat_gmi_inside[inds_2],tb_s1_gmi_inside[inds_2,:], inds_RN2, lon_gmi_inside[inds_3], lat_gmi_inside[inds_3],tb_s1_gmi_inside[inds_3,:], inds_RN3


#------------------------------------------------------------------------------  
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
	if 'VRAD' in radar.fields.keys():  
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
        hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
                'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
        cmaphid = colors.ListedColormap(hid_colors)
        #cmaphid.set_bad('white')
        #cmaphid.set_under('white')
        #cmaphid.set_over('white')
        # Figure
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        im_TH  = axes[0,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_THTH, cmap=cmap, vmin=vmin, vmax=vmax)
        im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, (grid_THTH-grid_TVTV)-opts['ZDRoffset'], cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)
        im_RHO = axes[2,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_RHO, cmap=pyart.graph.cm.RefDiff , vmin=0.7, vmax=1.)
        im_HID = axes[3,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=0.2, vmax=10)
        
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
    elif 'DBZHCC' in radar.fields.keys():
        dz_field='DBZHCC'
    elif 'reflectivity' in radar.fields.keys():
        dz_field='reflectivity'	
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
    
    
    cb.set_ticks(np.arange(0.4, 10, 0.9))
    cb.ax.set_yticklabels(['','Drizzle','Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def adjust_meth_colorbar_for_pyart(cb, tropical=False):
    if not tropical:
        cb.set_ticks(np.arange(1.25, 5, 0.833))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z)', 'R(Zrain)'])
    else:
        cb.set_ticks(np.arange(1.3, 6, 0.85))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z_all)', 'R(Z_c)', 'R(Z_s)'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def plot_gmi(fname, options, radar, lon_pfs, lat_pfs, icoi):

    # coi       len(contorno89.collections[0].get_paths()) = cantidad de contornos.
    #           me interesa tomar los vertices del contorno de interes. 
    #           Entonces abajo hago este loop   
    #               for ii in range(len(coi)):
    #                   i = coi[ii]
    #                   for ik in range(len(contorno89.collections[0].get_paths()[i].vertices)):  
    # Como saber cual es el de interes? lon_pfs y lat_pfs estan adentro de ese contorn. entonces podria
    # borrar coi del input y que loopee por todos ... 
    #plt.matplotlib.rc('font', family='serif', size = 20)
    #plt.rcParams['font.sans-serif'] = ['Helvetica']

    if options['radar_name'] == 'RMA1':
        reflectivity_name = 'TH'   
        nlev = 0 
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]

    elif options['radar_name'] == 'DOW7':
        reflectivity_name = 'DBZHCC'   
        lats        = radar.gate_latitude['data']
        lons        = radar.gate_longitude['data']
        ZH          = radar.fields[reflectivity_name]['data']
    elif options['radar_name'] == 'CSPR2':
        reflectivity_name = 'corrected_reflectivity'   
        lats        = radar.gate_latitude['data']
        lons        = radar.gate_longitude['data']
        ZH          = radar.fields[reflectivity_name]['data']
	
    elif options['radar_name'] == 'RMA5':
        reflectivity_name = 'DBZH'   
        lats        = radar.gate_latitude['data']
        lons        = radar.gate_longitude['data']
        ZH          = radar.fields[reflectivity_name]['data']	
	
    s_sizes=450
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

    S1_sub_lat  = lat_gmi.copy()
    S1_sub_lon  = lon_gmi.copy()
    S1_sub_tb   = tb_s1_gmi.copy()
    S2_sub_tb   = tb_s2_gmi.copy()
    S2_sub_lat  = lat_s2_gmi.copy()
    S2_sub_lon  = lon_s2_gmi.copy()	

    # Tambien se puenden hacer recortes guardando los indices. ejemplo para S1: 
    idx1 = (lat_gmi>=options['ylim_min']-5) & (lat_gmi<=options['ylim_max']+5) & (lon_gmi>=options['xlim_min']-5) & (lon_gmi<=options['xlim_max']+5)
    #idx1 = (lat_gmi>=options['ylim_min']) & (lat_gmi<=options['ylim_max']+1) & (lon_gmi>=options['xlim_min']) & (lon_gmi<=options['xlim_max']+2)
    #S1_sub_lat  = lat_gmi[:,:][idx1] 
    #S1_sub_lon  = lon_gmi[:,:][idx1]
    #S1_subch89V = tb_s1_gmi[:,:,7][idx1]		

    idx2 = (lat_s2_gmi>=options['ylim_min']-5) & (lat_s2_gmi<=options['ylim_max']+5) & (lon_s2_gmi>=options['xlim_min']-5) & (lon_s2_gmi<=options['xlim_max']+5)
    #idx2 = (lat_s2_gmi>=options['ylim_min']) & (lat_s2_gmi<=options['ylim_max']+1) & (lon_s2_gmi>=options['xlim_min']) & (lon_s2_gmi<=options['xlim_max']+2)
    #S2_sub_lat  = lat_s2_gmi[:,:][idx2] 
    #S2_sub_lon  = lon_s2_gmi[:,:][idx2]

    # CALCULATE PCTs
    #for j in range(lon_gmi.shape[1]):
    #  tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan   	
    #  lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  
    #  lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  	
    # PCT10, PCT19, PCT37, PCT89 = calc_PCTs(tb_s1_gmi)

    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    S2_sub_lat = np.where(idx2 != False, S2_sub_lat, np.nan) 
    S2_sub_lon = np.where(idx2 != False, S2_sub_lon, np.nan) 
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    for i in range(tb_s2_gmi.shape[2]):
        S2_sub_tb[:,:,i]  = np.where(np.isnan(S2_sub_lon) != 1, tb_s2_gmi[:,:,i], np.nan)	
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)

    # griddata
    #from scipy.interpolate import griddata
    #reso_x = 0.005
    #reso_y = 0.005
    #xgrid = np.arange(options['xlim_min'], options['xlim_max']+reso_x, reso_x )
    #ygrid = np.arange(options['ylim_min'], options['ylim_max']+reso_y, reso_y )
    #xx, yy = np.meshgrid(xgrid, ygrid)
    #BT37 = griddata( (S1_sub_lon, S1_sub_lat), tb_s1_gmi[:,:,5][idx1], (xx,yy), method='nearest')	
    #BT85 = griddata( (S1_sub_lon, S1_sub_lat), tb_s1_gmi[:,:,7][idx1], (xx,yy), method='nearest')	
    #BT166 = griddata( (S2_sub_lon, S2_sub_lat), tb_s2_gmi[:,:,0][idx2], (xx,yy), method='nearest')	

    fig = plt.figure(figsize=(30,10)) 
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
    im = plt.scatter(S1_sub_lon, S1_sub_lat, c=S1_sub_tb[:,:,5], s=s_sizes, marker='h', vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    #im = plt.pcolormesh(xx, yy, BT37, vmin=50, vmax=300, cmap=cmaps['turbo_r'])    
    plt.title('BT 37 GHz')
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max'],1), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, '-k', linewidth=1)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=20, markerfacecolor="magenta",
            markeredgecolor='magenta', markeredgewidth=1.5) 
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    plt.contour(lon_gmi, lat_gmi, PCT89, [200, 225], colors=('m'), linewidths=2);
    plt.plot(np.nan, np.nan, '-m', linewidth=2, label='PCT89 200/225 K ')
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
    im = plt.scatter(S1_sub_lon, S1_sub_lat, c=S1_sub_tb[:,:,7], marker='h', s=s_sizes, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    #im = plt.pcolormesh(xx, yy, BT85, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 89 GHz')
    ax2.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax2.set_yticks(np.arange(options['ylim_min'], options['ylim_max'],1), crs=crs_latlon)
    ax2.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=20, markerfacecolor="magenta",
            markeredgecolor='magenta', markeredgewidth=1.5)   
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    contorno89 = plt.contour(lon_gmi, lat_gmi, PCT89, [200,225], colors=('m'), linewidths=2);            

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
    im = plt.scatter(S2_sub_lon, S2_sub_lat, c=S2_sub_tb[:,:,0], marker='h', s=s_sizes, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    #im = plt.pcolormesh(xx, yy, BT166, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 166 GHz')
    ax3.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax3.set_yticks(np.arange(options['ylim_min'], options['ylim_max'],1), crs=crs_latlon)
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
    contorno89 = plt.contour(lon_gmi, lat_gmi, PCT89, [200,225], colors=('m'), linewidths=2);            
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
    ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.03])   # [left, bottom, width, height] or Bbox 
    cbar = fig.colorbar(im, cax=ax_cbar, shrink=0.8, ticks=np.arange(50,300,50), extend='both', orientation="horizontal", label='TBV (K)')   

    # antes hacia esto a mano: 
    # contour n1 (OJO ESTOS A MANO!) pero ahora no solo me interesa el que tiene el PF sino 
    # otros posibles para comparar pq no se detectaron como P_hail ... estos podrian ser a mano entonces ... 
    # tirar en dos partes. primero plot_gmi que me tira la figura con TODOS los contornos y luego la otra con los que
    # me interesan ... 

    fig.savefig(options['fig_dir']+'GMI_basicTBs.png', dpi=300, transparent=False)  
    #plt.close()



    #----------------------------------------------------------------------------------------
    # NEW FIGURE. solo dos paneles: Same as above but plt lowest level y closest to freezing level!
    #----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[7,6])
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    axes.pcolormesh(lons, lats, ZH, cmap=cmap, vmax=vmax, vmin=vmin)
    axes.set_title('Ground Level')
    axes.set_xlim([options['xlim_min']-5, options['xlim_max']+5])
    axes.set_ylim([options['ylim_min']-5, options['ylim_max']+5])
    # -----
    # CONTORNO CORREGIDO POR PARALAJE Y PODER CORRER LOS ICOIS, simplemente pongo nans fuera del area de interes ... 
    #contorno89 = axes.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 225], colors=(['k']), linewidths=1.5);

    # ---- GET COIs:
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
        #convexhull = ConvexHull(vertices)
        array_points = np.array(vertices)
        #------- testing from https://stackoverflow.com/questions/57260352/python-concave-hull-polygon-of-a-set-of-lines 
        alpha = 0.95 * alphashape.optimizealpha(array_points)
        hull_pts_CONCAVE = alphashape.alphashape(array_points, alpha)
        hull_coors_CONCAVE = hull_pts_CONCAVE.exterior.coords.xy
        check_points = np.vstack((hull_coors_CONCAVE)).T
        concave_path = Path(check_points)
        #-------
        ##datapts = np.column_stack((S1_sub_lon,S1_sub_lat))
        datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] ))
        # Ahora como agarro los Zh, ZDR, etc inside the contour ... 
        datapts_RADAR_NATIVE = np.column_stack((np.ravel(lons),np.ravel(lats)))
        #
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[14,12])   
        axes.pcolormesh(lon_gmi, lat_gmi, PCT89, cmap = cmaps['turbo_r'] ); plt.xlim([-70,-60]); plt.ylim([-40,-20]); 
        axes.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200])
        #axes.title(str(item))


        if ii==0:
            inds_1  = concave_path.contains_points(datapts)
            #axes.plot(S1_sub_lon[inds_1], S1_sub_lat[inds_1], 'o', markersize=10, markerfacecolor='black')
            axes.plot(lon_gmi[:,:][idx1][inds_1], lat_gmi[:,:][idx1][inds_1], 'o', markersize=10, markerfacecolor='black')
            dummy = axes.plot(np.nan, np.nan, 'o', markersize=20, markerfacecolor='black', label='icoi:'+str(icoi[0]))
        if ii==1:
            inds_2  = concave_path.contains_points(datapts)
            #axes.plot(S1_sub_lon[inds_2], S1_sub_lat[inds_2], 'o', markersize=10, markerfacecolor='darkblue')
            axes.plot(lon_gmi[:,:][idx1][inds_2], lat_gmi[:,:][idx1][inds_2], 'o', markersize=10, markerfacecolor='black')
            dummy = axes.plot(np.nan, np.nan, 'o', markersize=20, markerfacecolor='darkblue', label='icoi:'+str(icoi[1]))
        if ii==2:
            inds_3  = concave_path.contains_points(datapts)
            #axes.plot(S1_sub_lon[inds_3], S1_sub_lat[inds_3], 'o', markersize=10, markerfacecolor='darkred')
            axes.plot(lon_gmi[:,:][idx1][inds_3], lat_gmi[:,:][idx1][inds_3], 'o', markersize=10, markerfacecolor='black')
            dummy = axes.plot(np.nan, np.nan, 'o', markersize=20, markerfacecolor='darkred', label='icoi:'+str(icoi[2]))

    print(ii)
    # LIMITS	 
    axes.set_xlim([options['xlim_min'], options['xlim_max']])
    axes.set_ylim([options['ylim_min'], options['ylim_max']])
    # TITLE
    axes.set_title('nsweep 0: contours of interest (COI)')
    # RADAR RINGS	
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    # Addlabels to icois! 
    axes.legend()

    # Add labels:
    labels = ["200 K"] 
    for i in range(len(labels)):
        contorno89.collections[i].set_label(labels[i])
    axes.legend(loc='upper left')

    fig.savefig(options['fig_dir']+'GMI_icois_onZH.png', dpi=300, transparent=False)  
    #plt.close()
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
def check_transec_CSPR(radar, test_transect, lon_pf, lat_pf, options):
	
    nlev  = 0  
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[13,12])
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats = radar.gate_latitude['data'][start_index:end_index]
    lons = radar.gate_longitude['data'][start_index:end_index]
    pcm1 = axes.pcolormesh(lons, lats, radar.fields['corrected_reflectivity']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
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
    TransectNo = np.nanmin(np.asarray(abs(azimuths-test_transect)<=0.5).nonzero())
    lon_transect  = lons[start_index:end_index][TransectNo,:]
    lat_transect    = lats[start_index:end_index][TransectNo,:]
    plt.plot(lon_transect, lat_transect, '-k')	
    plt.title('Transecta Nr:'+ str(test_transect), Fontsize=20)
    for iPF in range(len(lat_pf)):
        plt.plot(lon_pf[iPF], lat_pf[iPF], marker='*', markersize=40, markerfacecolor="None",
            markeredgecolor='black', markeredgewidth=2, label='GMI(PF) center') 

    # read 
    f = h5py.File( options['gmi_dir']+options['gfile'], 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    for j in range(lon_gmi.shape[1]):
        tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
        tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan   
        lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
        lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  
        lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
        lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  	
    PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 	
    CS = axes.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 225], colors=(['black', 'gray']), linewidths=1.5)
    labels_cont = ['GMI 200K contour', 'GMI 225K contour']
    for i in range(len(labels_cont)):
        CS.collections[i].set_label(labels_cont[i])
    if len(options['REPORTES_meta'])>0:
        for ireportes in range(len(options['REPORTES_geo'])):
            axes.plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], '*', markeredgecolor='black', markerfacecolor='black', markersize=10, label=options['REPORTES_meta'][ireportes])
        plt.legend() 
    
    fig.savefig(options['fig_dir']+'PPI_transect_'+'azi'+str(test_transect)+'.png', dpi=300,transparent=False)   
    return 









#------------------------------------------------------------------------------
#------------------------------------------------------------------------------














def check_transec(radar, test_transect, lon_pf, lat_pf, options):
	
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

    # read 
    f = h5py.File( options['gmi_dir']+options['gfile'], 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    for j in range(lon_gmi.shape[1]):
        tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
        tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan   
        lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
        lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  
        lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
        lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  	
    PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 	
    CS = axes.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 225], colors=(['black', 'gray']), linewidths=1.5)
    labels_cont = ['GMI 200K contour', 'GMI 225K contour']
    for i in range(len(labels_cont)):
        CS.collections[i].set_label(labels_cont[i])
    if len(options['REPORTES_meta'])>0:
        for ireportes in range(len(options['REPORTES_geo'])):
            axes.plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], '*', markeredgecolor='black', markerfacecolor='black', markersize=10, label=options['REPORTES_meta'][ireportes])
        plt.legend() 
    
    fig.savefig(options['fig_dir']+'PPI_transect_'+'azi'+str(test_transect)+'.png', dpi=300,transparent=False)   
    return 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def make_pseudoRHISfromGrid_DOW7(gridded_radar, radar, azi_oi, titlecois, xlims_xlims_mins, xlims_xlims, alt_ref, tfield_ref, options): 

    plt.matplotlib.rc('font', family='serif', size = 20)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['font.serif'] = ['Helvetica']
    THname = 'DBZHCC' 
    TVname = 'ZDRC'
    start_index = radar.sweep_start_ray_index['data']
    end_index   = radar.sweep_end_ray_index['data']
    lats        = radar.gate_latitude['data']
    lons        = radar.gate_longitude['data']
    azimuths    = radar.azimuth['data']

    fig, axes = plt.subplots(nrows=4, ncols=3, constrained_layout=True, figsize=[13,12])

    for iz in range(len(azi_oi)):
        target_azimuth = azimuths[azi_oi[iz]]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()	 # <==== ?????
        grid_lon   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_lon[:]   = np.nan
        grid_lat   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_lat[:]   = np.nan
        grid_THTH  = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_THTH[:]  = np.nan
        grid_ZDR   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_ZDR[:]  = np.nan
        grid_alt   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_alt[:]   = np.nan
        grid_range = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_range[:] = np.nan
        grid_RHO   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_RHO[:]   = np.nan
        grid_HID   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_HID[:]   = np.nan
        grid_KDP   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_KDP[:]  = np.nan

        # need to find x/y pair for each gate at the surface 
        for i in range(lons[filas,:].shape[2]):	
            # First, find the index of the grid point nearest a specific lat/lon.   
            abslat = np.abs(gridded_radar.point_latitude['data'][0,:,:]  - lats[filas,i])
            abslon = np.abs(gridded_radar.point_longitude['data'][0,:,:] - lons[filas,i])
            c = np.maximum(abslon, abslat)
            ([xloc], [yloc]) = np.where(c == np.min(c))	
            grid_lon[:,i]   = gridded_radar.point_longitude['data'][:,xloc,yloc]
            grid_lat[:,i]   = gridded_radar.point_latitude['data'][:,xloc,yloc]
            grid_ZDR[:,i]   = gridded_radar.fields[TVname]['data'][:,xloc,yloc]
            grid_THTH[:,i]  = gridded_radar.fields[THname]['data'][:,xloc,yloc]
            grid_RHO[:,i]   = gridded_radar.fields['RHOHV']['data'][:,xloc,yloc]
            grid_alt[:,i]   = gridded_radar.z['data'][:]
            x               = gridded_radar.point_x['data'][:,xloc,yloc]
            y               = gridded_radar.point_y['data'][:,xloc,yloc]
            z               = gridded_radar.point_z['data'][:,xloc,yloc]
            grid_range[:,i] = ( x**2 + y**2 + z**2 ) ** 0.5
            grid_KDP[:,i]   = gridded_radar.fields['KDP']['data'][:,xloc,yloc]
            grid_HID[:,i]   = gridded_radar.fields['HID']['data'][:,xloc,yloc]
	
        ni = grid_HID.shape[0]
        nj = grid_HID.shape[1]
        for i in range(ni):
            rho_h = grid_RHO[i,:]
            zh_h = grid_THTH[i,:]
            for j in range(nj):
                if (rho_h[j]<0.7) or (zh_h[j]<0):
                    grid_THTH[i,j] = np.nan
                    grid_ZDR[i,j]  = np.nan
                    grid_RHO[i,j]  = np.nan			
				
				
        #Filters
        #grid_TVTV[np.where(grid_RHO<0.6)] = np.nan	
        #grid_THTH[np.where(grid_RHO<0.6)] = np.nan	
        #grid_RHO[np.where(grid_RHO<0.6)] = np.nan	
        #grid_KDP[np.where(grid_RHO<0.6)] = np.nan	

        #--- need to add tempertaure to grided data ... 
        #radar_z = gridded_radar.point_z['data']
        #shape   = np.shape(radar_z)
        #rad_T1d = np.interp(radar_z.ravel(), alt_ref, tfield_ref)   # interpolate_sounding_to_radar(snd_T, snd_z, radar)
        #radargrid_TT = np.reshape(rad_T1d, shape)
        #gridded_radar = add_field_to_radar_object(np.reshape(rad_T1d, shape), gridded_radar, field_name='sounding_temperature')      
        #- Add height field for 4/3 propagation
        #gridded_radar = add_field_to_radar_object( gridded_radar.point_z['data'], gridded_radar, field_name = 'height')    	
        #iso0 = np.ma.mean(gridded_radar.fields['height']['data'][np.where(np.abs(gridded_radar.fields['sounding_temperature']['data']) < 0)])
        #gridded_radar.fields['height_over_iso0'] = deepcopy(gridded_radar.fields['height'])
        #gridded_radar.fields['height_over_iso0']['data'] -= iso0 
        #
        #for i in range(lons[filas,:].shape[2]):	
        #    scores          = csu_fhc.csu_fhc_summer(dz=grid_THTH[:,i], zdr=(grid_TVTV[:,i]-grid_THTH[:,i]) - options['ZDRoffset'], 
	#						 rho=grid_RHO[:,i], kdp=grid_KDP[:,i], 
        #                                    use_temp=True, band='C', T=radargrid_TT)
        #    grid_HID[:,i] = np.argmax(scores, axis=0) + 1 

        #grid_HID[np.where(grid_RHO<0.7)] = np.nan

        #---- plot hid ppi  
        hid_colors = ['White', 'LightBlue','MediumBlue', 'DarkOrange', 'LightPink',
                'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
        cmaphid = colors.ListedColormap(hid_colors)
        #cmaphid.set_bad('white')
        #cmaphid.set_under('white')
        # Figure
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        im_TH  = axes[0,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_THTH, cmap=cmap, vmin=vmin, vmax=vmax)

        im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_ZDR, cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)

        im_RHO = axes[2,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_RHO, cmap=pyart.graph.cm.RefDiff , vmin=0.7, vmax=1.)

        im_HID = axes[3,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=0.2, vmax=10)

        axes[0,iz].set_title('coi='+titlecois[iz]) 
        if iz == 1:
            axes[0,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[1,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[2,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[3,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])
	
        if iz == 2:
            axes[0,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[1,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[2,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[3,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])
	
        if iz == 3:
            axes[0,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[1,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[2,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[3,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])
		
        if iz == 0:
            axes[0,0].set_ylabel('Altitude (km)')
            axes[1,0].set_ylabel('Altitude (km)')
            axes[2,0].set_ylabel('Altitude (km)')
            axes[3,0].set_ylabel('Altitude (km)')
            axes[0,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[1,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[2,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,0].set_xlabel('Range (km)')
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])
	
        if iz == len(azi_oi)-1: 
	    # Add colorbars #ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
            pm1    = axes[0,iz-1].get_position().get_points().flatten()
            p_last = axes[0,iz].get_position().get_points().flatten(); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.76, 0.02, 0.2])  
            cbar    = fig.colorbar(im_TH,  cax=ax_cbar, shrink=0.9,  ticks=np.arange(0,60.01,10), label='ZH')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            pm2    = axes[1,iz-1].get_position().get_points().flatten()

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.55, 0.02, 0.2])  
            cbar    = fig.colorbar(im_ZDR, cax=ax_cbar, shrink=0.9,  ticks=np.arange(-2.,5.01,1.), label='ZDR')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            pm3   = axes[2,iz-1].get_position().get_points().flatten()

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.28, 0.02, 0.2])  
            cbar    = fig.colorbar(im_RHO, cax=ax_cbar, shrink=0.9, ticks=np.arange(0.7,1.01,0.1), label='RHO')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.03, 0.02, 0.2])  
            cbar    = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            cbar = adjust_fhc_colorbar_for_pyart(cbar)
            #cbar.cmap.set_under('white')

            pm2    = axes[3,iz-1].get_position().get_points().flatten()

		
    #- savefile
    fig.savefig(options['fig_dir']+'PseudoRHIS_GRIDDED'+'.png', dpi=300,transparent=False)   
    #plt.close()
    del grid_THTH, grid_RHO, grid_ZDR, grid_HID
	
    #-------------------------------
    for i in range(20):
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
        im_HID = ax.pcolormesh(gridded_radar.point_longitude['data'][i,:,:], 
        gridded_radar.point_latitude['data'][i,:,:], gridded_radar.fields['HID']['data'][i,:,:], cmap=cmaphid, vmin=0.2, vmax=10)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)  
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title('HID '+ str( round(grid_alt[i,0]/1e3,1)) +' km')
        cbar = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')
        cbar = adjust_fhc_colorbar_for_pyart(cbar)
        ax.set_xlim([options['xlim_min'], options['xlim_max']])	
        ax.set_ylim([options['ylim_min'], options['ylim_max']])
        fig.savefig(options['fig_dir']+'RHIS_GRIDDED_verticalLevel_'+str(i)+'.png', dpi=300,transparent=False); plt.close()
    return
	
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------	


#------------------------------------------------------------------------------
def make_pseudoRHISfromGrid(gridded_radar, radar, azi_oi, titlecois, xlims_xlims_mins, xlims_xlims, alt_ref, tfield_ref, options): 

    plt.matplotlib.rc('font', family='serif', size = 20)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['font.serif'] = ['Helvetica']

    if 'TH' in radar.fields.keys():  
            THname= 'TH'
            TVname= 'TV'
            KDPname='corrKDP'
    elif 'DBZHCC' in radar.fields.keys():        
           THname = 'DBZHCC'
           KDPname='corrKDP'
    elif 'corrected_reflectivity' in radar.fields.keys():        
           TH   = 'corrected_reflectivity'
           ZDRname =  'corrected_differential_reflectivity'
           RHOHVname = 'copol_correlation_coeff'       
           PHIname = 'filtered_corrected_differential_phase'       
           KDPname = 'filtered_corrected_specific_diff_phase'
           THname =  'corrected_reflectivity'
    elif 'DBZH' in radar.fields.keys():        
           THname = 'DBZH'
           KDPname ='corrKDP'
           TVname   = 'DBZV'  
           TVname   = 'RHOHV'  
		
    nlev = 0 
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    azimuths    = radar.azimuth['data'][start_index:end_index]

    fig, axes = plt.subplots(nrows=4, ncols=3, constrained_layout=True, figsize=[13,12])

    for iz in range(len(azi_oi)):
        target_azimuth = azimuths[options['alternate_azi'][iz]]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()		
        if options['radar_name'] == 'CSPR2':
            del filas
            target_azimuth = azimuths[options['alternate_azi'][iz]]
            filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()			
        grid_lon   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_lon[:]   = np.nan
        grid_lat   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_lat[:]   = np.nan
        grid_THTH  = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_THTH[:]  = np.nan
        grid_TVTV  = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_TVTV[:]  = np.nan
        grid_ZDR   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_ZDR[:]  = np.nan
        grid_alt   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_alt[:]   = np.nan
        grid_range = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_range[:] = np.nan
        grid_RHO   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_RHO[:]   = np.nan
        grid_HID   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_HID[:]   = np.nan
        grid_KDP   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_KDP[:]  = np.nan

        # need to find x/y pair for each gate at the surface 
        for i in range(lons[filas,:].shape[2]):	
            # First, find the index of the grid point nearest a specific lat/lon.   
            abslat = np.abs(gridded_radar.point_latitude['data'][0,:,:]  - lats[filas,i])
            abslon = np.abs(gridded_radar.point_longitude['data'][0,:,:] - lons[filas,i])
            c = np.maximum(abslon, abslat)
            ([xloc], [yloc]) = np.where(c == np.min(c))	
            grid_lon[:,i]   = gridded_radar.point_longitude['data'][:,xloc,yloc]
            grid_lat[:,i]   = gridded_radar.point_latitude['data'][:,xloc,yloc]
            grid_TVTV[:,i]  = gridded_radar.fields[TVname]['data'][:,xloc,yloc]
            #grid_ZDR[:,i] = gridded_radar.fields[ZDRname]['data'][:,xloc,yloc]
            grid_ZDR[:,i] = gridded_radar.fields[THname]['data'][:,xloc,yloc]-gridded_radar.fields[TVname]['data'][:,xloc,yloc]								   
            grid_THTH[:,i]  = gridded_radar.fields[THname]['data'][:,xloc,yloc]
            grid_RHO[:,i]   = gridded_radar.fields[RHOHVname]['data'][:,xloc,yloc]
            grid_alt[:,i]   = gridded_radar.z['data'][:]
            x               = gridded_radar.point_x['data'][:,xloc,yloc]
            y               = gridded_radar.point_y['data'][:,xloc,yloc]
            z               = gridded_radar.point_z['data'][:,xloc,yloc]
            grid_range[:,i] = ( x**2 + y**2 + z**2 ) ** 0.5
            grid_KDP[:,i]   = gridded_radar.fields[KDPname]['data'][:,xloc,yloc]
            grid_HID[:,i]   = gridded_radar.fields['HID']['data'][:,xloc,yloc]

        ni = grid_HID.shape[0]
        nj = grid_HID.shape[1]
        for i in range(ni):
            rho_h = grid_RHO[i,:]
            zh_h = grid_THTH[i,:]
            for j in range(nj):
                if (rho_h[j]<0.7) or (zh_h[j]<0):
                    grid_THTH[i,j]  = np.nan
                    grid_TVTV[i,j]  = np.nan
                    grid_RHO[i,j]  = np.nan			


        #Filters
        #grid_TVTV[np.where(grid_RHO<0.6)] = np.nan	
        #grid_THTH[np.where(grid_RHO<0.6)] = np.nan	
        #grid_RHO[np.where(grid_RHO<0.6)] = np.nan	
        #grid_KDP[np.where(grid_RHO<0.6)] = np.nan	

        #--- need to add tempertaure to grided data ... 
        #radar_z = gridded_radar.point_z['data']
        #shape   = np.shape(radar_z)
        #rad_T1d = np.interp(radar_z.ravel(), alt_ref, tfield_ref)   # interpolate_sounding_to_radar(snd_T, snd_z, radar)
        #radargrid_TT = np.reshape(rad_T1d, shape)
        #gridded_radar = add_field_to_radar_object(np.reshape(rad_T1d, shape), gridded_radar, field_name='sounding_temperature')      
        #- Add height field for 4/3 propagation
        #gridded_radar = add_field_to_radar_object( gridded_radar.point_z['data'], gridded_radar, field_name = 'height')    	
        #iso0 = np.ma.mean(gridded_radar.fields['height']['data'][np.where(np.abs(gridded_radar.fields['sounding_temperature']['data']) < 0)])
        #gridded_radar.fields['height_over_iso0'] = deepcopy(gridded_radar.fields['height'])
        #gridded_radar.fields['height_over_iso0']['data'] -= iso0 
        #
        #for i in range(lons[filas,:].shape[2]):	
        #    scores          = csu_fhc.csu_fhc_summer(dz=grid_THTH[:,i], zdr=(grid_TVTV[:,i]-grid_THTH[:,i]) - options['ZDRoffset'], 
	#						 rho=grid_RHO[:,i], kdp=grid_KDP[:,i], 
        #                                    use_temp=True, band='C', T=radargrid_TT)
        #    grid_HID[:,i] = np.argmax(scores, axis=0) + 1 

        #grid_HID[np.where(grid_RHO<0.7)] = np.nan

        #---- plot hid ppi  
        hid_colors = ['White', 'LightBlue','MediumBlue', 'DarkOrange', 'LightPink',
                'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
        cmaphid = colors.ListedColormap(hid_colors)
        #cmaphid.set_bad('white')
        #cmaphid.set_under('white')
        # Figure
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        im_TH  = axes[0,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_THTH, cmap=cmap, vmin=vmin, vmax=vmax)

        #im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, (grid_THTH-grid_TVTV)-options['ZDRoffset'], cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)
        im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, (grid_ZDR)-options['ZDRoffset'], cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)

        im_RHO = axes[2,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_RHO, cmap=pyart.graph.cm.RefDiff , vmin=0.7, vmax=1.)

        im_HID = axes[3,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=0.2, vmax=10)

        axes[0,iz].set_title('coi='+titlecois[iz]) 
        if iz == 1:
            axes[0,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[1,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[2,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[3,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])

        if iz == 2:
            axes[0,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[1,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[2,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[3,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])

        if iz == 3:
            axes[0,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[1,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[2,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[3,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])

        if iz == 0:
            axes[0,0].set_ylabel('Altitude (km)')
            axes[1,0].set_ylabel('Altitude (km)')
            axes[2,0].set_ylabel('Altitude (km)')
            axes[3,0].set_ylabel('Altitude (km)')
            axes[0,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[1,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[2,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,0].set_xlabel('Range (km)')
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])

        if iz == len(azi_oi)-1: 
	    # Add colorbars #ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
            pm1    = axes[0,iz-1].get_position().get_points().flatten()
            p_last = axes[0,iz].get_position().get_points().flatten(); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.76, 0.02, 0.2])  
            cbar    = fig.colorbar(im_TH,  cax=ax_cbar, shrink=0.9,  ticks=np.arange(0,60.01,10), label='ZH')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            pm2    = axes[1,iz-1].get_position().get_points().flatten()

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.55, 0.02, 0.2])  
            cbar    = fig.colorbar(im_ZDR, cax=ax_cbar, shrink=0.9,  ticks=np.arange(-2.,5.01,1.), label='ZDR')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            pm3   = axes[2,iz-1].get_position().get_points().flatten()

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.28, 0.02, 0.2])  
            cbar    = fig.colorbar(im_RHO, cax=ax_cbar, shrink=0.9, ticks=np.arange(0.7,1.01,0.1), label='RHO')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.03, 0.02, 0.2])  
            cbar    = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            cbar = adjust_fhc_colorbar_for_pyart(cbar)
            #cbar.cmap.set_under('white')

            pm2    = axes[3,iz-1].get_position().get_points().flatten()


    #- savefile
    fig.savefig(options['fig_dir']+'PseudoRHIS_GRIDDED'+'.png', dpi=300,transparent=False)   
    #plt.close()
    del grid_THTH, grid_RHO, grid_TVTV, grid_HID

    #-------------------------------
    for i in range(20):
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
        im_HID = ax.pcolormesh(gridded_radar.point_longitude['data'][i,:,:], 
        gridded_radar.point_latitude['data'][i,:,:], gridded_radar.fields['HID']['data'][i,:,:], cmap=cmaphid, vmin=0.2, vmax=10)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)  
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title('HID '+ str( round(grid_alt[i,0]/1e3,1)) +' km')
        cbar = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')
        cbar = adjust_fhc_colorbar_for_pyart(cbar)
        ax.set_xlim([options['xlim_min'], options['xlim_max']])	
        ax.set_ylim([options['ylim_min'], options['ylim_max']])
        fig.savefig(options['fig_dir']+'RHIS_GRIDDED_verticalLevel_'+str(i)+'.png', dpi=300,transparent=False); plt.close()
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
def plot_rhi_RMA(radar, xlim_range1, xlim_range2, test_transect, ZDRoffset, freezing_lev, radar_T, options):

    radar_name = options['radar_name']
    print(radar.fields.keys())

    plt.matplotlib.rc('font', family='serif', size = 20)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['font.serif'] = ['Helvetica']
    maxalt = 15

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
    PHIDP_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); RHO_transect[:]=np.nan
    approx_altitude = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); approx_altitude[:]=np.nan
    color           = np.full((  len(radar.sweep_start_ray_index['data']), lats0.shape[1], 4), np.nan)
    gate_range      = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); gate_range[:]=np.nan
    HID_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); HID_transect[:]=np.nan
    KDP_transect      = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); KDP_transect[:]=np.nan
    alt_43aproox      = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); alt_43aproox[:]=np.nan

    azydims = lats0.shape[1]-1

    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]       
        if radar_name == 'RMA1':
            ZHZH    = radar.fields['TH']['data'][start_index:end_index]
            TV      = radar.fields['TV']['data'][start_index:end_index]
            ZDRZDR  = (ZHZH-TV)-ZDRoffset   
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]       
            PHIPHI  = radar.fields['corrPHIDP']['data'][start_index:end_index]       
            KDPKDP  = radar.fields['corrKDP']['data'][start_index:end_index]       
            HIDHID  =  radar.fields['HID']['data'][start_index:end_index]       

        elif radar_name == 'RMA5':
            ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
            if 'DBZV' in radar.fields.keys(): 
                TV     = radar.fields['DBZV']['data'][start_index:end_index]     
                ZDRZDR = ZHZH-TV   
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]    
            KDPKDP  = radar.fields['corrKDP']['data'][start_index:end_index]       
            PHIPHI  = radar.fields['corrPHIDP']['data'][start_index:end_index]       
            KDPKDP  = radar.fields['corrKDP']['data'][start_index:end_index]       
            HIDHID  =  radar.fields['HID']['data'][start_index:end_index]    
	
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
            KDPKDP  = radar.fields['corrKDP']['data'][start_index:end_index]       
            HIDHID  =  radar.fields['HID']['data'][start_index:end_index]    
            PHIPHI  = radar.fields['corrPHIDP']['data'][start_index:end_index]       
	
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
            KDPKDP  = radar.fields['corrKDP']['data'][start_index:end_index]       
            HIDHID  =  radar.fields['HID']['data'][start_index:end_index]    
	
        elif radar_name == 'CSPR2':
            ZHZH = radar.fields['corrected_reflectivity']['data'][start_index:end_index]
            TH   = radar.fields['corrected_reflectivity']['data'][start_index:end_index]
            ZDRZDR  =  radar.fields['corrected_differential_reflectivity']['data'][start_index:end_index]
            RHORHO  = radar.fields['copol_correlation_coeff']['data'][start_index:end_index]       
            PHIPHI  = radar.fields['filtered_corrected_differential_phase']['data'][start_index:end_index]       
            KDPKDP  = radar.fields['filtered_corrected_specific_diff_phase']['data'][start_index:end_index]       
            HIDHID  =  radar.fields['HID']['data'][start_index:end_index]     
            ZDRZDR[RHORHO<0.75]=np.nan
            RHORHO[RHORHO<0.75]=np.nan

        elif radar_name == 'DOW7':       
            TH   = radar.fields['TH']['data'][start_index:end_index]
            TV   = radar.fields['TV']['data'][start_index:end_index]
            ZDRZDR  = (ZHZH-TV)-ZDRoffset   
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]       
            PHIPHI  = radar.fields['corrPHIDP']['data'][start_index:end_index]       
            KDPKDP  = radar.fields['corrKDP']['data'][start_index:end_index]       
            HIDHID  =  radar.fields['HID']['data'][start_index:end_index]       
 

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

        elif radar_name == 'DOW7':
            ZHZH  = radar.fields['DBZHCC']['data'][start_index:end_index]
            TV     = radar.fields['DBZVCC']['data'][start_index:end_index]     
            ZDRZDR = radar.fields['ZDRC']['data'][start_index:end_index]      
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]
            KDPKDP  = radar.fields['KDP']['data'][start_index:end_index]   	
            HIDHID  =  radar.fields['HID']['data'][start_index:end_index]       
            PHIPHI  =  radar.fields['PHIDP']['data'][start_index:end_index]       

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
        PHIDP_transect[nlev,:]   = PHIPHI[filas,:]	
        HID_transect[nlev,:]     = HIDHID[filas,:]
        alt_43aproox[nlev,:]     = radar.fields['height']['data'][start_index:end_index][filas,:]
        [xgate, ygate, zgate]   = pyart.core.antenna_to_cartesian(gates_range[filas,:]/1e3, azimuths[filas],radar.get_elevation(nlev)[0]);
        alt_43aproox[nlev,:]    = np.ravel(radar.fields['height']['data'][start_index:end_index][filas,:])
        approx_altitude[nlev,:] = zgate/1e3;
        gate_range[nlev,:]      = gates_range[filas,:]/1e3;
	#
        #scores          = csu_fhc.csu_fhc_summer(dz=Ze_transect[nlev,:], zdr=ZDR_transect[nlev,:], 
        #                                     rho=RHO_transect[nlev,:], kdp=KDP_transect[nlev,:], 
        #                                     use_temp=True, band='C', T=radar_T)
        #HID_transect[nlev,:]  = np.argmax(scores, axis=0) + 1 
    #---------------------------------------- REFLECTIVITY
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(111)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude, Ze_transect, cmap=colormaps('ref'), vmin=0, vmax=60)
    plt.close()

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
    fig2, axes = plt.subplots(nrows=6,ncols=1,constrained_layout=True,figsize=[10,20]) 

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
         axes[0].set_ylim([0, maxalt])
         axes[0].set_ylabel('Altitude (km)')
         axes[0].grid()
         axes[0].set_xlim((xlim_range1, xlim_range2))
         norm = matplotlib.colors.Normalize(vmin=0.,vmax=60.)
         cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormaps('ref'))
         cax.set_array(Ze_transect)
         cbar_z = fig2.colorbar(cax, ax=axes[0], shrink=1.1, ticks=np.arange(0,60.01,10), label='Zh (dBZ)')
         axes[0].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, Ze_transect
    axes[0].set_title('Transect Nr. '+str(test_transect))
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
        axes[1].set_ylim([0, maxalt])
        axes[1].set_ylabel('Altitude (km)')
        axes[1].grid()
        axes[1].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=-2.,vmax=5.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_ZDR)
        cax.set_array(ZDR_transect)
        cbar_zdr = fig2.colorbar(cax, ax=axes[1], shrink=1.1, ticks=np.arange(-2.,5.01,1.), label='ZDR')     
        axes[1].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, ZDR_transect
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
        axes[2].set_ylim([0, maxalt])
        axes[2].set_ylabel('Altitude (km)')
        axes[2].grid()
        axes[2].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0.7,vmax=1.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.RefDiff)
        cax.set_array(RHO_transect)
        cbar_rho = fig2.colorbar(cax, ax=axes[2], shrink=1.1, ticks=np.arange(0.7,1.01,0.1), label=r'$\rho_{hv}$')     
        axes[2].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, RHO_transect

    #---------------------------------------- PHIDP
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                PHIDP_transect,
                cmap = pyart.graph.cm.Wild25, vmin=0, vmax=360.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=PHIDP_transect[nlev,:],
                cmap= pyart.graph.cm.Wild25, vmin=0, vmax=360.)
        color[nlev,:,:] = sc.to_rgba(PHIDP_transect[nlev,:])   # pyart.graph.cm.RefDiff
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
        axes[3].set_ylim([0, maxalt])
        axes[3].set_ylabel('Altitude (km)')
        axes[3].grid()
        axes[3].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0,vmax=360.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.Wild25)
        cax.set_array(PHIDP_transect)
        cbar_Phidp = fig2.colorbar(cax, ax=axes[3], shrink=1.1, label=r'PHIDP')     
        axes[3].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, PHIDP_transect



    #---------------------------------------- KDP
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                KDP_transect,
                cmap = pyart.graph.cm.Theodore16, vmin=0, vmax=5.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=KDP_transect[nlev,:],
                cmap= pyart.graph.cm.Theodore16, vmin=0, vmax=5.)
        color[nlev,:,:] = sc.to_rgba(KDP_transect[nlev,:])   # pyart.graph.cm.RefDiff
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
            axes[4].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[4].set_ylim([0, maxalt])
        axes[4].set_ylabel('Altitude (km)')
        axes[4].grid()
        axes[4].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0,vmax=5.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.Theodore16)
        cax.set_array(KDP_transect)
        cbar_Phidp = fig2.colorbar(cax, ax=axes[4], shrink=1.1, label=r'KDP')     
        axes[4].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, KDP_transect








    #---------------------------------------- HID
    hid_colors = ['White', 'LightBlue','MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)
    #cmaphid.set_bad('white')
    #cmaphid.set_under('white')

    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                HID_transect,
                cmap = cmaphid, vmin=0.2, vmax=10)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=HID_transect[nlev,:],
                cmap = cmaphid, vmin=0.2, vmax=10)
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
            axes[5].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[5].set_ylim([0, maxalt])
        axes[5].set_ylabel('Altitude (km)')
        axes[5].grid()
        axes[5].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0.2,vmax=10)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmaphid)
        cax.set_array(HID_transect)
        cbar_HID = fig2.colorbar(cax, ax=axes[5], shrink=1.1, label=r'HID')    
        cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	
        axes[5].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)
    del mycolorbar, x, y, P1, inter, LS, HID_transect

    #- savefile
    fig2.savefig(options['fig_dir']+'PseudoRHI_'+'Transect_'+str(test_transect)+'.png', dpi=300,transparent=False)   
    #plt.close()


    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def stack_ppis(radar, files_list, options, freezing_lev, radar_T, tfield_ref, alt_ref): 
	
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
            #
            radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
            radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
            dzh_  = radar.fields['DBZHCC']['data'].copy()
            dzv_  = radar.fields['DBZVCC']['data'].copy()
            dZDR  = radar.fields['ZDRC']['data'].copy()
            drho_ = radar.fields['RHOHV']['data'].copy()
            dkdp_ = radar.fields['KDP']['data'].copy()
            dphi_ = radar.fields['PHIDP']['data'].copy()
            # Filters
            ni = dzh_.shape[0]
            nj = dzh_.shape[1]
            for i in range(ni):
                rho_h = drho_[i,:]
                zh_h = dzh_[i,:]
                for j in range(nj):
                    if (rho_h[j]<0.7) or (zh_h[j]<30):
                        dzh_[i,j]  = np.nan
                        dzv_[i,j]  = np.nan
                        drho_[i,j]  = np.nan
                        dkdp_[i,j]  = np.nan
                        dphi_[i,j]  = np.nan
            scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=dZDR - options['ZDRoffset'], rho=drho_, kdp=dkdp_, use_temp=True, band='C', T=radar_T)            
            HIDHID = np.argmax(scores, axis=0) + 1 
	    #
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
                Ze[nlev,TransectNo,:]    = dzh_[TransectNo,:]
                ZDR[nlev,TransectNo,:]   = dZDR[TransectNo,:]  
                RHO[nlev,TransectNo,:]   = drho_[TransectNo,:]
                PHIDP[nlev,TransectNo,:] = dphi_[TransectNo,:]
                HID[nlev,TransectNo,:]   = HIDHID[TransectNo,:] 
                KDP[nlev,TransectNo,:]   = dkdp_[TransectNo,:]
                lon[nlev,TransectNo,:]   = lons[TransectNo,:]   
                lat[nlev,TransectNo,:]   = lats[TransectNo,:] 

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
    HID_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); HID_all[:]=np.nan  
    ii = 0
    for i in range(azy.shape[0]):
        for j in range(Ze.shape[1]):
            Ze_all[ii,:]    = Ze[i,j,:]
            ZDR_all[ii,:]   = ZDR[i,j,:]
            RHOHV_all[ii,:] = RHO[i,j,:]
            PHIDP_all[ii,:] = PHIDP[i,j,:]
            KDP_all[ii,:] = KDP[i,j,:]
            HID_all[ii,:] = HID[i,j,:]
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
    
    #- HID
    ref_dict_HID = get_metadata('HID')
    ref_dict_HID['data'] = np.array(HID_all)
	
    radar_stack.fields = {'DBZHCC': ref_dict_ZH, 
			  'ZDRC':   ref_dict_ZDR,
			  'RHOHV':  ref_dict_RHOHV,
			  'PHIDP':  ref_dict_PHIDP,
			  'KDP':    ref_dict_KDP,
			  'HID':    ref_dict_HID}
			  

		
    return radar_stack
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------	
#------------------------------------------------------------------------------
def stack_ppis_CSPR(radar, options, freezing_lev, radar_T, tfield_ref, alt_ref): 

    #- HERE MAKE PPIS SIMILAR TO RMA1S ... ? to achive the gridded field ... 
    #- Radar sweep
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]	
    lats0        = radar.gate_latitude['data'][start_index:end_index]
    lons0        = radar.gate_longitude['data'][start_index:end_index]
    files_list =  radar.fixed_angle['data']         
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
    alt_43aproox    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); alt_43aproox[:]=np.nan
    #
    gate_range      = np.zeros( [len(files_list), lats0.shape[1] ]); gate_range[:]=np.nan
    azy   = np.zeros( [len(files_list), lats0.shape[0] ]); azy[:]=np.nan
    fixed_angle     = np.zeros( [len(files_list)] ); fixed_angle[:]=np.nan
    #
    nlev = 0
    for file in files_list:
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]	
        fixed_angle[nlev] = radar.fixed_angle['data'].data[nlev]
        azy[nlev,:]  = radar.azimuth['data'][start_index:end_index]
        azimuths     = radar.azimuth['data'][start_index:end_index]
        ZHZH    = radar.fields['corrected_reflectivity']['data'][start_index:end_index]
        ZDRZDR  = radar.fields['corrected_differential_reflectivity']['data'][start_index:end_index] 
        RHORHO  = radar.fields['copol_correlation_coeff']['data'][start_index:end_index]
        KDPKDP  = radar.fields['filtered_corrected_specific_diff_phase']['data'][start_index:end_index]  	
        PHIPHI  = radar.fields['filtered_corrected_differential_phase']['data'][start_index:end_index]   
        #
        radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
        radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
        dzh_  = radar.fields['corrected_reflectivity']['data'].copy()
        dZDR  = radar.fields['corrected_differential_reflectivity']['data'].copy()
        drho_ = radar.fields['copol_correlation_coeff']['data'].copy()
        dkdp_ = radar.fields['filtered_corrected_specific_diff_phase']['data'].copy()
        dphi_ = radar.fields['filtered_corrected_differential_phase']['data'].copy()
        # Filters
        ni = dzh_.shape[0]
        nj = dzh_.shape[1]
        for i in range(ni):
            rho_h = drho_[i,:]
            zh_h = dzh_[i,:]
            for j in range(nj):
                if (rho_h[j]<0.7) or (zh_h[j]<30):
                    dzh_[i,j]  = np.nan
                    drho_[i,j]  = np.nan
                    dkdp_[i,j]  = np.nan
                    dphi_[i,j]  = np.nan
        scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=dZDR - options['ZDRoffset'], rho=drho_, kdp=dkdp_, use_temp=True, band='C', T=radar_T)            
        HIDHID = np.argmax(scores, axis=0) + 1 
        gateZ    = radar.gate_z['data']
        gateX    = radar.gate_x['data']
        gateY    = radar.gate_y['data']
        gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
        #
        lats        = radar.gate_latitude['data']
        lons        = radar.gate_longitude['data']	     
        #------- aca hay que stack correctamente por azimuths? 
        for TransectNo in range(lats0.shape[0]): 
            [xgate, ygate, zgate]    = pyart.core.antenna_to_cartesian(gates_range[TransectNo,:]/1e3, azimuths[TransectNo], fixed_angle[nlev] );       
            Ze[nlev,TransectNo,:]    = dzh_[TransectNo,:]
            ZDR[nlev,TransectNo,:]   = dZDR[TransectNo,:]  
            RHO[nlev,TransectNo,:]   = drho_[TransectNo,:]
            PHIDP[nlev,TransectNo,:] = dphi_[TransectNo,:]
            HID[nlev,TransectNo,:]   = HIDHID[TransectNo,:] 
            KDP[nlev,TransectNo,:]   = dkdp_[TransectNo,:]
            lon[nlev,TransectNo,:]   = lons[TransectNo,:]   
            lat[nlev,TransectNo,:]   = lats[TransectNo,:] 

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
    HID_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); HID_all[:]=np.nan  
    ii = 0
    for i in range(azy.shape[0]):
        for j in range(Ze.shape[1]):
            Ze_all[ii,:]    = Ze[i,j,:]
            ZDR_all[ii,:]   = ZDR[i,j,:]
            RHOHV_all[ii,:] = RHO[i,j,:]
            PHIDP_all[ii,:] = PHIDP[i,j,:]
            KDP_all[ii,:] = KDP[i,j,:]
            HID_all[ii,:] = HID[i,j,:]
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

    #- HID
    ref_dict_HID = get_metadata('HID')
    ref_dict_HID['data'] = np.array(HID_all)

    radar_stack.fields = {'DBZHCC': ref_dict_ZH, 
			  'ZDRC':   ref_dict_ZDR,
			  'RHOHV':  ref_dict_RHOHV,
			  'PHIDP':  ref_dict_PHIDP,
			  'KDP':    ref_dict_KDP,
			  'HID':    ref_dict_HID}



    return radar_stack
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------	

def plot_rhi_DOW7(radar, files_list, xlim_range1, xlim_range2, test_transect, ZDRoffset, freezing_lev, radar_T, options, tfield_ref, alt_ref):

    radar_name = options['radar_name']
    print(radar.fields.keys())

    plt.matplotlib.rc('font', family='serif', size = 20)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['font.serif'] = ['Helvetica']

    #- Radar sweep
    lats0        = radar.gate_latitude['data']
    lons0        = radar.gate_longitude['data']
    azimuths     = radar.azimuth['data']
			
    Ze_transect     = np.zeros( [len(files_list), lats0.shape[1] ]); Ze_transect[:]=np.nan
    ZDR_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); ZDR_transect[:]=np.nan
    PHI_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); PHI_transect[:]=np.nan
    lon_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); lon_transect[:]=np.nan
    lat_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); lat_transect[:]=np.nan
    RHO_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); RHO_transect[:]=np.nan
    PHIDP_transect  = np.zeros( [len(files_list), lats0.shape[1] ]); RHO_transect[:]=np.nan
    approx_altitude = np.zeros( [len(files_list), lats0.shape[1] ]); approx_altitude[:]=np.nan
    color           = np.full(( len(files_list), lats0.shape[1], 4), np.nan)
    gate_range      = np.zeros( [len(files_list), lats0.shape[1] ]); gate_range[:]=np.nan
    HID_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); HID_transect[:]=np.nan
    KDP_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); KDP_transect[:]=np.nan
    alt_43aproox    = np.zeros( [len(files_list), lats0.shape[1] ]); alt_43aproox[:]=np.nan
	
    azydims = lats0.shape[1]-1

    nlev = 0
    for file in files_list:
      if 'low_v176' in file:
        radar   = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/DOW7/'+file) 
        ZHZH    = radar.fields['DBZHCC']['data']
        TV      = radar.fields['DBZVCC']['data']   
        ZDRZDR  = radar.fields['ZDRC']['data']      
        RHORHO  = radar.fields['RHOHV']['data']
        KDPKDP  = radar.fields['KDP']['data']  	
        PHIPHI  = radar.fields['PHIDP']['data']   
        #
        radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
        radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
        dzh_  = radar.fields['DBZHCC']['data'].copy()
        dzv_  = radar.fields['DBZVCC']['data'].copy()
        dZDR  = radar.fields['ZDRC']['data'].copy()
        drho_ = radar.fields['RHOHV']['data'].copy()
        dkdp_ = radar.fields['KDP']['data'].copy()
        # dkdp_[np.where(drho_.data==radar.fields['RHOHV']['data'].fill_value)] = np.nan
        # Filters
        ni = dzh_.shape[0]
        nj = dzh_.shape[1]
        for i in range(ni):
            rho_h = drho_[i,:]
            zh_h = dzh_[i,:]
            for j in range(nj):
                if (rho_h[j]<0.7) or (zh_h[j]<30):
                    dzh_[i,j]  = np.nan
                    dzv_[i,j]  = np.nan
                    drho_[i,j]  = np.nan
                    dkdp_[i,j]  = np.nan
        scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=dZDR - options['ZDRoffset'], 
					     rho=drho_, kdp=dkdp_, 
                         use_temp=True, band='C', T=radar_T)
        HIDHID = np.argmax(scores, axis=0) + 1 
        radar.add_field_like('KDP','HID', HIDHID, replace_existing=True)
	
        lats        = radar.gate_latitude['data']
        lons        = radar.gate_longitude['data']
        # En verdad buscar azimuth no transecta ... 
        azimuths    = radar.azimuth['data']
        # ojo que esto no se si aplica ... 
        TransectNo = test_transect
	#target_azimuth = azimuths[test_transect]  #- target azimuth for nlev=0 test case is 301.5
        #filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect[nlev,:]     = lons[TransectNo,:]
        lat_transect[nlev,:]     = lats[TransectNo,:]
        #
        gateZ    = radar.gate_z['data']
        gateX    = radar.gate_x['data']
        gateY    = radar.gate_y['data']
        gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
        #
        Ze_transect[nlev,:]      = ZHZH[TransectNo,:]
        ZDR_transect[nlev,:]     = ZDRZDR[TransectNo,:]
        RHO_transect[nlev,:]     = RHORHO[TransectNo,:]
        KDP_transect[nlev,:]     = KDPKDP[TransectNo,:]	
        PHIDP_transect[nlev,:]   = PHIPHI[TransectNo,:]	
        HID_transect[nlev,:]     = HIDHID[TransectNo,:]	
        #alt_43aproox[nlev,:]     = radar.fields['height']['data'][filas,:]
        [xgate, ygate, zgate]   = pyart.core.antenna_to_cartesian(gates_range[TransectNo,:]/1e3, azimuths[TransectNo], np.double(file[41:45]) );
	# eso del paper de granizo: [xgate, ygate, zgate] = pyart.core.antenna_to_cartesian(gate_range[TransectNo,:]/1e3, azimuths[TransectNo], 
        #alt_43aproox[nlev,:]    = np.ravel(radar.fields['height']['data'][filas,:])
        approx_altitude[nlev,:] = zgate/1e3;
        gate_range[nlev,:]      = gates_range[TransectNo,:]/1e3;
        nlev = nlev + 1
	#
        #scores          = csu_fhc.csu_fhc_summer(dz=Ze_transect[nlev,:], zdr=ZDR_transect[nlev,:], 
        #                                     rho=RHO_transect[nlev,:], kdp=KDP_transect[nlev,:], 
        #                                     use_temp=True, band='C', T=radar_T)
        #HID_transect[nlev,:]  = np.argmax(scores, axis=0) + 1 
    #---------------------------------------- REFLECTIVITY
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(111)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude, Ze_transect, cmap=colormaps('ref'), vmin=0, vmax=60)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons (scatter plot para sacar el color de cada pixel)
    print(len(radar.sweep_start_ray_index['data']))
    for nlev in range(len(files_list)):
         fig = plt.figure(figsize=[30,10])
         fig.add_subplot(221)
         sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                 s=1,c=Ze_transect[nlev,:],
                 cmap=colormaps('ref'), vmin=0, vmax=60)
         color[nlev,:,:] = sc.to_rgba(Ze_transect[nlev,:])
         plt.close()

    #- Try polygons
    fig2, axes = plt.subplots(nrows=6,ncols=1,constrained_layout=True,figsize=[10,20]) 

    fig1 = plt.figure(figsize=(20,20))
    for nlev in range(len(files_list)):
         if nlev > 10: continue
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
    
    del mycolorbar, x, y, P1, inter, LS, Ze_transect
    axes[0].set_title('Transect Nr. '+str(test_transect))
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
    for nlev in range(len(files_list)):
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
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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

    del mycolorbar, x, y, P1, inter, LS, ZDR_transect
    #---------------------------------------- RHOHV
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                RHO_transect,
                cmap = pyart.graph.cm.RefDiff, vmin=0.7, vmax=1.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(files_list)):
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
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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

    del mycolorbar, x, y, P1, inter, LS, RHO_transect

    #---------------------------------------- PHIDP
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                PHIDP_transect,
                cmap = pyart.graph.cm.Wild25, vmin=0, vmax=360.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(files_list)):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=PHIDP_transect[nlev,:],
                cmap= pyart.graph.cm.Wild25, vmin=0, vmax=360.)
        color[nlev,:,:] = sc.to_rgba(PHIDP_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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
        norm = matplotlib.colors.Normalize(vmin=0,vmax=360.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.Wild25)
        cax.set_array(PHIDP_transect)
        cbar_Phidp = fig2.colorbar(cax, ax=axes[3], shrink=1.1, label=r'PHIDP')     
        axes[3].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, PHIDP_transect



    #---------------------------------------- KDP
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                KDP_transect,
                cmap = pyart.graph.cm.Theodore16, vmin=0, vmax=5.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(files_list)):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=KDP_transect[nlev,:],
                cmap= pyart.graph.cm.Theodore16, vmin=0, vmax=5.)
        color[nlev,:,:] = sc.to_rgba(KDP_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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
            axes[4].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[4].set_ylim([0, 20])
        axes[4].set_ylabel('Altitude (km)')
        axes[4].grid()
        axes[4].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0,vmax=5.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.Theodore16)
        cax.set_array(KDP_transect)
        cbar_Phidp = fig2.colorbar(cax, ax=axes[4], shrink=1.1, label=r'KDP')     
        axes[4].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, KDP_transect








    #---------------------------------------- HID
    hid_colors = ['White', 'LightBlue','MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)
    #cmaphid.set_bad('white')
    #cmaphid.set_under('white')

    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                HID_transect,
                cmap = cmaphid, vmin=0.2, vmax=10)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(files_list)):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=HID_transect[nlev,:],
                cmap = cmaphid, vmin=0.2, vmax=10)
        color[nlev,:,:] = sc.to_rgba(HID_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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
            axes[5].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[5].set_ylim([0, 20])
        axes[5].set_ylabel('Altitude (km)')
        axes[5].grid()
        axes[5].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0.2,vmax=10)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmaphid)
        cax.set_array(HID_transect)
        cbar_HID = fig2.colorbar(cax, ax=axes[5], shrink=1.1, label=r'HID')    
        cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	
        axes[5].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)
    del mycolorbar, x, y, P1, inter, LS, HID_transect

    #- savefile
    fig2.savefig(options['fig_dir']+'PseudoRHI_'+'Transect_'+str(test_transect)+'.png', dpi=300,transparent=False)   
    #plt.close()
	
	
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------	
def plot_rhi_CSPR2(radar, xlim_range1, xlim_range2, test_transect, ZDRoffset, freezing_lev, radar_T, options, tfield_ref, alt_ref):

    radar_name = options['radar_name']
    print(radar.fields.keys())

    plt.matplotlib.rc('font', family='serif', size = 20)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['font.serif'] = ['Helvetica']

    #- Radar sweep
    lats0        = radar.gate_latitude['data']
    lons0        = radar.gate_longitude['data']
    azimuths     = radar.azimuth['data']

    files_list = radar.fixed_angle['data']
			
    Ze_transect     = np.zeros( [len(files_list), lats0.shape[1] ]); Ze_transect[:]=np.nan
    ZDR_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); ZDR_transect[:]=np.nan
    PHI_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); PHI_transect[:]=np.nan
    lon_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); lon_transect[:]=np.nan
    lat_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); lat_transect[:]=np.nan
    RHO_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); RHO_transect[:]=np.nan
    PHIDP_transect  = np.zeros( [len(files_list), lats0.shape[1] ]); RHO_transect[:]=np.nan
    approx_altitude = np.zeros( [len(files_list), lats0.shape[1] ]); approx_altitude[:]=np.nan
    color           = np.full(( len(files_list), lats0.shape[1], 4), np.nan)
    gate_range      = np.zeros( [len(files_list), lats0.shape[1] ]); gate_range[:]=np.nan
    HID_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); HID_transect[:]=np.nan
    KDP_transect    = np.zeros( [len(files_list), lats0.shape[1] ]); KDP_transect[:]=np.nan
    alt_43aproox    = np.zeros( [len(files_list), lats0.shape[1] ]); alt_43aproox[:]=np.nan
	
    azydims = lats0.shape[1]-1

    ZHZH    = radar.fields['corrected_reflectivity']['data']
    ZDRZDR  = radar.fields['corrected_differential_reflectivity']['data']      
    RHORHO  = radar.fields['copol_correlation_coeff']['data']
    KDPKDP  = radar.fields['filtered_corrected_specific_diff_phase']['data']  	
    PHIPHI  = radar.fields['filtered_corrected_differential_phase']['data']   
    #
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    dzh_  = radar.fields['corrected_reflectivity']['data'].copy()
    dZDR  = radar.fields['corrected_differential_reflectivity']['data'].copy()
    drho_ = radar.fields['copol_correlation_coeff']['data'].copy()
    dkdp_ = radar.fields['filtered_corrected_specific_diff_phase']['data'].copy()
    # Filters
    ni = dzh_.shape[0]
    nj = dzh_.shape[1]
    for i in range(ni):
         rho_h = drho_[i,:]
         zh_h = dzh_[i,:]
         for j in range(nj):
             if (rho_h[j]<0.7) or (zh_h[j]<30):
                  dzh_[i,j]  = np.nan
                  dZDR[i,j]  = np.nan
                  drho_[i,j]  = np.nan
                  dkdp_[i,j]  = np.nan
    scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=dZDR - options['ZDRoffset'], 
					     rho=drho_, kdp=dkdp_, use_temp=True, band='C', T=radar_T)
    HIDHID = np.argmax(scores, axis=0) + 1 
    radar.add_field_like('copol_correlation_coeff','HID', HIDHID, replace_existing=True)
	
    files_list = radar.fixed_angle['data']
    lats        = radar.gate_latitude['data']
    lons        = radar.gate_longitude['data']
    for nlev in range(len(files_list)):
        fig = plt.figure()
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        pcm1 = plt.pcolormesh(lons[start_index:end_index], lats[start_index:end_index], radar.fields['corrected_reflectivity']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(pcm1, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
        cbar.cmap.set_under(under)
        azimuths = radar.azimuth['data'][start_index:end_index]
        TransectNo = np.nanmin(np.asarray(abs(azimuths-test_transect)<=0.5).nonzero())
        lon_transect[nlev,:]     = lons[start_index:end_index][TransectNo,:]
        lat_transect[nlev,:]     = lats[start_index:end_index][TransectNo,:]      
        plt.plot(lon_transect[nlev,:], lat_transect[nlev,:], '-k')	
        #
        gateZ    = radar.gate_z['data'][start_index:end_index]
        gateX    = radar.gate_x['data'][start_index:end_index]
        gateY    = radar.gate_y['data'][start_index:end_index]
        gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
        #
        Ze_transect[nlev,:]      = ZHZH[start_index:end_index][TransectNo,:]
        ZDR_transect[nlev,:]     = ZDRZDR[start_index:end_index][TransectNo,:]
        RHO_transect[nlev,:]     = RHORHO[start_index:end_index][TransectNo,:]
        KDP_transect[nlev,:]     = KDPKDP[start_index:end_index][TransectNo,:]	
        PHIDP_transect[nlev,:]   = PHIPHI[start_index:end_index][TransectNo,:]	
        HID_transect[nlev,:]     = HIDHID[start_index:end_index][TransectNo,:]	
        #alt_43aproox[nlev,:]     = radar.fields['height']['data'][filas,:]
        [xgate, ygate, zgate]   = pyart.core.antenna_to_cartesian(gates_range[TransectNo,:]/1e3, azimuths[TransectNo], files_list[nlev] );
	# eso del paper de granizo: [xgate, ygate, zgate] = pyart.core.antenna_to_cartesian(gate_range[TransectNo,:]/1e3, azimuths[TransectNo], 
        #alt_43aproox[nlev,:]    = np.ravel(radar.fields['height']['data'][filas,:])
        approx_altitude[nlev,:] = zgate/1e3;
        gate_range[nlev,:]      = gates_range[TransectNo,:]/1e3;
	#
        #scores          = csu_fhc.csu_fhc_summer(dz=Ze_transect[nlev,:], zdr=ZDR_transect[nlev,:], 
        #                                     rho=RHO_transect[nlev,:], kdp=KDP_transect[nlev,:], 
        #                                     use_temp=True, band='C', T=radar_T)
        #HID_transect[nlev,:]  = np.argmax(scores, axis=0) + 1 
    #---------------------------------------- REFLECTIVITY
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(111)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude, Ze_transect, cmap=colormaps('ref'), vmin=0, vmax=60)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons (scatter plot para sacar el color de cada pixel)
    print(len(radar.sweep_start_ray_index['data']))
    for nlev in range(len(files_list)):
         fig = plt.figure(figsize=[30,10])
         fig.add_subplot(221)
         sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                 s=1,c=Ze_transect[nlev,:],
                 cmap=colormaps('ref'), vmin=0, vmax=60)
         color[nlev,:,:] = sc.to_rgba(Ze_transect[nlev,:])
         plt.close()

    #- Try polygons
    fig2, axes = plt.subplots(nrows=6,ncols=1,constrained_layout=True,figsize=[10,20]) 

    fig1 = plt.figure(figsize=(20,20))
    for nlev in range(len(files_list)):
         if nlev > 10: continue
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
    
    del mycolorbar, x, y, P1, inter, LS, Ze_transect
    axes[0].set_title('Transect Nr. '+str(test_transect))
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
    for nlev in range(len(files_list)):
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
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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

    del mycolorbar, x, y, P1, inter, LS, ZDR_transect
    #---------------------------------------- RHOHV
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                RHO_transect,
                cmap = pyart.graph.cm.RefDiff, vmin=0.7, vmax=1.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(files_list)):
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
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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

    del mycolorbar, x, y, P1, inter, LS, RHO_transect

    #---------------------------------------- PHIDP
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                PHIDP_transect,
                cmap = pyart.graph.cm.Wild25, vmin=0, vmax=360.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(files_list)):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=PHIDP_transect[nlev,:],
                cmap= pyart.graph.cm.Wild25, vmin=0, vmax=360.)
        color[nlev,:,:] = sc.to_rgba(PHIDP_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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
        norm = matplotlib.colors.Normalize(vmin=0,vmax=360.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.Wild25)
        cax.set_array(PHIDP_transect)
        cbar_Phidp = fig2.colorbar(cax, ax=axes[3], shrink=1.1, label=r'PHIDP')     
        axes[3].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, PHIDP_transect



    #---------------------------------------- KDP
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                KDP_transect,
                cmap = pyart.graph.cm.Theodore16, vmin=0, vmax=5.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(files_list)):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=KDP_transect[nlev,:],
                cmap= pyart.graph.cm.Theodore16, vmin=0, vmax=5.)
        color[nlev,:,:] = sc.to_rgba(KDP_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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
            axes[4].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[4].set_ylim([0, 20])
        axes[4].set_ylabel('Altitude (km)')
        axes[4].grid()
        axes[4].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0,vmax=5.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.Theodore16)
        cax.set_array(KDP_transect)
        cbar_Phidp = fig2.colorbar(cax, ax=axes[4], shrink=1.1, label=r'KDP')     
        axes[4].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, P1, inter, LS, KDP_transect








    #---------------------------------------- HID
    hid_colors = ['White', 'LightBlue','MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)
    #cmaphid.set_bad('white')
    #cmaphid.set_under('white')

    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                HID_transect,
                cmap = cmaphid, vmin=0.2, vmax=10)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(files_list)):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=HID_transect[nlev,:],
                cmap = cmaphid, vmin=0.2, vmax=10)
        color[nlev,:,:] = sc.to_rgba(HID_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(files_list)):
        if nlev > 10: continue
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
            axes[5].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[5].set_ylim([0, 20])
        axes[5].set_ylabel('Altitude (km)')
        axes[5].grid()
        axes[5].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0.2,vmax=10)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmaphid)
        cax.set_array(HID_transect)
        cbar_HID = fig2.colorbar(cax, ax=axes[5], shrink=1.1, label=r'HID')    
        cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	
        axes[5].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)
    del mycolorbar, x, y, P1, inter, LS, HID_transect

    #- savefile
    fig2.savefig(options['fig_dir']+'PseudoRHI_'+'Transect_'+str(test_transect)+'.png', dpi=300,transparent=False)   
    #plt.close()
	
	
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


def despeckle_phidp(phi, rho, zh):
    '''
    Elimina pixeles aislados de PhiDP
    '''

    # Unmask data and despeckle
    dphi = phi.copy()
    
    # Descartamos pixeles donde RHO es menor que un umbral (e.g., 0.7) o no está definido (e.g., NaN)
    dphi[np.isnan(rho)] = np.nan
    
    # Calculamos la textura de RHO (rhot) y descartamos todos los pixeles de PHIDP por encima
    # de un umbral de rhot (e.g., 0.25)
    rhot = wrl.dp.texture(rho)
    rhot_thr = 0.25
    dphi[rhot > rhot_thr] = np.nan
    
    # Eliminamos pixeles aislados rodeados de NaNs
    # https://docs.wradlib.org/en/stable/generated/wradlib.dp.linear_despeckle.html     
    dphi = wrl.dp.linear_despeckle(dphi, ndespeckle=5, copy=False)
	
    ni = phi.shape[0]
    nj = phi.shape[1]

    #for i in range(ni):
    #    rho_h = rho[i,:]
    #    zh_h = zh[i,:]
    #    for j in range(nj):
    #        if (rho_h[j]<0.7) or (zh_h[j]<30):
    #            dphi[i,j]  = np.nan		
	
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

            elif a > diferencia:  # np.abs(a) ? 
                v2[l] = v1[l] + 360 # para -180to180 1ue le sumo 360, aca v1[l]-360
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

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------    
def check_increasing(A):
  
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def correct_phidp(phi, rho_data, zh, sys_phase, diferencia):

    phiphi = phi.copy()
    rho = rho_data.copy()
    ni = phi.shape[0]
    nj = phi.shape[1]
    for i in range(ni):
        rho_h = rho[i,:]
        zh_h = zh[i,:]
        for j in range(nj):
            if (rho_h[j]<0.7) or (zh_h[j]<30):
                phiphi[i,j]  = np.nan 
                rho[i,j]     = np.nan 
    phiphi[:,0:20]  = np.nan 
    rho[:,0:20]    = np.nan 
	
    dphi = despeckle_phidp(phiphi, rho, zh)
    uphi_i = unfold_phidp(dphi, rho, diferencia) 
    uphi_accum = [] 	
    for i in range(ni):
        phi_h = uphi_i[i,:]
        for j in range(1,nj-1,1):
            if phi_h[j] <= np.nanmax(np.fmax.accumulate(phi_h[0:j])): 
              	uphi_i[i,j] = uphi_i[i,j-1] 

    # Reemplazo nan por sys_phase para que cuando reste esos puntos queden en cero <<<<< ojo aca! 
    uphi = uphi_i.copy()
    uphi = np.where(np.isnan(uphi), sys_phase, uphi)
    phi_cor = subtract_sys_phase(uphi, sys_phase)
    # phi_cor[rho<0.7] = np.nan
    phi_cor[phi_cor < 0] = np.nan #antes <= ? 
    phi_cor[np.isnan(phi_cor)] = 0 #agregado para RMA1?

    # Smoothing final:
    for i in range(ni):
        phi_cor[i,:] = pyart.correct.phase_proc.smooth_and_trim(phi_cor[i,:], window_len=20,
                                            window='flat')
    return dphi, uphi_i, phi_cor

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

    radar.add_field_like('RHOHV','NEW_kdp',bb, replace_existing=True)
	
    return radar 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_HID_PPI_CSPR2(radar, options, nlev, azimuth_ray, diff_value, tfield_ref, alt_ref):

    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]

    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    rhoHV = radar.fields['copol_correlation_coeff']['data'][start_index:end_index]
    PHIDP = radar.fields['filtered_corrected_differential_phase']['data'][start_index:end_index]
    KDP   = radar.fields['filtered_corrected_specific_diff_phase']['data'][start_index:end_index]
    RHIs_nlev = radar.fields['HID']['data'][start_index:end_index]
    ZHFIELD = 'corrected_reflectivity'
    ZH =  radar.fields[ZHFIELD]['data'][start_index:end_index]

    #---- plot hid ppi  
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid       = colors.ListedColormap(hid_colors)

    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
    pcm1 = axes.pcolormesh(lons, lats, RHIs_nlev, cmap = cmaphid, vmin=0.2, vmax=10)
    axes.set_title('HID nlev '+str(nlev)+' PPI (at '+str(options['time_pfs'])+')')
    axes.set_xlim([options['xlim_min'], options['xlim_max']])
    axes.set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    cbar_HID = plt.colorbar(pcm1, ax=axes, shrink=1.1, label=r'HID')    
    cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	



    # agregar: 	
    # read 
    f = h5py.File( options['gmi_dir']+options['gfile'], 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    for j in range(lon_gmi.shape[1]):  # ANTES ERA +-5
            tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+10),:] = np.nan
            tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-10),:] = np.nan   
            lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+10),:] = np.nan
            lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-10),:] = np.nan  
            lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+10),:] = np.nan
            lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-10),:] = np.nan  	
    PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 	
    CS = axes.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 225], colors=(['black', 'gray']), linewidths=1.5)
    labels_cont = ['GMI 200K contour', 'GMI 225K contour']
    for i in range(len(labels_cont)):
      CS.collections[i].set_label(labels_cont[i])
    if len(options['REPORTES_meta'])>0:
    	for ireportes in range(len(options['REPORTES_geo'])):
        	axes.plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], '*', markeredgecolor='black', markerfacecolor='black', markersize=10, label=options['REPORTES_meta'][ireportes])
    	plt.legend() 

    fig.savefig(options['fig_dir']+'PPIs_HID'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)   
    #plt.close()

    return
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_HID_PPI(radar, options, nlev, azimuth_ray, diff_value, tfield_ref, alt_ref):

    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]

    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    rhoHV = radar.fields['RHOHV']['data'][start_index:end_index]
    PHIDP = radar.fields['PHIDP']['data'][start_index:end_index]
    if 'KDP' in radar.fields.keys():  
        KDP   = radar.fields['KDP']['data'][start_index:end_index]
    RHIs_nlev = radar.fields['HID']['data'][start_index:end_index]
    if 'TH' in radar.fields.keys():  
        ZHFIELD = 'TH'
        ZH =  radar.fields['TH']['data'][start_index:end_index]
    elif 'DBZHCC' in radar.fields.keys():
        ZHFIELD = 'DBZHCC'
        ZH =  radar.fields['DBZHCC']['data'][start_index:end_index]
	
    #---- plot hid ppi  
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid       = colors.ListedColormap(hid_colors)

    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
    pcm1 = axes.pcolormesh(lons, lats, RHIs_nlev, cmap = cmaphid, vmin=0.2, vmax=10)
    axes.set_title('HID nlev '+str(nlev)+' PPI (at '+str(options['time_pfs'])+')')
    axes.set_xlim([options['xlim_min'], options['xlim_max']])
    axes.set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    cbar_HID = plt.colorbar(pcm1, ax=axes, shrink=1.1, label=r'HID')    
    cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	



    # agregar: 	
    # read 
    f = h5py.File( options['gmi_dir']+options['gfile'], 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    for j in range(lon_gmi.shape[1]):  # ANTES ERA +-5
            tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+10),:] = np.nan
            tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-10),:] = np.nan   
            lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+10),:] = np.nan
            lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-10),:] = np.nan  
            lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+10),:] = np.nan
            lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-10),:] = np.nan  	
    PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 	
    CS = axes.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 225], colors=(['black', 'gray']), linewidths=1.5)
    labels_cont = ['GMI 200K contour', 'GMI 225K contour']
    for i in range(len(labels_cont)):
      CS.collections[i].set_label(labels_cont[i])
    if len(options['REPORTES_meta'])>0:
    	for ireportes in range(len(options['REPORTES_geo'])):
        	axes.plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], '*', markeredgecolor='black', markerfacecolor='black', markersize=10, label=options['REPORTES_meta'][ireportes])
    	plt.legend() 

    fig.savefig(options['fig_dir']+'PPIs_HID'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)   
    #plt.close()

    return 


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_sys_phase_simple(radar):

    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]

    phases_nlev = []
    for nlev in range(radar.nsweeps-1):
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats  = radar.gate_latitude['data'][start_index:end_index]
        lons  = radar.gate_longitude['data'][start_index:end_index]
        if 'TH' in radar.fields.keys():  
            TH    = radar.fields['TH']['data'][start_index:end_index]
            TV    = radar.fields['TV']['data'][start_index:end_index]
            RHOHV = radar.fields['RHOHV']['data'][start_index:end_index]
            PHIDP = np.array(radar.fields['PHIDP']['data'][start_index:end_index])
            PHIDP[np.where(PHIDP==radar.fields['PHIDP']['data'].fill_value)] = np.nan
        elif 'DBZHCC' in radar.fields.keys():
            TH    = radar.fields['DBZHCC']['data'][start_index:end_index]
            TV    = radar.fields['DBZVCC']['data'][start_index:end_index]
            RHOHV = radar.fields['RHOHV']['data'][start_index:end_index]
            PHIDP = np.array(radar.fields['PHIDP']['data'][start_index:end_index])
            PHIDP[np.where(PHIDP==radar.fields['PHIDP']['data'].fill_value)] = np.nan		
        elif 'DBZH' in radar.fields.keys():
            TH    = radar.fields['DBZH']['data'][start_index:end_index]
            TV    = radar.fields['DBZV']['data'][start_index:end_index]
            RHOHV = radar.fields['RHOHV']['data'][start_index:end_index]
            PHIDP = np.array(radar.fields['PHIDP']['data'][start_index:end_index])
            PHIDP[np.where(PHIDP==radar.fields['PHIDP']['data'].fill_value)] = np.nan	
        rhv = RHOHV.copy()
	z_h = TH.copy()
        PHIDP = np.where( (rhv>0.7) & (z_h>30), PHIDP, np.nan)
        # por cada radial encontrar first non nan value: 
        phases = []
        for radial in range(radar.sweep_end_ray_index['data'][0]):
            if firstNonNan(PHIDP[radial,30:]):
                phases.append(firstNonNan(PHIDP[radial,30:]))
        phases_nlev.append(np.median(phases))
    phases_out = np.nanmedian(phases_nlev) 

    return phases_out

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_sys_phase_simple_dow7(radar):

    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]

    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    TH    = radar.fields['DBZHCC']['data'][start_index:end_index]
    TV    = radar.fields['DBZVCC']['data'][start_index:end_index]
    RHOHV = radar.fields['RHOHV']['data'][start_index:end_index]
    PHIDP = np.array(radar.fields['PHIDP']['data'][start_index:end_index])
    PHIDP[np.where(PHIDP==radar.fields['PHIDP']['data'].fill_value)] = np.nan		
    rhv = RHOHV.copy()
    z_h = TH.copy()
    PHIDP = np.where( (rhv>0.7) & (z_h>30), PHIDP, np.nan)
    # por cada radial encontrar first non nan value: 
    phases = []
    for radial in range(radar.sweep_end_ray_index['data'][0]):
        if firstNonNan(PHIDP[radial,30:]):
            phases.append(firstNonNan(PHIDP[radial,30:]))
    phases_nlev = np.median(phases)

    return phases_nlev

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def get_sys_phase_simple_CSPR2(radar):

    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]

    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    TH    = radar.fields['attenuation_corrected_reflectivity_h']['data'][start_index:end_index]
    RHOHV = radar.fields['copol_correlation_coeff']['data'][start_index:end_index]
    PHIDP = np.array(radar.fields['differential_phase']['data'][start_index:end_index])+360
    PHIDP[np.where(PHIDP==radar.fields['differential_phase']['data'].fill_value)] = np.nan		
    rhv = RHOHV.copy()
    z_h = TH.copy()
    PHIDP = np.where( (rhv>0.7) & (z_h>30), PHIDP, np.nan)
    # por cada radial encontrar first non nan value: 
    phases = []
    for radial in range(radar.sweep_end_ray_index['data'][0]):
        if firstNonNan(PHIDP[radial,30:]):
            phases.append(firstNonNan(PHIDP[radial,30:]))
    phases_nlev = np.median(phases)

    return phases_nlev
#------------------------------------------------------------------------------

def get_sys_phase(radar, ncp_lev, rhohv_lev, ncp_field, rhv_field, phidp_field):

    """
    Adapted from pyart 
    ----------
    Determine the system phase.
    Parameters
    ----------
    radar : Radar
        Radar object for which to determine the system phase.
    ncp_lev : float, optional
        Miminum normal coherent power level. Regions below this value will
        not be included in the phase calculation.
    rhohv_lev : float, optional
        Miminum copolar coefficient level. Regions below this value will not
        be included in the phase calculation.
    ncp_field, rhv_field, phidp_field : str, optional
        Field names within the radar object which represent the normal
        coherent power, the copolar coefficient, and the differential phase
        shift. A value of None for any of these parameters will use the
        default field name as defined in the Py-ART configuration file.

    Returns
    -------
    sys_phase : float or None
        Estimate of the system phase. None is not estimate can be made.
    """
	
    # parse the field parameters
    if ncp_field is None:
        ncp_field = get_field_name('normalized_coherent_power')
    if rhv_field is None:
        rhv_field = get_field_name('cross_correlation_ratio')
    if phidp_field is None:
        phidp_field = get_field_name('differential_phase')

    ncp   = radar.fields[ncp_field]['data'][:, 30:]
    rhv   = radar.fields[rhv_field]['data'][:, 30:]
    phidp = np.array(radar.fields[phidp_field]['data'])[:, 30:]
    last_ray_idx = radar.sweep_end_ray_index['data'][0]

    return _det_sys_phase(ncp, rhv, phidp, last_ray_idx, ncp_lev,
                          rhohv_lev)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# this function adapted from the Scipy Cookbook:
# http://www.scipy.org/Cookbook/SignalSmooth
def smooth_and_trim(x, window_len=11, window='hanning'):
    """
    Smooth data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    Parameters
    ----------
    x : array
        The input signal.
    window_len : int, optional
        The dimension of the smoothing window; should be an odd integer.
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman' or 'sg_smooth'. A flat window will produce a moving
        average smoothing.
    Returns
    -------
    y : array
        The smoothed signal with length equal to the input signal.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    valid_windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman',
                     'sg_smooth']
    if window not in valid_windows:
        raise ValueError("Window is on of " + ' '.join(valid_windows))

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(int(window_len), 'd')
    elif window == 'sg_smooth':
        w = np.array([0.1, .25, .3, .25, .1])
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len / 2):len(x) + int(window_len / 2)]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def _det_sys_phase(ncp, rhv, phidp, last_ray_idx, ncp_lev=0.4,
                   rhv_lev=0.7):
    """ ADAPTED FROM PYART 
    Determine the system phase, see :py:func:`det_sys_phase`. """
    good = False
    phases = []
    for radial in range(last_ray_idx + 1):
        meteo = np.logical_and(ncp[radial, :] > ncp_lev,
                               rhv[radial, :] > rhv_lev)
        mpts = np.where(meteo)
        if len(mpts[0]) > 25:
            good = True
            msmth_phidp = smooth_and_trim(phidp[radial, mpts[0]], 9)
            phases.append(msmth_phidp[0:25].min())
    if not good:
        return None
    return np.median(phases)

#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def correct_PHIDP_KDP(radar, options, nlev, azimuth_ray, diff_value, tfield_ref, alt_ref):

    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # OJO CON MOVING AVERAGE EN pyart.correct.phase_proc.smooth_and_trim QUE USO VENTANA DE 40! 	
    # breakpoint()
    #----- cambiado para RMA3: 	
    #sys_phase  = get_sys_phase(radar, ncp_lev=0.8, rhohv_lev=0.7, 
    #	 						ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP')
    #dphi, uphi, corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], radar.fields['TH']['data'], 
    #					   sys_phase, diff_value)
    #------ REMPLAZADO POR:
    if 'TH' in radar.fields.keys():  
        sys_phase = get_sys_phase_simple(radar)
    elif 'DBZHCC' in radar.fields.keys():
        sys_phase = get_sys_phase_simple_dow7(radar)
    elif 'DBZH' in radar.fields.keys():
        sys_phase = get_sys_phase_simple(radar)	
    # replace PHIDP w/ np.nan
    #PHIORIG = radar.fields['PHIDP']['data'].copy() 
    #PHIDP_nans = radar.fields['PHIDP']['data'].copy() 
    #PHIDP_nans[np.where(PHIDP_nans.data==radar.fields['PHIDP']['data'].fill_value)] = np.nan
    #mask = radar.fields['PHIDP']['data'].data.copy()    
    #mask[:] = False
    #PHIDP_nans.mask = mask
    #radar.add_field_like('PHIDP', 'PHIDP', PHIDP_nans, replace_existing=True)
    if 'TH' in radar.fields.keys():  
        dphi, uphi, corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], radar.fields['TH']['data'], sys_phase, 280)
    elif 'DBZHCC' in radar.fields.keys():
        dphi, uphi, corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], radar.fields['DBZHCC']['data'], sys_phase, 280)
    elif 'DBZH' in radar.fields.keys():
        dphi, uphi, corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], radar.fields['DBZH']['data'], sys_phase, 280)
    #------------	
    #------------	
    radar.add_field_like('RHOHV','corrPHIDP', corr_phidp, replace_existing=True)

    # Y CALCULAR KDP! 
    calculated_KDP = wrl.dp.kdp_from_phidp(corr_phidp, winlen=options['window_calc_KDP'], dr=(radar.range['data'][1]-radar.range['data'][0])/1e3, 
					   method='lanczos_conv', skipna=True)	
    radar.add_field_like('RHOHV','corrKDP', calculated_KDP, replace_existing=True)

    # AGREGAR HID?
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    if 'TH' in radar.fields.keys():  
        dzh_  = radar.fields['TH']['data'].copy()
        dzv_  = radar.fields['TV']['data'].copy()
    elif 'DBZHCC' in radar.fields.keys():
        dzh_  = radar.fields['DBZHCC']['data'].copy()
        dzv_  = radar.fields['DBZVCC']['data'].copy()
    elif 'DBZH' in radar.fields.keys():
        dzh_  = radar.fields['DBZH']['data'].copy()
        dzv_  = radar.fields['DBZV']['data'].copy()	
    drho_ = radar.fields['RHOHV']['data'].copy()
    dkdp_ = radar.fields['corrKDP']['data'].copy()

    # ESTO DE ACA ABAJO PROBADO PARA RMA3:  
    dkdp_[np.where(drho_.data==radar.fields['RHOHV']['data'].fill_value)] = np.nan

    # Filters
    ni = dzh_.shape[0]
    nj = dzh_.shape[1]
    for i in range(ni):
        rho_h = drho_[i,:]
        zh_h = dzh_[i,:]
        for j in range(nj):
            if (rho_h[j]<0.7) or (zh_h[j]<30):
                dzh_[i,j]  = np.nan
                dzv_[i,j]  = np.nan
                drho_[i,j]  = np.nan
                dkdp_[i,j]  = np.nan

    scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=(dzh_-dzv_) - options['ZDRoffset'], 
					     rho=drho_, kdp=dkdp_, 
                                             use_temp=True, band='C', T=radar_T)

    RHIs_nlev = np.argmax(scores, axis=0) + 1 
    radar.add_field_like('corrKDP','HID', RHIs_nlev, replace_existing=True)

    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]

    #-EJEMPLO de azimuth
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[azimuth_ray]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()

    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    rhoHV = radar.fields['RHOHV']['data'][start_index:end_index]
    PHIDP = radar.fields['PHIDP']['data'][start_index:end_index]
    if 'KDP' in radar.fields.keys():  
    	KDP   = radar.fields['KDP']['data'][start_index:end_index]

    fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True,
                        figsize=[14,7])
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,0].pcolormesh(lons, lats, rhoHV, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,0].set_title('RHOHV radar nlev '+str(nlev)+' PPI')
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
    pcm1 = axes[0,1].pcolormesh(lons, lats, radar.fields['PHIDP']['data'][start_index:end_index], cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,1].set_title('Phidp radar nlev '+str(nlev)+' PPI')
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
    if 'KDP' in radar.fields.keys():  
    	pcm1 = axes[0,2].pcolormesh(lons, lats, KDP, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
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
    axes[0,2].set_title('KDP radar nlev '+str(nlev)+' PPI')

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    if 'TH' in radar.fields.keys():  
        ZHFIELD = 'TH'
        THH =  radar.fields['TH']['data'][start_index:end_index]
        pcm1 = axes[1,0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    elif 'DBZHCC' in radar.fields.keys():
        ZHFIELD = 'DBZHCC'
        THH =  radar.fields['DBZHCC']['data'][start_index:end_index]
        pcm1 = axes[1,0].pcolormesh(lons, lats, radar.fields['DBZHCC']['data'][start_index:end_index], cmap=cmap, 
			  vmin=vmin, vmax=vmax)		
    elif 'DBZH' in radar.fields.keys():
        ZHFIELD = 'DBZH'
        THH =  radar.fields['DBZH']['data'][start_index:end_index]
        pcm1 = axes[1,0].pcolormesh(lons, lats, radar.fields['DBZH']['data'][start_index:end_index], cmap=cmap, 
			  vmin=vmin, vmax=vmax)	
    axes[1,0].set_title('ZH nlev '+str(nlev)+' PPI')
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
    axes[1,1].contour(lons,lats, THH, [45], colors='k', linewidths=0.8)  
    axes[1,1].set_title('CORR Phidp radar nlev '+str(nlev)+'  PPI')
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
    axes[1,2].contour(lons,lats, THH, [45], colors='k', linewidths=0.8)  
    axes[1,2].set_title('Calc. KDP nlev '+str(nlev)+' PPI')
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
    #axes[1,2].contour(lons, lats, radar.fields['TH']['data'][start_index:end_index], [45], colors='k') 
    #axes[0,0].set_ylim([-32.5, -31.2])
    #axes[0,0].set_xlim([-65.3, -64.5])
    fig.savefig(options['fig_dir']+'PPIs_KDPcorr'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)   
    #plt.close()

    #----------------------------------------------------------------------
    #-figure
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True,figsize=[14,10])
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['RHOHV']['data'][start_index:end_index][filas,:])*100, '-k', label='RHOHV')
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields[ZHFIELD]['data'][start_index:end_index][filas,:]), '-r', label='ZH')
    axes[0].legend()
    axes[1].plot(radar.range['data']/1e3, np.ravel(radar.fields['PHIDP']['data'][start_index:end_index][filas,:]), 'or', label='obs. phidp')
    axes[1].plot(radar.range['data']/1e3, np.ravel(dphi[start_index:end_index][filas,:]), '*b', label='despeckle phidp'); 
    axes[1].plot(radar.range['data']/1e3, np.ravel(uphi[start_index:end_index][filas,:]), color='darkgreen', label='unfolded phidp');
    axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]+sys_phase), color='magenta', label='phidp corrected');
    axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]), color='purple', label='phidp corrected-sysphase');
    axes[1].set_ylim([-5, 360])
    axes[1].legend()
    axes[2].plot(radar.range['data']/1e3, np.ravel(calculated_KDP[start_index:end_index][filas,:]), color='k', label='Calc. KDP');
    if 'KDP' in radar.fields.keys():  
        axes[2].plot(radar.range['data']/1e3, np.ravel(radar.fields['KDP']['data'][start_index:end_index][filas,:]), color='gray', label='Obs. KDP');
    axes[2].legend()
    #axes[0].set_xlim([50, 120])
    #axes[1].set_xlim([50, 120])
    #axes[2].set_xlim([50, 120])
    axes[2].set_ylim([-1, 5])
    axes[2].grid(True) 
    axes[2].plot([0, np.nanmax(radar.range['data']/1e3)], [0, 0], color='darkgreen', linestyle='-') 
    fig.savefig(options['fig_dir']+'PHIcorrazi'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)    
    #plt.close()

    return radar

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def DOW7_NOcorrect_PHIDP_KDP(radar, options, nlev, azimuth_ray, diff_value, tfield_ref, alt_ref):

    #---- plot hid ppi  
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
           'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)

    # AGREGAR HID?
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    if 'TH' in radar.fields.keys():  
        dzh_  = radar.fields['TH']['data'].copy()
        dzv_  = radar.fields['TV']['data'].copy()
    elif 'DBZHCC' in radar.fields.keys():
        dzh_  = radar.fields['DBZHCC']['data'].copy()
        dzv_  = radar.fields['DBZVCC']['data'].copy()
        dZDR  = radar.fields['ZDRC']['data'].copy()
    drho_ = radar.fields['RHOHV']['data'].copy()
    dkdp_ = radar.fields['KDP']['data'].copy()

    # ESTO DE ACA ABAJO PROBADO PARA RMA3:  
    dkdp_[np.where(drho_.data==radar.fields['RHOHV']['data'].fill_value)] = np.nan

    # Filters
    ni = dzh_.shape[0]
    nj = dzh_.shape[1]
    for i in range(ni):
        rho_h = drho_[i,:]
        zh_h = dzh_[i,:]
        for j in range(nj):
            if (rho_h[j]<0.7) or (zh_h[j]<30):
                dzh_[i,j]  = np.nan
                dzv_[i,j]  = np.nan
                drho_[i,j]  = np.nan
                dkdp_[i,j]  = np.nan

    scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=dZDR - options['ZDRoffset'], 
					     rho=drho_, kdp=dkdp_, 
                                             use_temp=True, band='C', T=radar_T)

    RHIs_nlev = np.argmax(scores, axis=0) + 1 
    radar.add_field_like('KDP','HID', RHIs_nlev, replace_existing=True)


    #-EJEMPLO de azimuth
    azimuths = radar.azimuth['data']
    target_azimuth = azimuths[azimuth_ray]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()

    lats  = radar.gate_latitude['data']
    lons  = radar.gate_longitude['data']
    rhoHV = radar.fields['RHOHV']['data']
    PHIDP = radar.fields['PHIDP']['data']
    if 'KDP' in radar.fields.keys():  
    	KDP   = radar.fields['KDP']['data']

    fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True,
                        figsize=[14,7])
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,0].pcolormesh(lons, lats, rhoHV, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,0].set_title('RHOHV radar nlev '+str(nlev)+' PPI')
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
    pcm1 = axes[0,1].pcolormesh(lons, lats, radar.fields['PHIDP']['data'], cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,1].set_title('Phidp radar nlev '+str(nlev)+' PPI')
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
    if 'KDP' in radar.fields.keys():  
    	pcm1 = axes[0,2].pcolormesh(lons, lats, KDP, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
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
    axes[0,2].set_title('KDP radar nlev '+str(nlev)+' PPI')

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    if 'TH' in radar.fields.keys():  
        ZHFIELD = 'TH'
        THH =  radar.fields['TH']['data']
        pcm1 = axes[1,0].pcolormesh(lons, lats, radar.fields['TH']['data'], cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    elif 'DBZHCC' in radar.fields.keys():
        ZHFIELD = 'DBZHCC'
        THH =  radar.fields['DBZHCC']['data']
        pcm1 = axes[1,0].pcolormesh(lons, lats, radar.fields['DBZHCC']['data'], cmap=cmap, 
			  vmin=vmin, vmax=vmax)		
    axes[1,0].set_title('ZH nlev '+str(nlev)+' PPI')
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
    pcm1 = axes[1,1].pcolormesh(lons, lats, radar.fields['PHIDP']['data'], cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[1,1].contour(lons,lats, THH, [45], colors='k', linewidths=0.8)  
    axes[1,1].set_title('CORR Phidp radar nlev '+str(nlev)+'  PPI')
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

    pcm1 = axes[1,2].pcolormesh(lons, lats, RHIs_nlev, cmap = cmaphid, vmin=0.2, vmax=10)
    axes[1,2].contour(lons,lats, THH, [45], colors='k', linewidths=0.8)  
    axes[1,2].set_title('HID '+str(nlev)+' PPI')
    axes[1,2].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1,2].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    axes[1,2].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')
    cbar_HID = plt.colorbar(pcm1, ax=axes[1,2], shrink=1.1, label=r'HID')    
    cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	
	
    fig.savefig(options['fig_dir']+'PPIs_KDPcorr'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)   
    #plt.close()

    return radar

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def CSPR2_correct_PHIDP_KDP(radar, options, nlev, azimuth_ray, diff_value, tfield_ref, alt_ref):

    #---- plot hid ppi  
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
           'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)

    # AGREGAR HID?
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  

    dzh_  = radar.fields['corrected_reflectivity']['data'].copy()
    dZDR  = radar.fields['corrected_differential_reflectivity']['data'].copy()
    drho_ = radar.fields['copol_correlation_coeff']['data'].copy()
    dkdp_ = radar.fields['filtered_corrected_specific_diff_phase']['data'].copy()
    ZHFIELD = 'corrected_reflectivity'
    # ESTO DE ACA ABAJO PROBADO PARA RMA3:  
    dkdp_[np.where(drho_.data==radar.fields['copol_correlation_coeff']['data'].fill_value)] = np.nan

    #------------	
    #------------		
    sys_phase = get_sys_phase_simple_CSPR2(radar)
    # replace PHIDP w/ np.nan
    #PHIORIG = radar.fields['PHIDP']['data'].copy() 
    #PHIDP_nans = radar.fields['PHIDP']['data'].copy() 
    #PHIDP_nans[np.where(PHIDP_nans.data==radar.fields['PHIDP']['data'].fill_value)] = np.nan
    #mask = radar.fields['PHIDP']['data'].data.copy()    
    #mask[:] = False
    #PHIDP_nans.mask = mask
    #radar.add_field_like('PHIDP', 'PHIDP', PHIDP_nans, replace_existing=True)
    #dphi, uphi, corr_phidp = correct_phidp((radar.fields['differential_phase']['data'])+180, radar.fields['copol_correlation_coeff']['data'], 
    #					   radar.fields['attenuation_corrected_reflectivity_h']['data'], sys_phase, 280)
    #------------	
    #radar.add_field_like('reflectivity','corrPHIDP', corr_phidp, replace_existing=True)

    # Y CALCULAR KDP! 
    #calculated_KDP = wrl.dp.kdp_from_phidp(corr_phidp, winlen=options['window_calc_KDP'], dr=(radar.range['data'][1]-radar.range['data'][0])/1e3, 
    #					   method='lanczos_conv', skipna=True)	
    #radar.add_field_like('reflectivity','corrKDP', calculated_KDP, replace_existing=True)
    #------------	
    #------------	

    # Filters
    ni = dzh_.shape[0]
    nj = dzh_.shape[1]
    for i in range(ni):
        rho_h = drho_[i,:]
        zh_h = dzh_[i,:]
        for j in range(nj):
            if (rho_h[j]<0.7) or (zh_h[j]<30):
                dzh_[i,j]  = np.nan
                dZDR[i,j]  = np.nan
                drho_[i,j]  = np.nan
                dkdp_[i,j]  = np.nan		

    scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=dZDR, 
					     rho=drho_, kdp=dkdp_, 
                                             use_temp=True, band='C', T=radar_T)

    RHIs_nlev = np.argmax(scores, axis=0) + 1 
    radar.add_field_like('filtered_corrected_differential_phase','HID', RHIs_nlev, replace_existing=True)

    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]

    #-EJEMPLO de azimuth
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[azimuth_ray]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()

    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    rhoHV = radar.fields['copol_correlation_coeff']['data'][start_index:end_index]
    PHIDP = (radar.fields['filtered_corrected_differential_phase']['data'][start_index:end_index])+360
    KDP   = radar.fields['filtered_corrected_specific_diff_phase']['data'][start_index:end_index]

    fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True,
                        figsize=[14,7])
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,0].pcolormesh(lons, lats, rhoHV, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,0].set_title('RHOHV radar nlev '+str(nlev)+' PPI')
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
    axes[0,1].set_title('Phidp radar nlev '+str(nlev)+' PPI')
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
    THH =  radar.fields['attenuation_corrected_reflectivity_h']['data'][start_index:end_index]
    #axes[0,2].contour(lons,lats, THH, [45], colors='k', linewidths=0.8)  
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
    axes[0,2].set_title('KDP radar nlev '+str(nlev)+' PPI')

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    ZHFIELD = 'corrected_reflectivity'
    THH =  radar.fields['corrected_reflectivity']['data'][start_index:end_index]
    pcm1 = axes[1,0].pcolormesh(lons, lats, THH, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[1,0].set_title('ZH nlev '+str(nlev)+' PPI')
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

    pcm1 = axes[1,2].pcolormesh(lons, lats, RHIs_nlev[start_index:end_index], cmap = cmaphid, vmin=0.2, vmax=10)
    axes[1,2].contour(lons,lats, THH, [45], colors='k', linewidths=0.8)  
    axes[1,2].set_title('HID '+str(nlev)+' PPI')
    axes[1,2].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1,2].set_ylim([options['ylim_min'], options['ylim_max']])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
    axes[1,2].plot(np.ravel(lons[filas,:]),np.ravel(lats[filas,:]), '-k')
    cbar_HID = plt.colorbar(pcm1, ax=axes[1,2], shrink=1.1, label=r'HID')    
    cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	

    fig.savefig(options['fig_dir']+'PPIs_KDPcorr'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)   
    #plt.close()

    #----------------------------------------------------------------------
    #-figure
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True,figsize=[14,10])
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['copol_correlation_coeff']['data'][start_index:end_index][filas,:])*100, '-k', label='RHOHV')
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields[ZHFIELD]['data'][start_index:end_index][filas,:]), '-r', label='ZH')
    axes[0].legend()
    axes[1].plot(radar.range['data']/1e3, np.ravel(radar.fields['differential_phase']['data'][start_index:end_index][filas,:]), 'or', label='phidp')
    axes[1].legend()
    axes[1].set_xlim([0,100])
    axes[2].plot(radar.range['data']/1e3, np.ravel(radar.fields['filtered_corrected_specific_diff_phase']['data'][start_index:end_index][filas,:]), color='gray', label='KDP');
    axes[2].legend()
    #axes[0].set_xlim([50, 120])
    #axes[1].set_xlim([50, 120])
    #axes[2].set_xlim([50, 120])
    axes[2].set_ylim([-1, 5])
    axes[2].grid(True) 
    axes[2].plot([0, 300], [0, 0], color='darkgreen', linestyle='-') 
    axes[2].set_xlim([0,100])
    axes[0].set_xlim([0,100])
    fig.savefig(options['fig_dir']+'PHIcorrazi'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)    
#plt.close()




    return radar

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def check_this(azimuth_ray): 
	
    options = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5, 
	    'ZDRoffset': 4, 'ylim_max_zoom':-30.5}
    rfile     = 'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile)
    sys_phase  = pyart.correct.phase_proc.det_sys_phase(radar, ncp_lev=0.8, rhohv_lev=0.7, 
							ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP')
    # OJO CON MOVING AVERAGE EN pyart.correct.phase_proc.smooth_and_trim QUE USO VENTANA DE 40! 	
    dphi, uphi, corr_phidp, uphi2, phi_masked = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], radar.fields['TH']['data'], sys_phase, 280)
    
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
    axes[1].plot(radar.range['data']/1e3, np.ravel(phi_masked[start_index:end_index][filas,:]), 'or', label='obs. phidp masked')
    axes[1].plot(radar.range['data']/1e3, np.ravel(dphi[start_index:end_index][filas,:]), '*b', label='despeckle phidp'); 
    axes[1].plot(radar.range['data']/1e3, np.ravel(uphi[start_index:end_index][filas,:]), color='darkgreen', label='unfolded phidp');
    axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]), color='magenta', label='phidp corrected');
    #plt.plot(radar.range['data']/1e3, np.ravel(uphidp_2[filas,:]), color='k', label='uphidp2');
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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_contour_info(contorno, icois, datapts_in): 

    # Get verticies: 
    for item in contorno.collections:
        for i in item.get_paths():
            v = i.vertices
            x = v[:, 0]
            y = v[:, 1] 

    # Get vertices of these polygon type shapes
    for ii in range(len(icois)): 
        X1 = []; Y1 = []; vertices = []
        for ik in range(len(contorno.collections[0].get_paths()[int(icois[ii])].vertices)): 
            X1.append(contorno.collections[0].get_paths()[icois[ii]].vertices[ik][0])
            Y1.append(contorno.collections[0].get_paths()[icois[ii]].vertices[ik][1])
            vertices.append([contorno.collections[0].get_paths()[icois[ii]].vertices[ik][0], 
                                        contorno.collections[0].get_paths()[icois[ii]].vertices[ik][1]])
        #convexhull = ConvexHull(vertices)
        array_points = np.array(vertices)

        #hull_path   = Path( array_points[convexhull.vertices] )
        #------- testing from https://stackoverflow.com/questions/57260352/python-concave-hull-polygon-of-a-set-of-lines 
        alpha = 0.95 * alphashape.optimizealpha(array_points)
        hull_pts_CONCAVE   = alphashape.alphashape(array_points, alpha)
        hull_coors_CONCAVE = hull_pts_CONCAVE.exterior.coords.xy
        check_points = np.vstack((hull_coors_CONCAVE)).T
        concave_path = Path(check_points)
        
        if ii == 0:
            inds_1   = concave_path.contains_points(datapts_in)
            TB_inds      = [inds_1]
       	if ii == 1:
            inds_2   = concave_path.contains_points(datapts_in)
            TB_inds      = [inds_1, inds_2]
        if ii == 2:
            inds_3   = concave_path.contains_points(datapts_in)
            TB_inds      = [inds_1, inds_2, inds_3]
    

    return TB_inds



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def visual_coi_identification(options, radar, fname):

    # ojo que aca agarro los verdaderos PCTMIN, no los que me pasó Sarah B. que estan 
    # ajustados a TMI footprints. 
    #breakpoint()
    # read file
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

    idx1 = (lat_gmi>=options['ylim_min']-5) & (lat_gmi<=options['ylim_max']+5) & (lon_gmi>=options['xlim_min']-5) & (lon_gmi<=options['xlim_max']+5)

    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)
    ##------------------------------------------------------------------------------------------------
    if 'TH' in radar.fields.keys():  
       THNAME= 'TH'
    elif 'DBZHCC' in radar.fields.keys():        
       THNAME= 'DBZHCC'

    nlev=0
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[14,12])
    axes.pcolormesh(lon_gmi, lat_gmi, PCT89); plt.xlim([-70,-60]); plt.ylim([-40,-20])

    #----------------------------------------------------------------------------------------
    # Test plot figure: General figure with Zh and the countours identified 
    #----------------------------------------------------------------------------------------
    test_this = 1
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[14,12])
    #-- Zh: 
    if 'TH' in radar.fields.keys():  
        radarTH = radar.fields['TH']['data'][start_index:end_index]
        radarZDR = (radar.fields['TH']['data'][start_index:end_index])-(radar.fields['TV']['data'][start_index:end_index])-options['ZDRoffset']
    elif 'DBZH' in radar.fields.keys():
        radarTH = radar.fields['DBZH']['data'][start_index:end_index]
    elif 'reflectivity' in radar.fields.keys(): 
        radarTH = radar.fields['DBZH']['data'][start_index:end_index]
    elif 'DBZHCC' in radar.fields.keys(): 
        radarTH = radar.fields['DBZHCC']['data'][start_index:end_index]
        radarZDR = radar.fields['ZDRC']['data'][start_index:end_index]
    elif 'attenuation_corrected_reflectivity_h' in radar.fields.keys(): 
        radarTH = radar.fields['attenuation_corrected_reflectivity_h']['data'][start_index:end_index]
    elif 'DBZH' in radar.fields.keys(): 
        radarTH = radar.fields['DBZH']['data'][start_index:end_index]	
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes.pcolormesh(lons, lats, radarTH, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes.grid(True)
    axes.legend(loc='upper left')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    contorno89 = plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200])# , colors=(['r']), linewidths=1.5);
    contorno89_FIX = plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:])# , [200], colors=(['k']), linewidths=1.5);

    axes.set_xlim([-70,-50]); axes.set_ylim([-40,-20])

    # Find the contour of intere
    for item in contorno89.collections:
        counter=0
        for i in item.get_paths():
            v = i.vertices
            x = v[:, 0]
            y = v[:, 1]
            plt.plot(x,y, label=str(counter))
            print(i)
            plt.legend(loc=1)# , ncol=2)
            counter=counter+1
    
    datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] ))
    
    # Get vertices of these polygon type shapes
    for item in range(len(contorno89.collections[0].get_paths())):
        X1 = []; Y1 = []; vertices = []
        for ik in range(len(contorno89.collections[0].get_paths()[item].vertices)): 
            X1.append(contorno89.collections[0].get_paths()[item].vertices[ik][0])
            Y1.append(contorno89.collections[0].get_paths()[item].vertices[ik][1])
            vertices.append([contorno89.collections[0].get_paths()[item].vertices[ik][0], 
                                    contorno89.collections[0].get_paths()[item].vertices[ik][1]])
        
        convexhull = ConvexHull(vertices)
        array_points = np.array(vertices)
        hull_path   = Path( array_points[convexhull.vertices] )
        inds = hull_path.contains_points(datapts)
        
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[14,12])   
        plt.pcolormesh(lon_gmi, lat_gmi, PCT89); plt.xlim([-70,-50]); plt.ylim([-40,-20]); 
        plt.plot(lon_gmi[:,:][idx1][inds], lat_gmi[:,:][idx1][inds],'xm','markersize'=20 ); 
        plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200])
        plt.title(str(item))
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#def plot_scatter(options, GMI_tbs1_37, GMI_tbs1_85, RN_inds, radar, icois): 
def plot_scatter(options, radar, icois, fname):

    # ojo que aca agarro los verdaderos PCTMIN, no los que me pasó Sarah B. que estan 
    # ajustados a TMI footprints. 
    #breakpoint()
    # read file
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

    idx1 = (lat_gmi>=options['ylim_min']-5) & (lat_gmi<=options['ylim_max']+5) & (lon_gmi>=options['xlim_min']-5) & (lon_gmi<=options['xlim_max']+5)
	
    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    ##------------------------------------------------------------------------------------------------
    #for j in range(lon_gmi.shape[1]):
    #  tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan   
    #  lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  
    #  lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  	
	
    ## keep domain of interest only by keeping those where the center nadir obs is inside domain
    ##inside_s1   = np.logical_and(np.logical_and(lon_gmi >= options['xlim_min']-5, lon_gmi <=  options['xlim_max']+5), 
    ##                          np.logical_and(lat_gmi >= options['ylim_min']-5, lat_gmi <= options['ylim_max']+5))
    ##inside_s2   = np.logical_and(np.logical_and(lon_s2_gmi >= options['xlim_min']-5, lon_s2_gmi <=  options['xlim_max']+5), 
    ##                                     np.logical_and(lat_s2_gmi >= options['ylim_min']-5, lat_s2_gmi <= options['ylim_max']+5))    
    ##lon_gmi_inside   = lon_gmi[inside_s1] 
    ##lat_gmi_inside   = lat_gmi[inside_s1] 	
    ##lon_gmi2_inside  = lon_s2_gmi[inside_s2] 	
    ##lat_gmi2_inside  = lat_s2_gmi[inside_s2] 	
    ##tb_s1_gmi_inside = tb_s1_gmi[inside_s1, :]
    ##
    ##PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 
    ##------------------------------------------------------------------------------------------------

    # Tambien se puenden hacer recortes guardando los indices. ejemplo para S1: 
    #idx1 = (lat_gmi>=options['ylim_min']) & (lat_gmi<=options['ylim_max']+1) & (lon_gmi>=options['xlim_min']) & (lon_gmi<=options['xlim_max']+2)
    #S1_sub_lat  = lat_gmi[:,:][idx1] 
    #S1_sub_lon  = lon_gmi[:,:][idx1]
    #S1_subch89V = tb_s1_gmi[:,:,7][idx1]		

    #idx2 = (lat_s2_gmi>=options['ylim_min']) & (lat_s2_gmi<=options['ylim_max']+1) & (lon_s2_gmi>=options['xlim_min']) & (lon_s2_gmi<=options['xlim_max']+2)
    #S2_sub_lat  = lat_s2_gmi[:,:][idx2] 
    #S2_sub_lon  = lon_s2_gmi[:,:][idx2]

    # CALCULATE PCTs
    #for j in range(lon_gmi.shape[1]):
    #  tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan   	
    #  lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  
    #  lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
    #  lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  	
		
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)
    ##------------------------------------------------------------------------------------------------
    if 'TH' in radar.fields.keys():  
       THNAME= 'TH'
       RHOHVname = 'RHOHV'
    elif 'DBZHCC' in radar.fields.keys():        
       THNAME= 'DBZHCC'
       RHOHVname = 'RHOHV'
    elif 'corrected_reflectivity' in radar.fields.keys():        
       THNAME= 'corrected_reflectivity'	
       RHOHVname = 'copol_correlation_coeff'
       ZDRname = 'corrected_differential_reflectivity'
    elif 'DBZH' in radar.fields.keys():        
       THNAME= 'DBZH'	
       RHOHVname = 'RHOHV'
       TVNAME= 'DBZV'	

            #ZHZH = radar.fields['corrected_reflectivity']['data'][start_index:end_index]
            #TH   = radar.fields['corrected_reflectivity']['data'][start_index:end_index]
            #ZDRZDR  =  radar.fields['corrected_differential_reflectivity']['data'][start_index:end_index]
            #RHORHO  = radar.fields['copol_correlation_coeff']['data'][start_index:end_index]       
            #PHIPHI  = radar.fields['filtered_corrected_differential_phase']['data'][start_index:end_index]       
            #KDPKDP  = radar.fields['filtered_corrected_specific_diff_phase']['data'][start_index:end_index]    


    nlev=0
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    #fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
    #                    figsize=[14,12])
    #axes.pcolormesh(lon_gmi, lat_gmi, PCT89); plt.xlim([-70,-60]); plt.ylim([-40,-20])

    #----------------------------------------------------------------------------------------
    # Test plot figure: General figure with Zh and the countours identified 
    #----------------------------------------------------------------------------------------
    test_this = 1
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[14,12])
    #-- Zh: 
    if 'TH' in radar.fields.keys():  
        radarTH = radar.fields['TH']['data'][start_index:end_index]
        radarZDR = (radar.fields['TH']['data'][start_index:end_index])-(radar.fields['TV']['data'][start_index:end_index])-options['ZDRoffset']
    elif 'DBZH' in radar.fields.keys():
        radarTH = radar.fields['DBZH']['data'][start_index:end_index]
        radarZDR = radar.fields['DBZH']['data'][start_index:end_index]-radar.fields['DBZV']['data'][start_index:end_index]
	#elif 'reflectivity' in radar.fields.keys(): 
    #    radarTH = radar.fields['DBZH']['data'][start_index:end_index]
    elif 'DBZHCC' in radar.fields.keys(): 
        radarTH = radar.fields['DBZHCC']['data'][start_index:end_index]
        radarZDR = radar.fields['ZDRC']['data'][start_index:end_index]
    elif 'corrected_reflectivity' in radar.fields.keys(): 
        radarTH = radar.fields['corrected_reflectivity']['data'][start_index:end_index]
        radarZDR = radar.fields[ZDRname]['data'][start_index:end_index]

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes.pcolormesh(lons, lats, radarTH, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes.grid(True)
    axes.legend(loc='upper left')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    contorno89 = plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200] , colors=(['r']), linewidths=1.5);
    contorno89_FIX = plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:] , [200], colors=(['k']), linewidths=1.5);
    
    axes.set_xlim([options['xlim_min'], options['xlim_max']]) 
    axes.set_ylim([options['ylim_min'], options['ylim_max']])

    if test_this == 0:
        plt.close()
	
    datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] )) 
    ##datapts = np.column_stack((lon_gmi_inside,lat_gmi_inside))
    ##datapts = np.column_stack(( np.ravel(lon_gmi), np.ravel(lat_gmi) ))
    datapts_RADAR_NATIVE = np.column_stack(( np.ravel(lons),np.ravel(lats) ))

    TB_inds = get_contour_info(contorno89, icois, datapts)
    RN_inds_parallax =  get_contour_info(contorno89_FIX, icois, datapts_RADAR_NATIVE)

    GMI_tbs1_37 = []
    GMI_tbs1_85 = [] 	
	
    S1_sub_tb_v2 = S1_sub_tb[:,:,:][idx1]		

    for ii in range(len(TB_inds)): 
     	GMI_tbs1_37.append( S1_sub_tb_v2[TB_inds[ii],5] ) 
     	GMI_tbs1_85.append( S1_sub_tb_v2[TB_inds[ii],7] ) 
	
    ##for ii in range(len(TB_inds)): 	
    ## 	GMI_tbs1_37.append( tb_s1_gmi_inside[TB_inds[ii],5] ) 
    ## 	GMI_tbs1_85.append( tb_s1_gmi_inside[TB_inds[ii],7] ) 
    #for ii in range(len(TB_inds)): 	
    # 	GMI_tbs1_37.append( tb_s1_gmi[TB_inds[ii],5] ) 
    # 	GMI_tbs1_85.append( tb_s1_gmi[TB_inds[ii],7] ) 
	
    if len(icois)==1:
        colors_plot = ['k']
        labels_plot = [str('icoi=')+str(icois[0])] 	
	
    if len(icois)==2:
        colors_plot = ['k', 'darkblue']
        labels_plot = [str('icoi=')+str(icois[0]), str('icoi=')+str(icois[1])] 
	
    if len(icois)==3:
        colors_plot = ['k', 'darkblue', 'darkred']
        labels_plot = [str('icoi=')+str(icois[0]), str('icoi=')+str(icois[1]), str('icoi=')+str(icois[2])] 

    # Filters
    ni = radarTH.shape[0]
    nj = radarTH.shape[1]
    for i in range(ni):
        rho_h = radar.fields[RHOHVname]['data'][start_index:end_index][i,:]
        zh_h  = radarTH[i,:].copy()
        for j in range(nj):
            if (rho_h[j]<0.7) or (zh_h[j]<30):
                radarZDR[i,j]  = np.nan
                radarTH[i,j]  = np.nan

    #------------------------------------------------------
    # FIGURE CHECK CONTORNOS
    fig = plt.figure(figsize=(20,7)) 
    pcm1 = axes.pcolormesh(lons, lats, radarTH, cmap=cmap, vmin=vmin, vmax=vmax)
    for ic in range(len(GMI_tbs1_37)):
        plt.plot(lon_gmi[:,:][idx1][TB_inds[ic]], lat_gmi[:,:][idx1][TB_inds[ic]],'x' ); 
	###plt.plot( S1_sub_lon[TB_inds[ic]], S1_sub_lat[TB_inds[ic]], 'xr')
	##plt.plot( lon_gmi_inside[TB_inds[ic]], 	lat_gmi_inside[TB_inds[ic]], 'xr')	
        #plt.plot( lon_gmi[TB_inds[ic]], lat_gmi[TB_inds[ic]], 'xr')	       
        plt.plot( np.ravel(lons)[RN_inds_parallax[ic]], 	np.ravel(lats)[RN_inds_parallax[ic]], 'om')
    plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200], colors=(['r']), linewidths=1.5);
    plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);

	
    #------------------------------------------------------
    # FIGURE scatter plot check
    # scatter plots the tbs y de Zh a ver si esta ok 
    fig = plt.figure(figsize=(20,7)) 
    gs1 = gridspec.GridSpec(1, 2)
    #------------------------------------------------------
    ax1 = plt.subplot(gs1[0,0])
    print('CALCULATED PF(MINBTs) FROM CONTOURS: ')
    for ic in range(len(GMI_tbs1_37)):
        print('------- Nr. icoi: '+str(icois[ic])+' -------')
        plt.scatter(GMI_tbs1_37[ic], GMI_tbs1_85[ic], s=40, marker='*', color=colors_plot[ic], label=labels_plot[ic])
        TB_s1 = tb_s1_gmi[idx1][TB_inds[ic],:]         
	##TB_s1 = tb_s1_gmi_inside[TB_inds[ic],:]
        #TB_s1 = tb_s1_gmi[TB_inds[ic],:]
        print('MIN10PCTs: '  +str(np.min(2.5  * TB_s1[:,0] - 1.5  * TB_s1[:,1])) ) 
        print('MIN19PCTs: '  +str(np.min(2.4  * TB_s1[:,2] - 1.4  * TB_s1[:,3])) ) 
        print('MIN37PCTs: '  +str(np.min(2.15 * TB_s1[:,5] - 1.15 * TB_s1[:,6])) ) 
        print('MIN85PCTs: '  +str(np.min(1.7  * TB_s1[:,7] - 0.7  * TB_s1[:,8])) ) 
    plt.grid(True)
    plt.legend()
    #plt.xlim([140,260])
    #plt.ylim([100,240])
    plt.xlabel('TBV(37)')
    plt.ylabel('TBV(85)')

    #------------------------------------------------------	
    ax1 = plt.subplot(gs1[0,1])
    for ic in range(len(RN_inds_parallax)):
        plt.scatter(np.ravel(radarTH)[RN_inds_parallax[ic]], np.ravel(radarZDR)[RN_inds_parallax[ic]]-options['ZDRoffset'], s=20, marker='x', color=colors_plot[ic], label=labels_plot[ic])
    plt.xlabel('ZH')
    plt.ylabel('ZDR')	
    plt.ylim([-15, 10])
    plt.xlim([30, 65])
    plt.grid(True)
    fig.savefig(options['fig_dir']+'variable_scatter_plots.png', dpi=300,transparent=False)   
    #plt.close()

    #- HISTOGRAMA hist2d con ZH y ZDR con algunas stats (areas, minTBs) 	
    from matplotlib.colors import LogNorm
    props = dict(boxstyle='round', facecolor='white')
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True,
                            figsize=[14,12])
    vmax_sample = [] 
    for ic in range(len(RN_inds_parallax)):
        a_x = (np.ravel(radarTH)[RN_inds_parallax[ic]]).copy()
        a_y = (np.ravel(radarZDR)[RN_inds_parallax[ic]]-options['ZDRoffset']).copy()
        a_x = a_x[~np.isnan(a_x)]   
        a_y = a_y[~np.isnan(a_y)]   
        xbin = np.arange(0,80,4)
        ybin = np.arange(-15,10,1)
        H, xedges, yedges = np.histogram2d(a_x, a_y, bins=(xbin, ybin), density=True )
        H = np.rot90(H)
        H = np.flipud(H)
        vmax_sample.append( np.nanmax(np.reshape(H, [-1,1] ))) 
    for ic in range(len(RN_inds_parallax)):
        a_x = (np.ravel(radarTH)[RN_inds_parallax[ic]]).copy()
        a_y = (np.ravel(radarZDR)[RN_inds_parallax[ic]]-options['ZDRoffset']).copy()
        a_x = a_x[~np.isnan(a_x)]   
        a_y = a_y[~np.isnan(a_y)]   
        xbin = np.arange(0,80,4)
        ybin = np.arange(-15,10,1)
        H, xedges, yedges = np.histogram2d(a_x, a_y, bins=(xbin, ybin), density=True )
        H = np.rot90(H)
        H = np.flipud(H)    
        Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
        pcm1 = axes[ic].pcolormesh(xedges, yedges, Hmasked, vmin=0, vmax=np.nanmax(vmax_sample))
        plt.colorbar(pcm1, ax=axes[ic])
        axes[ic].set_title(labels_plot[ic])
        axes[ic].grid(True)
        axes[ic].set_xlim([30, 65])
        axes[ic].set_ylim([-15, 10]); axes[ic].set_ylabel('ZDR')
        TB_s1 = tb_s1_gmi[idx1][TB_inds[ic],:]  
        #TB_s1 = tb_s1_gmi[TB_inds[ic],:]
        pix89  = len(TB_s1[:,7])
        area_ellipse89 = 3.141592 * 7 * 4 # ellise has area 7x4 km
        area89 = pix89*(area_ellipse89)
        gates45dbz = np.ravel(radarTH)[RN_inds_parallax[ic]]
        gates45dbz = gates45dbz[~np.isnan(gates45dbz)]
        gates45dbz = len(gates45dbz)
        #area45 = len(zh45)(240*240)/1000
        if icois[ic] == options['icoi_PHAIL']:
            axes[ic].set_title(labels_plot[ic]+', Phail: '+str(options['phail']))
        s = 'PF(MINPCTs)'
        s = s + '\n' + 'MIN10PCTs: ' + str(np.round(np.min(2.5  * TB_s1[:,0] - 1.5  * TB_s1[:,1]),1)) + ' K'
        s = s + '\n' + 'MIN19PCTs: ' + str(np.round(np.min(2.4  * TB_s1[:,2] - 1.4  * TB_s1[:,3]),1)) + ' K'
        s = s + '\n' + 'MIN37PCTs: ' + str(np.round(np.min(2.15 * TB_s1[:,5] - 1.15 * TB_s1[:,6]),1)) + ' K'
        s = s + '\n' + 'MIN85PCTs: ' + str(np.round(np.min(1.7  * TB_s1[:,7] - 0.7  * TB_s1[:,8]),1)) + ' K'
        s = s + '\n' '----------'
        s = s + '\n' + '89PCT approx. area: ' + str(np.round(area89,1)) + ' km'
        s = s + '\n' + '89PCT total footprints: ' + str(pix89)	
        s = s + '\n' + '45dBZ radar gates: ' + str(np.round(gates45dbz,1)) 
   	 #for ii in range(len( options['MINPCTs_labels'] )):  
        #    s = s+'\n'+options['MINPCTs_labels'][ii]+': '+str(options['MINPCTs'][ii])
        axes[ic].text(62,-10, s, bbox=props, fontsize=10)
        del s, TB_s1
        axes[ic].set_xlabel('Zh (dBZ)')

    #------------------------------------------------------
    # FIGURE histogram de los TBs. 
    MINPCTS_icois = np.zeros((len(RN_inds_parallax), 4)); MINPCTS_icois[:]=np.nan
    for ic in range(len(RN_inds_parallax)):
        TB_s1 = tb_s1_gmi[idx1][TB_inds[ic],:]         
        #TB_s1   = tb_s1_gmi[TB_inds[ic],:]
        MINPCTs = []
        MINPCTs.append(np.round(np.min(2.5  * TB_s1[:,0] - 1.5  * TB_s1[:,1]),1))
        MINPCTs.append(np.round(np.min(2.4  * TB_s1[:,2] - 1.4  * TB_s1[:,3]),1))
        MINPCTs.append(np.round(np.min(2.15 * TB_s1[:,5] - 1.15 * TB_s1[:,6]),1))
        MINPCTs.append(np.round(np.min(1.7  * TB_s1[:,7] - 0.7  * TB_s1[:,8]),1))
        MINPCTS_icois[ic,:] = MINPCTs
        del MINPCTs

    #MINPCTS_icois = MINPCTS_icois.T
    fig = plt.figure(figsize=(20,7)) 
    #-
    barlabels = []
    for ic in range(len(RN_inds_parallax)): 
        if icois[ic] == options['icoi_PHAIL']:
            barlabels.append(labels_plot[ic]+', Phail: '+str(options['phail']))
        else:
            barlabels.append(labels_plot[ic])
    #-
    name = ['MINPCT10','MINPCT19','MINPCT37','MIN89PCT']
    barWidth = 0.15 
    # Set position of bar on X axis   
    br1 = np.arange(len(name)) #---- adjutst!
    plt.bar(br1, MINPCTS_icois[0,:], color='darkblue',  width = barWidth, label=barlabels[0])
    if len(RN_inds_parallax) == 2:
        br2 = [x + barWidth for x in br1] 
        plt.bar(br2, MINPCTS_icois[1,:], color='darkred',   width = barWidth, label=barlabels[1])
    if len(RN_inds_parallax) == 3:
        br2 = [x + barWidth for x in br1] 
        br3 = [x + barWidth for x in br2]
        plt.bar(br2, MINPCTS_icois[1,:], color='darkred',   width = barWidth, label=barlabels[1])
        plt.bar(br3, MINPCTS_icois[2,:], color='darkgreen', width = barWidth, label=barlabels[2])
    plt.ylabel('MINPCT (K)')  
    plt.xticks([r + barWidth for r in range(len(name))], name)   # adjutst! len() 
    plt.legend()
    plt.title('Min. observed PCTs for COI')
    plt.grid(True)

    del radar 

    return
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def ignore_plots():
	#----------------------------------------------------------------------------------------------
        #----- plot [PERHAPS COMMENT! ]  ===> DIFRFERENCE BETWEEN GROUND LEVEL NATIVE GRID AND GRIDDED LEVEL 0 ? 
        nlev = 0 
        start_index = radar.sweep_start_ray_index['data'][0]
        end_index   = radar.sweep_end_ray_index['data'][0]
        lats  = radar.gate_latitude['data'][start_index:end_index]
        lons  = radar.gate_longitude['data'][start_index:end_index]
        TH    = radar.gate_longitude['data'][start_index:end_index]
        azimuths = radar.azimuth['data'][start_index:end_index]
        target_azimuth = azimuths[options['azimuth_ray']]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        radar_gateZ = []
        for i in range(np.array(radar.range['data']).shape[0]):
            radar_gateZ.append(return_altitude(radar.elevation['data'][start_index:end_index][nlev], target_azimuth, np.array(radar.range['data'])[i]/1e3))
        figure_transect_gridded(lons, radarTH, filas, gridded)
	
        return
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def summary_radar_obs(radar, fname, options):  

        # read 
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
        nlev = 0 
        start_index   = radar.sweep_start_ray_index['data'][0]
        end_index   = radar.sweep_end_ray_index['data'][0]
        lats  = radar.gate_latitude['data'][start_index:end_index]
        lons  = radar.gate_longitude['data'][start_index:end_index]
        TH    = radar.gate_longitude['data'][start_index:end_index] 
        	#-------------------------- ZH y contornos y RHO
        fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=[24,6])
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        if 'TH' in radar.fields.keys():  
            THNAME= 'TH'
            RHOHVname='RHOHV'
            pcm1 = axes[0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
        elif 'DBZHCC' in radar.fields.keys():        
            THNAME= 'DBZHCC'
            RHOHVname='RHOHV'
            pcm1 = axes[0].pcolormesh(lons, lats, radar.fields['DBZHCC']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
        elif 'corrected_reflectivity' in radar.fields.keys():        
            THNAME= 'corrected_reflectivity'
            pcm1 = axes[0].pcolormesh(lons, lats, radar.fields['corrected_reflectivity']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
            RHOHVname='copol_correlation_coeff'
        elif 'DBZH' in radar.fields.keys():        
            THNAME= 'DBZH'
            pcm1 = axes[0].pcolormesh(lons, lats, radar.fields['DBZH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
            RHOHVname='RHOHV'
	
        plt.colorbar(pcm1, ax=axes[0])
        axes[0].set_xlim([options['x_supermin'], options['x_supermax']])	
        axes[0].set_ylim([options['y_supermin'], options['y_supermax']])	
        axes[0].set_title('ZH (w/ 45dBZ contour)')
        axes[0].contour(lons[:], lats[:], radar.fields[THNAME]['data'][start_index:end_index][:], [45], colors=(['navy']), linewidths=2);	
        axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200,225], colors=(['black', 'black']), linewidths=1.5);
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
        axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	

        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
        pcm1 = axes[1].pcolormesh(lons, lats, radar.fields[RHOHVname]['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm1, ax=axes[1])
        axes[1].set_xlim([options['x_supermin'], options['x_supermax']])	
        axes[1].set_ylim([options['y_supermin'], options['y_supermax']])
        axes[1].set_title('RHOHV (w/ 45dBZ contour)')
        axes[1].contour(lons[:], lats[:], radar.fields[THNAME]['data'][start_index:end_index][:], [45], colors=(['navy']), linewidths=2);
        CS = axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200,225], colors=(['black', 'black']), linewidths=1.5);
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
        axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        #-------------------------- DOPPLER FOR OVERVIEW - SC
       	if 'VRAD' in radar.fields.keys():  
            VEL = radar.fields['VRAD']['data'][start_index:end_index]
            vel_texture = pyart.retrieve.calculate_velocity_texture(radar, vel_field='VRAD', nyq=39.9)
            radar.add_field('velocity_texture', vel_texture, replace_existing=True)
            velocity_dealiased = pyart.correct.dealias_region_based(radar, vel_field='VRAD', nyquist_vel=39.9,centered=True)
            radar.add_field('corrected_velocity', velocity_dealiased, replace_existing=True)
            VEL_cor = radar.fields['corrected_velocity']['data'][start_index:end_index]
        elif 'corrected_velocity' in radar.fields.keys():  
            VEL_cor = radar.fields['corrected_velocity']['data'][start_index:end_index]
        if 'VRAD' in radar.fields.keys() or 'corrected_velocity' in radar.fields.keys():  
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('doppler')
            pcm1 = axes[2].pcolormesh(lons, lats, VEL_cor, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(pcm1, ax=axes[2], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            axes[2].set_xlim([options['x_supermin'], options['x_supermax']])	
            axes[2].set_ylim([options['y_supermin'], options['y_supermax']])
            axes[2].set_title('Vr corrected (w/ 45dBZ contour)')
            axes[2].contour(lons[:], lats[:], radar.fields[THNAME]['data'][start_index:end_index][:], [45], colors=(['navy']), linewidths=2);
            CS=axes[2].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200,225], colors=(['black', 'black']), linewidths=1.5);
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
            axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
            axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
            axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)

            if len(options['REPORTES_meta'])>0:
                for ireportes in range(len(options['REPORTES_geo'])):
                    axes[2].plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], '*', markeredgecolor='black', markerfacecolor='black', markersize=10, label=options['REPORTES_meta'][ireportes])

        labels_cont = ['GMI 200K contour', 'GMI 225K contour']
        for i in range(len(labels_cont)):
          CS.collections[i].set_label(labels_cont[i])
          if len(options['REPORTES_meta'])>0:
              for ireportes in range(len(options['REPORTES_geo'])):
                  plt.plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], '*', markeredgecolor='black', markerfacecolor='black', markersize=10, label=options['REPORTES_meta'][ireportes])
              plt.legend(fontsize=11) 

        fig.savefig(options['fig_dir']+'PPIs_Summary'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)   
        #plt.close()

        #----------------------------------------------------------------------------------------
        # Test plot figure: General figure with Zh and the countours identified 
        #----------------------------------------------------------------------------------------
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[14,12])
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        pcm1 = axes.pcolormesh(lons, lats, radar.fields[THNAME]['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
        cbar.cmap.set_under(under)
        cbar.cmap.set_over(over)
        axes.grid(True)
        for iPF in range(len(options['lat_pfs'])): 
           axes.plot(options['lon_pfs'][iPF], options['lat_pfs'][iPF], marker='*', markersize=20, markerfacecolor="None",
            markeredgecolor='black', markeredgewidth=2, label='GMI(PF) center') 
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
        axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8, label='ring: 10 km')
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8, label='ring: 50 km')
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8, label='ring: 100 km')
        CS = plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 225], colors=(['k', 'k']), linestyles=(['-','--']), linewidths=1.5);
        plt.xlim([options['xlim_min'], options['xlim_max']])
        plt.ylim([options['ylim_min'], options['ylim_max']])
        labels_cont = ['GMI 200K contour', 'GMI 225K contour']
        for i in range(len(labels_cont)):
          CS.collections[i].set_label(labels_cont[i])
        if len(options['REPORTES_meta'])>0:
            for ireportes in range(len(options['REPORTES_geo'])):
                plt.plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], '*', markeredgecolor='black', markerfacecolor='black', markersize=10, label=options['REPORTES_meta'][ireportes])
        plt.legend(fontsize=11) 
        if options['radar_name'] == 'DOW7':
            general_title='radar at '+options['rfile'][20:24]+' UTC and PF at '+str(options['time_pfs'])+')'	
        else:
            general_title='radar at '+options['rfile'][15:19]+' UTC and PF at '+str(options['time_pfs'])+')'
        plt.suptitle(general_title)

        fig.savefig(options['fig_dir']+'PPIs_MAIN_Summary'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)   
        #plt.close()

        return


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def get_prioritymap(options, radar, grided):

   	# HID priority:  hail (HL), high-density graupel (HG), low-density graupel (LG), aggregates (AG), 
   	# ice crystals (CR) combined with vertically oriented crystals (VI), and a final group that 
   	# combines all liquid-phase hydrometeors and wet snow (WS). 
   	# HID types:           Species #:  
   	# -------------------------------
   	# ------ OJO QUE CERO NO VAS A TENER PQ FILTRASTE ESO ANTES? 
   	# eso es lo que sale de scores, a esto se le suma +1
        # Drizzle                  1  -> rank: 6
        # Rain                     2  -> rank: 6    
        # Ice Crystals             3  -> rank: 4
        # Aggregates               4  -> rank: 3
        # Wet Snow                 5  -> rank: 6
        # Vertical Ice             6  -> rank: 5
        # Low-Density Graupel      7  -> rank: 2
        # High-Density Graupel     8  -> rank: 1
        # Hail                     9  -> rank: 0
        # Big Drops                10 -> rank: 6
   	# -------------------------------
   	# -------------------------------	
   	# una forma puede ser asignando prioridades ... 
       priority = {9: 0, 8: 1, 7: 2, 4: 3, 3: 4, 6: 5, 5: 6, 10: 7, 2: 8, 1: 9, 0: 10}
       nz = grided.fields['HID']['data'].shape[0]
       nx = grided.fields['HID']['data'].shape[1]
       ny = grided.fields['HID']['data'].shape[2]
       priority_map = np.zeros((nx,ny)); priority_map[:] = np.nan
       for ix in range(nx):
           for iy in range(ny):
               #ix = 268
               #iy = 430
               HID_col = np.zeros((grided.fields['HID']['data'][:,ix,iy].shape[0]))
               HID_col[:] = grided.fields['HID']['data'][:,ix,iy]	
               HID_col = HID_col.round()
               priority_map[ix,iy] = sorted(HID_col, key=priority.get)[0]			
       #---- plot hid ppi  
       hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
       cmaphid = colors.ListedColormap(hid_colors)
       fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
       pcm1 = axes.pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:], priority_map, cmap=cmaphid, vmin=0.2, vmax=10)
       axes.set_title('HID priority projection')
       axes.set_xlim([options['xlim_min'], options['xlim_max']])
       axes.set_ylim([options['ylim_min'], options['ylim_max']])
       [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
       axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
       [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
       axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
       [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
       axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
       cbar_HID = plt.colorbar(pcm1, ax=axes, shrink=1.1, label=r'HID')    
       cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	
	
       # agregar: 	
       # read 
       f = h5py.File( options['gmi_dir']+options['gfile'], 'r')
       tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
       lon_gmi = f[u'/S1/Longitude'][:,:] 
       lat_gmi = f[u'/S1/Latitude'][:,:]
       tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
       lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
       lat_s2_gmi = f[u'/S2/Latitude'][:,:]
       f.close()
       for j in range(lon_gmi.shape[1]):
            tb_s1_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
            tb_s1_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan   
            lat_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
            lat_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  
            lon_gmi[np.where(lat_gmi[:,j] >=  options['ylim_max']+5),:] = np.nan
            lon_gmi[np.where(lat_gmi[:,j] <=  options['ylim_min']-5),:] = np.nan  	
       PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 	
       CS = axes.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 225], colors=(['black', 'gray']), linewidths=1.5)
       labels_cont = ['GMI 200K contour', 'GMI 225K contour']
       for i in range(len(labels_cont)):
         CS.collections[i].set_label(labels_cont[i])
       if len(options['REPORTES_meta'])>0:
         for ireportes in range(len(options['REPORTES_geo'])):
             axes.plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], '*', markeredgecolor='black', markerfacecolor='black', markersize=10, label=options['REPORTES_meta'][ireportes])
         plt.legend() 


	
	
       fig.savefig(options['fig_dir']+'PriorityHID.png', dpi=300, transparent=False)  
       #plt.close()
       return priority_map


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def run_general_case(options, era5_file, lat_pfs, lon_pfs, time_pfs, icois, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input):

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    r_dir    = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'
    radar = pyart.io.read(r_dir+options['rfile'])
    
    if options['radar_name'] == 'RMA3':
        PHIORIG = radar.fields['PHIDP']['data'].copy() 
        PHIDP_nans = radar.fields['PHIDP']['data'].copy() 
        PHIDP_nans[np.where(PHIDP_nans.data==radar.fields['PHIDP']['data'].fill_value)] = np.nan
        mask = radar.fields['PHIDP']['data'].data.copy()    
        mask[:] = False
        PHIDP_nans.mask = mask
        radar.add_field_like('PHIDP', 'PHIDP', PHIDP_nans, replace_existing=True)
	
    if options['radar_name'] == 'RMA1':
        PHIORIG = radar.fields['PHIDP']['data'].copy() 
        mask = radar.fields['PHIDP']['data'].data.copy()    
        mask[:] = False
        PHIORIG.mask = mask
        radar.add_field_like('PHIDP', 'PHIDP', PHIORIG, replace_existing=True)
 
    if options['radar_name'] == 'RMA5':
        PHIORIG = radar.fields['PHIDP']['data'].copy() 
        mask = radar.fields['PHIDP']['data'].data.copy()    
        mask[:] = False
        PHIORIG.mask = mask
        radar.add_field_like('PHIDP', 'PHIDP', PHIORIG, replace_existing=True)
	
    if options['radar_name'] == 'RMA4':
        PHIORIG = radar.fields['PHIDP']['data'].copy() 
        mask = radar.fields['PHIDP']['data'].data.copy()    
        mask[:] = False
        PHIORIG.mask = mask
        radar.add_field_like('PHIDP', 'PHIDP', PHIORIG, replace_existing=True)
	
    alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel(era5_dir, era5_file, lat_pfs, lon_pfs) 
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    radar = add_43prop_field(radar)     

    if options['radar_name'] == 'DOW7':
        radar = DOW7_NOcorrect_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        plot_HID_PPI(radar, options, 0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        radar_stacked = stack_ppis(radar, options['files_list'], options, freezing_lev, radar_T, tfield_ref, alt_ref)
        for ic in range(len(xlims_xlims_input)): 
            check_transec(radar, azimuths_oi[ic], lon_pfs, lat_pfs, options)
            plot_rhi_DOW7(radar, options['files_list'], xlims_mins_input[ic], xlims_xlims_input[ic], azimuths_oi[ic], options['ZDRoffset'], freezing_lev, radar_T, options,tfield_ref, alt_ref) 
        grided  = pyart.map.grid_from_radars(radar_stacked, grid_shape=(41, 355, 355), grid_limits=((0.,20000,),  
	(-np.max(radar_stacked.range['data']), np.max(radar_stacked.range['data'])),(-np.max(radar_stacked.range['data']), 
              np.max(radar_stacked.range['data']))), roi_func='dist', min_radius=500.0, weighting_function='BARNES2')  
        gc.collect()
        make_pseudoRHISfromGrid_DOW7(grided, radar, azimuths_oi, labels_PHAIL, xlims_mins_input, xlims_xlims_input, alt_ref, tfield_ref, options)
        HID_priority2D = get_prioritymap(options, radar, grided)

    elif options['radar_name'] == 'CSPR2':
        radar = CSPR2_correct_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        plot_HID_PPI_CSPR2(radar, options, 0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        plot_HID_PPI_CSPR2(radar, options, 1, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        plot_HID_PPI_CSPR2(radar, options, 2, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
	#radar_stacked = stack_ppis_CSPR(radar,  options, freezing_lev, radar_T, tfield_ref, alt_ref)
        #for ic in range(len(xlims_xlims_input)): 
        #    check_transec_CSPR(radar, azimuths_oi[ic], lon_pfs, lat_pfs, options)
        #    plot_rhi_CSPR2(radar, xlims_mins_input[ic], xlims_xlims_input[ic], azimuths_oi[ic], options['ZDRoffset'], freezing_lev, radar_T, options, tfield_ref, alt_ref)
        grided  = pyart.map.grid_from_radars(radar, grid_shape=(41, 440, 440), grid_limits=((0.,20000,),  
	(-np.max(radar.range['data']), np.max(radar.range['data'])),(-np.max(radar.range['data']), 
              np.max(radar.range['data']))), roi_func='dist', min_radius=500.0, weighting_function='BARNES2')  
        gc.collect()
        make_pseudoRHISfromGrid(grided, radar, azimuths_oi, labels_PHAIL, xlims_mins_input, xlims_xlims_input, alt_ref, tfield_ref, options)
        #HID_priority2D = get_prioritymap(options, radar, grided)
	
    else: 
        radar = correct_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        plot_HID_PPI(radar, options, 0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        plot_HID_PPI(radar, options, 1, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        plot_HID_PPI(radar, options, 2, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        for ic in range(len(xlims_xlims_input)):
            check_transec(radar, azimuths_oi[ic], lon_pfs, lat_pfs, options)
            plot_rhi_RMA(radar, xlims_mins_input[ic], xlims_xlims_input[ic], azimuths_oi[ic], options['ZDRoffset'], freezing_lev, radar_T, options)
        # 500m grid! 
        grided  = pyart.map.grid_from_radars(radar, grid_shape=(40, 940, 940), grid_limits=((0.,20000,),   #20,470,470 is for 1km
      		(-np.max(radar.range['data']), np.max(radar.range['data'])),(-np.max(radar.range['data']), np.max(radar.range['data']))), roi_func='dist', min_radius=500.0, weighting_function='BARNES2')  
        gc.collect()
        make_pseudoRHISfromGrid(grided, radar, azimuths_oi, labels_PHAIL, xlims_mins_input, xlims_xlims_input, alt_ref, tfield_ref, options)
        HID_priority2D = get_prioritymap(options, radar, grided)
        gc.collect()
    
    plot_gmi(gmi_dir+options['gfile'], options, radar, lon_pfs, lat_pfs, icois)
    #visual_coi_identification(options, radar, gmi_dir+options['gfile'])
    summary_radar_obs(radar, gmi_dir+options['gfile'], options)
    gc.collect()

    plot_scatter(options, radar, icois, gmi_dir+options['gfile'])
    gc.collect()

    return



#---------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------- 
def firstNonNan(listfloats):
  for item in listfloats:
    if math.isnan(item) == False:
      return item

#---------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------- 





#---------------------------------------------------------------------------------------------- OLD RUN GENERAL additioanl stuff. 
#if len(icois) == 3: 
#[gridded, frezlev, GMI_lon_COI1, GMI_lat_COI1, GMI_tbs1_COI1, RN_inds_COI1, RB_inds_COI1, 
# GMI_lon_COI2, GMI_lat_COI2, GMI_tbs1_COI2, RN_inds_COI2, RB_inds_COI2,
#    GMI_lon_COI3, GMI_lat_COI3, GMI_tbs1_COI3, RN_inds_COI3, RB_inds_COI3] = plot_Zhppi_wGMIcontour(radar, lat_pfs, lon_pfs, 'radar at '+options['rfile'][15:19]+' UTC and PF at '+time_pfs[0]+' UTC', gmi_dir+options['gfile'], 0, options, era5_dir+era5_file, icoi=icois, use_freezingLev=0)
#GMI_tbs1_37 = [GMI_tbs1_COI1[:,5], GMI_tbs1_COI2[:,5], GMI_tbs1_COI3[:,5]]
#RN_inds     = [RN_inds_COI1, RN_inds_COI2, RN_inds_COI3]
#---- plot density figures 
#make_densityPlot(radarTH, radarZDR, RN_inds_COI1, RN_inds_COI2, RN_inds_COI3)
#gridedTH  = gridded.fields['TH']['data'][0,:,:]
#gridedZDR = (gridded.fields['TH']['data'][0,:,:]-gridded.fields['TV']['data'][0,:,:]) - opts['ZDRoffset']
#make_densityPlot(gridedTH, gridedZDR, RB_inds_COI1, RB_inds_COI2, RB_inds_COI3)
#----------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# ACA EMPIEZA EL MAIN! 
#------------------------------------------------------------------------------	

def main_20181111(): 

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'

    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- -de--- ---- ---- ---- ----- ---- ---- ---- 
    # ESTE CASO ELIMINADO - DETRAS DE LAS SIERRAS ... USAR CSPR2
    # CASO RMA1 - 20181111 at 1250: P(hail) = 0.653 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG
    #   2018	11	11	12	50	 -31.83	 -64.53	0.653		274.5656	302.1060	249.4227	190.0948	100.5397	197.7117	209.4600	1
    lon_pfs  = [-64.53]
    lat_pfs  = [-31.83]
    time_pfs = ['1250UTC']
    phail    = [0.653]
    MIN85PCT = [100.5397]
    MIN37PCT = [190.0948]
    MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    MINPCTs  = [274.57, 249.42, 190.09, 100.54, 197.71, 209.46]
    #
    #rfile    = 'cfrad.20181111_124509.0000_to_20181111_125150.0000_RMA1_0301_01.nc'
    rfile     = 'corcsapr2cmacppiM1.c1.20181111.130003.nc' #'corcsapr2cfrppiM1.a1.20181111.130003.nc'
    gfile     = '1B.GPM.GMI.TB2016.20181111-S113214-E130446.026724.V05A.HDF5'
    era5_file = '20181111_13_RMA1.grib'
    # REPORTES TWITTER ...  (de la base de datos de relampago solo a las 2340 en la zona, y en tweets en la madrugada 1216am) 
    reportes_granizo_twitterAPI_geo = [[]]
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -65.5, 'xlim_max': -63.6, 'ylim_min': -33, 'ylim_max': -31.5, 
    	    'ZDRoffset': 0, 'ylim_max_zoom':-30.5, 'rfile': 'CSPR2_data/'+rfile, 'gfile': gfile, 
    	    'window_calc_KDP': 7, 'azimuth_ray': 220, 'x_supermin':-65, 'x_supermax':-64,
    	    'y_supermin':-33, 'y_supermax':-31.5, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181111am/', 
    	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail, 
     	   'icoi_PHAIL': 3, 'radar_name':'CSPR2', 'alternate_azi':[210, 220]}
    icois_input  = [6,6] 
    azimuths_oi  = [30,19]
    labels_PHAIL = ['6[Phail = 0.653]','6[Phail = 0.653]'] 
    xlims_xlims_input  = [80, 80] 
    xlims_mins_input  = [0, 0]		
    run_general_case(opts, era5_file, lat_pfs, lon_pfs, time_pfs, icois_input, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input)
			
    return

def old_main(): 
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO SUPERCELDA: 20180208
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # 	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		 
    #	270.5074	292.9824	242.9253	207.4241	131.1081	198.2514	208.1400
    lon_pfs  = [-64.80]
    lat_pfs  = [-31.83]
    time_pfs = ['2058UTC']
    phail    = [0.534]
    MIN85PCT = [131.1081]
    MIN37PCT = [207.4052]
    MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    MINPCTs  = [270.51, 242.92, 207.42, 131.1081, 198.25, 208.14]
    #rfile   = 'cfrad.20180208_231641.0000_to_20180208_232230.0000_RMA1_0201_01.nc'
    rfile    = 'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5'  #21UTC
    era5_file = '20180208_21_RMA1.grib'
    # REPORTES TWITTER ... 
    # CDB capital (varios en base, e.g. https://t.co/Z94Z4z17Ev)
    # VCP (https://twitter.com/icebergdelsur/status/961717942714028032, https://t.co/RJakJjW8sl) gargatuan hail paper!
    # San Antonio de Arredondo (https://t.co/GJwBLvwHVJ ) > 6 cm
    reportes_granizo_twitterAPI_geo = [[-31.49, -64.54], [-31.42, -64.50], [-31.42, -64.19]]
    reportes_granizo_twitterAPI_meta = ['SAA (1930UTC)', 'VCP (1942UTC)', 'CDB (24UTC)']
    opts = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5, 
	    'ZDRoffset': 4, 'ylim_max_zoom':-30.5, 'rfile': 'RMA1/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 'azimuth_ray': 210, 'x_supermin':-65, 'x_supermax':-64,
	    'y_supermin':-33, 'y_supermax':-31.5, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20180208_RMA1/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail, 
	   'icoi_PHAIL': 3, 'radar_name':'RMA1'}
    icois_input  = [1,3,4] 
    azimuths_oi  = [356,220,192]
    labels_PHAIL = ['1','3[Phail = 0.534]','4'] 
    xlims_xlims_input  = [60, 100, 150] 
    xlims_mins_input  = [10, 40, 60]		
    run_general_case(opts, era5_file, lat_pfs, lon_pfs, time_pfs, icois_input, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input)
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO MSC RMA3 - 20190305: P(hail) = 0.737 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    #   MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		
    # 	278.0306	297.0986	249.9366	164.4755	 75.0826	199.9341	223.8100
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    lat_pfs  = [-25.95] 
    lon_pfs  = [-60.57]
    time_pfs = ['1252'] 
    phail    = [0.737]
    MIN85PCT = [75.0826]
    MIN37PCT = [164.4755] 
    MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    MINPCTs  = [278.03, 249.94, 164.48, 75.08, 223.81] 
    #
    rfile     = 'cfrad.20190305_124638.0000_to_20190305_125231.0000_RMA3_0200_01.nc'
    gfile     = '1B.GPM.GMI.TB2016.20190305-S123614-E140847.028498.V05A.HDF5'
    era5_file = '20190305_13.grib'
    #
    # REPORTES TWITTER ... 
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -63, 'xlim_max': -58, 'ylim_min': -27, 'ylim_max': -23, 'ylim_max_zoom': -24.5, 'ZDRoffset': 3, 
	    'rfile': 'RMA3/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 'azimuth_ray': 210, 
	    'x_supermin':-63, 'x_supermax':-58, 'y_supermin':-27, 'y_supermax':-23, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190305_RMA3/', 
	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail, 
	   'icoi_PHAIL':6, 'radar_name':'RMA3'}
    icois_input  = [6,7] 
    azimuths_oi  = [176,210,30]
    labels_PHAIL = ['6[Phail = 0.737]','7', ''] 
    xlims_xlims_input  = [150, 200, 150] 
    xlims_mins_input  = [0, 0, 0]	
    # OJO. sys_phase no le sirve que haya -9999. no toma masked array! 
    run_general_case(opts, era5_file, lat_pfs, lon_pfs, time_pfs, icois_input, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input)
	
	

    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA1 - 20181214: P(hail) = 0.967 ... USAR DOW7!
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG
    #  	ESTE FUERA DEL RANGO: 2018	12	14	03	09	 -31.30	 -65.99	0.839		268.7292	296.7003	224.6779	169.3689	 89.0871	199.9863	194.1800	1
    # 	2018	12	14	03	09	 -31.90	 -63.11	0.967		260.0201	306.4535	201.8675	133.9975	 71.0844	199.8376	212.5500	1
    # 	ESTE FUERA DEL RANGO: 2018	12	14	03	09	 -32.30	 -61.40	0.998		235.5193	307.7839	130.7862	 80.1157	 45.9117	199.9547	205.9700	1
    # 	ESTE FUERA DEL RANGO: 2018	12	14	03	10	 -33.90	 -59.65	0.863		274.4490	288.9589	239.0672	151.7338	 67.8216	195.6911	196.5800	1
    lon_pfs  = [-63.11] # [-61.40] [-59.65]
    lat_pfs  = [-31.90] # [-32.30] [-33.90]
    time_pfs = ['0310UTC']
    phail    = [0.967] # [0.998] [0.863]
    MIN85PCT = [71.08] # [45.91] [67.82] 
    MIN37PCT = [133.99] # [80.12] [151.73] 
    MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    #MINPCTs  = [268.73, 224.68, 169.37, 89.09, 199.99, 	194.18] 
    MINPCTs  = [260.02, 201.87, 133.99, 71.08, 199.84, 212.55]
    #MINPCTs  = [235.52, 130.79, 80.12, 45.91, 199.95, 205.97]
    #MINPCTs  = [274.45, 239.07, 151.73, 67.82, 195.69, 196.58]
    # 0304 is raining on top ... 'cfrad.20181214_030436.0000_to_20181214_031117.0000_RMA1_0301_01.nc'
    # rfile =  'cfrad.20181214_024550.0000_to_20181214_024714.0000_RMA1_0301_02.nc' 
    # rfile = 'cfrad.20181214_025529.0000_to_20181214_030210.0000_RMA1_0301_01.nc' 
    # USE DOW7 for lowest level
    rfile = 'cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc' 
    # mimic RMA1 file system: 
    counter = 0	    
    radar0      = pyart.io.read_cfradial('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/DOW7/' + 'cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc')
    start_index = radar0.sweep_start_ray_index['data'][0]
    end_index   = radar0.sweep_end_ray_index['data'][0]
    lats0       = radar0.gate_latitude['data'][start_index:end_index]
    files_list = ['cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc',
		  'cfrad.20181214_022019_DOW7low_v176_s02_el1.98_SUR.nc',
		  'cfrad.20181214_022031_DOW7low_v176_s03_el3.97_SUR.nc',
		  'cfrad.20181214_022043_DOW7low_v176_s04_el5.98_SUR.nc',
		  'cfrad.20181214_022055_DOW7low_v176_s05_el7.98_SUR.nc',
		  'cfrad.20181214_022106_DOW7low_v176_s06_el9.98_SUR.nc',
		  'cfrad.20181214_022118_DOW7low_v176_s07_el11.98_SUR.nc',
		  'cfrad.20181214_022130_DOW7low_v176_s08_el13.99_SUR.nc',
		  'cfrad.20181214_022142_DOW7low_v176_s09_el15.97_SUR.nc',
		  'cfrad.20181214_022154_DOW7low_v176_s10_el17.97_SUR.nc',
		  'cfrad.20181214_022206_DOW7low_v176_s11_el19.98_SUR.nc',	  
		  'cfrad.20181214_022218_DOW7low_v176_s12_el21.98_SUR.nc',
		  'cfrad.20181214_022230_DOW7low_v176_s13_el23.99_SUR.nc',
		  'cfrad.20181214_022241_DOW7low_v176_s14_el25.98_SUR.nc',	  
		  'cfrad.20181214_022253_DOW7low_v176_s15_el27.98_SUR.nc',
		  'cfrad.20181214_022305_DOW7low_v176_s16_el29.98_SUR.nc',
		  'cfrad.20181214_022317_DOW7low_v176_s17_el31.98_SUR.nc',
		  'cfrad.20181214_022329_DOW7low_v176_s18_el33.98_SUR.nc',	    
		  'cfrad.20181214_022341_DOW7low_v176_s19_el36.98_SUR.nc',
		  'cfrad.20181214_022353_DOW7low_v176_s20_el40.97_SUR.nc',
		  'cfrad.20181214_022405_DOW7low_v176_s21_el44.98_SUR.nc',
		  'cfrad.20181214_022416_DOW7low_v176_s22_el49.98_SUR.nc']	
    #	
    gfile     = '1B.GPM.GMI.TB2016.20181214-S015009-E032242.027231.V05A.HDF5'
    era5_file = '20181214_03_RMA1.grib'
    # REPORTES TWITTER ... 
    reportes_granizo_twitterAPI_geo = [[-32.19, -64.57]]
    reportes_granizo_twitterAPI_meta = [['0320UTC']]
    opts = {'xlim_min': -65.3, 'xlim_max': -63.3, 'ylim_min': -32.4, 'ylim_max': -31, 
	    'ZDRoffset': 0, 'ylim_max_zoom':-31, 'rfile': 'DOW7/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 'azimuth_ray': 60, 'x_supermin': -65.3, 'x_supermax':-63.3,
	    'y_supermin':-32.4, 'y_supermax':-31, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181214_RMA1/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail, 
	   'icoi_PHAIL': 16, 'radar_name':'DOW7', 'files_list':files_list}
    icois_input  = [15] 
    azimuths_oi  = [300, 273, 450]
    labels_PHAIL = ['', '', ''] 
    xlims_xlims_input  = [80, 80, 80] 
    xlims_mins_input  = [0, 0, 0]		
    run_general_case(opts, era5_file, lat_pfs, lon_pfs, time_pfs, icois_input, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input)
	
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- -de--- ---- ---- ---- ----- ---- ---- ---- 
    # ESTE CASO ELIMINADO - DETRAS DE LAS SIERRAS ... 
    # CASO RMA1 - 20181111 at 1250: P(hail) = 0.653 
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG
    #   2018	11	11	12	50	 -31.83	 -64.53	0.653		274.5656	302.1060	249.4227	190.0948	100.5397	197.7117	209.4600	1
    # lon_pfs  = [-64.53]
    # lat_pfs  = [-31.83]
    # time_pfs = ['1250UTC']
    # phail    = [0.653]
    # MIN85PCT = [100.5397]
    # MIN37PCT = [190.0948]
    # MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    # MINPCTs  = [274.57, 249.42, 190.09, 100.54, 197.71, 209.46]
    #
    # rfile     = 'cfrad.20181111_124509.0000_to_20181111_125150.0000_RMA1_0301_01.nc'
    # gfile     = '1B.GPM.GMI.TB2016.20181111-S113214-E130446.026724.V05A.HDF5'
    # era5_file = '20181111_13_RMA1.grib'
    # REPORTES TWITTER ...  (de la base de datos de relampago
    #reportes_granizo_twitterAPI_geo = [[-31.84, -64.98], [-30.73, -64.82], [-31.66, -64.43], [-30.67, -64.07], [-32.44, -64.40]]
    #reportes_granizo_twitterAPI_meta = ['19UCT', '21UTC', '2340UTC', '0035UTC', '0220UTC']
    # opts = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5, 
    # 	    'ZDRoffset': 1, 'ylim_max_zoom':-30.5, 'rfile': 'RMA1/'+rfile, 'gfile': gfile, 
    #	    'window_calc_KDP': 7, 'azimuth_ray': 220, 'x_supermin':-65, 'x_supermax':-64,
    #	    'y_supermin':-33, 'y_supermax':-31.5, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181111am_RMA1/', 
    #	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    #	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail, 
    # 	   'icoi_PHAIL': 3, 'radar_name':'RMA1'}
    #icois_input  = [2,3] 
    #azimuths_oi  = [215,110]
    #labels_PHAIL = ['2 []','3[Phail = ]'] 
    #xlims_xlims_input  = [150, 150] 
    #xlims_mins_input  = [0, 0]		
    #run_general_case(opts, era5_file, lat_pfs, lon_pfs, time_pfs, icois_input, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input)

	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA1 - 20190308: P(hail) = 0.895
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG
    #	2019	03	08	02	04	 -30.75	 -63.74	0.895		271.6930	298.6910	241.9306	147.7273	 62.1525	199.0994	226.0100	1
    lon_pfs  = [-63.74]
    lat_pfs  = [-30.75]
    time_pfs = ['0204UTC']
    phail    = [0.895]
    MIN85PCT = [62.15]
    MIN37PCT = [147.72]
    MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    MINPCTs  = [271.69, 241.93, 147.72, 62.15, 199.09, 226.01]
    #
    rfile    = 'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc'
    gfile     = '1B.GPM.GMI.TB2016.20190308-S004613-E021846.028537.V05A.HDF5'
    era5_file = '20190308_02_RMA1.grib'
    # REPORTES TWITTER ...  (de la base de datos de relampago solo a las 2340 en la zona, y en tweets en la madrugada 1216am) 
    reportes_granizo_twitterAPI_geo = [[]]
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -65.2, 'xlim_max': -62, 'ylim_min': -33, 'ylim_max': -30, 
    	    'ZDRoffset': 0.5, 'ylim_max_zoom':-30.5, 'rfile': 'RMA1/'+rfile, 'gfile': gfile, 
    	    'window_calc_KDP': 7, 'azimuth_ray': 50, 'x_supermin':-65, 'x_supermax':-62,
    	    'y_supermin':-32, 'y_supermax':-30, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190308/', 
    	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail, 
     	   'icoi_PHAIL': 3, 'radar_name':'RMA1'}
    icois_input  = [3,3] 
    azimuths_oi  = [50,30]
    labels_PHAIL = ['3 []','3 []'] 
    xlims_xlims_input  = [160, 160] 
    xlims_mins_input  = [0,0]		
    run_general_case(opts, era5_file, lat_pfs, lon_pfs, time_pfs, icois_input, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input)
		
	
    return
	


def main_RMA5_20200815(): 
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
	
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA5 - 20200815: P(hail) = 0.727
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG
    #	2020	08	15	02	15	 -24.17	 -55.94	0.547		276.0569	284.5401	247.3784	192.5272	110.7962	196.9116	144.4400	1
    #	2020	08	15	02	15	 -25.28	 -54.11	0.725		273.2686	290.1380	241.5902	181.1631	101.1417	199.9028	108.2200	1
    lon_pfs  = [ -54.11 ]
    lat_pfs  = [ -25.28 ]
    time_pfs = ['0215UTC']
    phail    = [ 0.725 ]
    MIN85PCT = [ 101.14]
    MIN37PCT = [ 181.16]
    MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    MINPCTs  = [273.23, 241.59, 181.16, 101.14, 199.90, 108.22] 
    rfile    = 'cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20200815-S015947-E033219.036720.V05A.HDF5'
    era5_file = '20200815_02.grib'
    # REPORTES TWITTER ... el 0814 aprox. 23HL
    # https://www.facebook.com/watch/?v=966163750473561 
    # https://revistacodigos.com/fuerte-temporal-de-lluvias-y-granizo-azoto-a-distintas-localidades-de-misiones/ 
    # https://www.elterritorio.com.ar/noticias/2020/08/14/672043-la-tormenta-con-granizo-afecto-a-varios-barrios-de-jardin-america
    # WANDA (-25.93, -54.57)
    # Jardin de america (-27.03, -55.24)
    # San vicente (-35.03, -58.43)
    # Salto Encantado (-27.06, -54.83)
    # Posadas (-27.39, -55.93)
    # Puerto Leoni (-26.98, -55.16)
    # Colonia Polana (-26.98, -55.32)
    # El Dorado (-26.41, -54.66)
    #reportes_granizo_twitterAPI_geo = [[-25.93, -54.57], [-27.03, -55.24], [-35.03, -58.43], [-27.06, -54.83], [-27.39, -55.93], [-26.98, -55.16], 
    #				      [-26.98, -55.32], [-26.41, -54.66]]
    #    reportes_granizo_twitterAPI_meta = ['Wanda', 'Jardin de America', 'San Vicente', 'Salto Encantado','Posadas','Puerto Leoni',
    #					'Colonia Polana','El Dorado']
    # En horas de la tarde/noche: Jardin de america. 
    # https://www.primeraedicion.com.ar/nota/100320388/temporal-de-granizo-afecto-a-la-zona-centro-de-misiones/: 
    # El evento climático tuvo a su vez, gran actividad eléctrica. En Jardín América contaron que en los barrios 162 Viviendas, 
    # El Ceibo y Lomas de Jardín los hielos eran del tamaño de una pelota de tenis. 
    # En las localidades aledañas como Leoni e Hipólito Yrigoyen también hubo reportes de daños.
    reportes_granizo_twitterAPI_geo = [[-25.93, -54.57], [-27.03, -55.24]] 
    reportes_granizo_twitterAPI_meta = ['Wanda', 'Jardin de America']
    opts = {'xlim_min': -55.0, 'xlim_max': -52.0, 'ylim_min': -27.5, 'ylim_max': -25.0, 
	    'ZDRoffset': 2, 'ylim_max_zoom':-25.0, 'rfile': 'RMA5/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 'azimuth_ray': 50, 'x_supermin':-55, 'x_supermax':-52,
	    'y_supermin':-27, 'y_supermax':-25, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20200815_RMA5/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail, 
	   'icoi_PHAIL': 3, 'radar_name':'RMA5','alternate_azi':[331, 335, 50]}
    icois_input  = [7,7,7] 
    azimuths_oi  = [331,335, 50]
    labels_PHAIL = ['[Phail = 0.725]','[Phail = 0.725]', '[]'] 
    xlims_xlims_input  = [190, 190, 150] 
    xlims_mins_input  = [0, 0, 0]		
    run_general_case(opts, era5_file, lat_pfs, lon_pfs, time_pfs, icois_input, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input)
	
    return

	
def main_RMA4_20180209(): 

	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA4 - 20180209: P(hail) = 0.762
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG
    #	2018	02	09	20	05	 -27.92	 -60.18	0.762		281.3578	304.4615	234.4998	165.5130	 71.3825	199.8727	199.5800	1

    lon_pfs  = [-60.18]
    lat_pfs  = [-27.92]
    time_pfs = ['2005UTC']
    phail    = [0.762]
    MIN85PCT = [71.38]
    MIN37PCT = [165.51]
    MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    MINPCTs  = [281.36, 234.49, 165.51, 71.38, 199.87, 199.58]
    rfile    = 'cfrad.20180209_200449.0000_to_20180209_201043.0000_RMA4_0200_01.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20180209-S184820-E202054.022451.V05A.HDF5' 
    era5_file = '20180209_20_RMA4.grib' 
    # REPORTES TWITTER ... 
    # CDB capital (varios en base, e.g. https://t.co/Z94Z4z17Ev)
    # VCP (https://twitter.com/icebergdelsur/status/961717942714028032, https://t.co/RJakJjW8sl) gargatuan hail paper!
    # San Antonio de Arredondo (https://t.co/GJwBLvwHVJ ) > 6 cm
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 
	    'ZDRoffset': 0,  # >>>FIND!! 
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 'azimuth_ray': 210, 
	    'x_supermin':-61.5, 'x_supermax':-56.5, 'y_supermin':-29.5, 'y_supermax':-26, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20180209_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail, 
	   'icoi_PHAIL': 3, 'radar_name':'RMA4','alternate_azi':[190, 225]}
    icois_input  = [1,1] 
    azimuths_oi  = [190,225]
    labels_PHAIL = ['3[Phail = 0.762]',''] 
    xlims_xlims_input  = [150,150] 
    xlims_mins_input  = [0,0]		
    run_general_case(opts, era5_file, lat_pfs, lon_pfs, time_pfs, icois_input, azimuths_oi, labels_PHAIL, xlims_xlims_input, xlims_mins_input)
	
    return
	
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA4 - 20181001: P(hail) = 0.965
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG

    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA4 - 20181031: P(hail) = 0.931
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG

    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA4 - 20181215: P(hail) = 0.747
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG

								
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA4 - 20181218: P(hail) = 0.964, 0.596
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA4 - 20190209: P(hail) = 0.989
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG
		
	
	
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA8 - 20181112: P(hail) = 0.740
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG

	
	
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # CASO RMA8 - 20181112: P(hail) = 0.758
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 	
    #	YEAR	MONTH	DAY	HOUR	MIN	  LAT	LON	P_hail_BC2019	MIN10PCT	MAX10PCT	MIN19PCT	MIN37PCT	MIN85PCT	MAX85PCT	MIN165V		FLAG

	
	
	
	
	
	
	
	
	
	
	
	
		
		
    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/'+rfile) 
    #------
    # copy PHIDP for sysphase
    PHIDP_nanMasked = radar.fields['PHIDP']['data'].copy() 
    PHIDP_nanMasked[np.where(PHIDP_nanMasked==radar.fields['PHIDP']['data'].fill_value)] = 0
    radar.add_field_like('PHIDP', 'PHIDP_unmasked', PHIDP_nanMasked)
    sys_phase  = pyart.correct.phase_proc.det_sys_phase(radar, ncp_lev=0.8, rhohv_lev=0.7, 
							ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP_unmasked')
    # replace PHIDP w/ np.nan
    PHIDP_nans = radar.fields['PHIDP']['data'].copy() 
    PHIDP_nans[np.where(PHIDP_nans.data==radar.fields['PHIDP']['data'].fill_value)] = np.nan
    radar.add_field_like('PHIDP', 'PHIDP', PHIDP_nans, replace_existing=True)
    dphi, uphi, corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], 
		radar.fields['RHOHV']['data'], radar.fields['TH']['data'], sys_phase, 280)    

    #------------------------------------------------------------------------------
    #-----------------------------------------
    # calcular todo junto (sys_phase y correcciones con mismo campo!) 
	
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=[14,7])
    pc0 = axes[0].pcolormesh(lons,lats, PHIDP_nans[start_index:end_index]); plt.colorbar(pc0, ax=axes[0]); axes[0].set_title('limpio para calcular sys phase')
    pc1 = axes[1].pcolormesh(lons,lats, rhv[start_index:end_index], vmin=0.7, vmax=1.0); plt.colorbar(pc1, ax=axes[1]); axes[1].set_title('RHOHV')
    pc2 = axes[2].pcolormesh(lons,lats, PHIORIG[start_index:end_index]); plt.colorbar(pc2, ax=axes[2]); axes[2].set_title('dato crudo')

    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA3/'+rfile)
    sys_phase = get_sys_phase_simple(radar)
    # replace PHIDP w/ np.nan
    PHIORIG = radar.fields['PHIDP']['data'].copy() 
    PHIDP_nans = radar.fields['PHIDP']['data'].copy() 
    PHIDP_nans[np.where(PHIDP_nans.data==radar.fields['PHIDP']['data'].fill_value)] = np.nan
    rhv = radar.fields['RHOHV']['data'].copy()
    z_h = radar.fields['TH']['data'].copy()
    #PHIDP_nans = np.where( (rhv>0.7) & (z_h>30), PHIDP_nans, np.nan)
    radar.add_field_like('PHIDP', 'PHIDP', PHIDP_nans, replace_existing=True)
    # check corrections 
    dphi, uphi, corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], 
		radar.fields['RHOHV']['data'], radar.fields['TH']['data'], sys_phase, 280)    

    #-EJEMPLO de azimuth
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[210]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
    #-figure
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True,figsize=[14,7])
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['RHOHV']['data'][start_index:end_index][filas,:])*100, '-k', label='RHOHV')
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['TH']['data'][start_index:end_index][filas,:]), '-r', label='ZH')
    axes[0].legend()
    axes[1].plot(radar.range['data']/1e3, np.ravel(radar.fields['PHIDP']['data'][start_index:end_index][filas,:]), 'or', label='obs. phidp')
    axes[1].plot(radar.range['data']/1e3, np.ravel(dphi[start_index:end_index][filas,:]), '*b', label='despeckle phidp'); 
    axes[1].plot(radar.range['data']/1e3, np.ravel(uphi[start_index:end_index][filas,:]), color='darkgreen', label='unfolded phidp');
    #axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]), color='magenta', label='phidp corrected');
    #plt.plot(radar.range['data']/1e3, np.ravel(uphidp_2[filas,:]), color='k', label='uphidp2');
    plt.legend()	


	
	
	
	
	
	
	
	
	
	
	
    # TESTING NLEV=0
    nlev = 0 
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    TH    = radar.fields['TH']['data'][start_index:end_index]
    TV    = radar.fields['TV']['data'][start_index:end_index]
    RHOHV = radar.fields['RHOHV']['data'][start_index:end_index]
    PHIDP = radar.fields['PHIDP_unmasked']['data'][start_index:end_index]
    plt.pcolormesh(lons,lats, PHIDP[start_index:end_index])
    plt.contour(lons, lats, TH, [45], colors='k')
    plt.colorbar()
    #
    # 	



	
	
	
	
	
    radar_rma1 = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' )
    PHIDP_rma1 = radar_rma1.fields['PHIDP']['data']
    start_index_rma1 = radar_rma1.sweep_start_ray_index['data'][nlev]
    end_index_rma1   = radar_rma1.sweep_end_ray_index['data'][nlev]
    lats_rma1  = radar_rma1.gate_latitude['data'][start_index:end_index]
    lons_rma1  = radar_rma1.gate_longitude['data'][start_index:end_index]
    plt.pcolormesh(lons,lats, PHIDP_rma1[start_index_rma1:end_index_rma1])
	

	
    # GRAFICAR
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    plt.pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.pcolormesh(lons, lats, TH.mask); plt.colorbar()
    #- USAR NUEVAS MASCARAS!!!
    radar = add_field_to_radar_object(radar.fields['RHOHV']['data'].data, radar, 'RHOHV_new', 
                         'Cross correlation ratio (RHOHV)', 'cross_correlation_ratio_hv', 'RHOHV')
		
		






    print(sys_phase)
	

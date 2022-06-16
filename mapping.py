from matplotlib import cm;
from os import listdir
import pyart
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from shapely.geometry import Polygon
import matplotlib
import h5py
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import platform
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from os.path import isfile, join
import pandas as pd
from pyart.correct import phase_proc
import xarray as xr
from pyart.core.transforms import antenna_to_cartesian
from copy import deepcopy
from csu_radartools import csu_fhc
import matplotlib.colors as colors
import wradlib as wrl    
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.path import Path
from cycler import cycler
import seaborn as sns

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
        vmax = 0.7
        max = 0.71
        intt = 0.1
        N = (vmax-vmin)/intt
        cmap = discrete_cmap(10, 'jet')
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
        cmap = discrete_cmap(int(N), 'jet')
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
def plot_Zhppi_wGMIcontour(radar, lat_pf, lon_pf, general_title, fname, nlev, options, era5_file, icoi, use_freezingLev):

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
        	inds_1 = hull_path.contains_points(datapts)
        	inds_RN1 = hull_path.contains_points(datapts_RADAR_NATIVE)
        	inds_RB1 = hull_path.contains_points(datapts_RADAR_BARNES)
		if ii==1:
        	inds_2 = hull_path.contains_points(datapts)
        	inds_RN2 = hull_path.contains_points(datapts_RADAR_NATIVE)
        	inds_RB2 = hull_path.contains_points(datapts_RADAR_BARNES)
		if ii==2:
        	inds_3 = hull_path.contains_points(datapts)
        	inds_RN3 = hull_path.contains_points(datapts_RADAR_NATIVE)
        	inds_RB3 = hull_path.contains_points(datapts_RADAR_BARNES)


    plt.xlim([options['xlim_min'], options['xlim_max']])
    plt.ylim([options['ylim_min'], options['ylim_max']])

    plt.suptitle(general_title, fontsize=14)

    #---- NEW FIGURE 
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

    # NEW FIGURE 
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
    CS1 = axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
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
    # axes[0].plot(np.ravel(lons)[inds_RN1], np.ravel(lats)[inds_RN1], 'x', markersize=10, markerfacecolor='black')
    #axes[0].plot(np.ravel(grided.point_longitude['data'][0,:,:])[inds_RB1], np.ravel(grided.point_latitude['data'][0,:,:])[inds_RB1], 'x', markersize=10, markerfacecolor='black')
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
    labels = ["200 K","240 K"] 
    for i in range(len(labels)):
        CS1.collections[i].set_label(labels[i])
    axes[1].legend(loc='upper left', fontsize=12)

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
	drho_ = radar.fields['RHO']['data']
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
    axes[2].set_title('HID GRIDDED (Freezing level) '+str(round(grided.z['data'][frezlev]/1e3,2))+' km)')
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
	
			
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True,
                        figsize=[14,4])
    pcm1 = axes[0].pcolormesh(lons, lats, radar.fields['RHOHV']['data'][start_index:end_index])
    axes[0].set_title('RHOHV radar nlev 0 PPI')
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
	axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);
	[lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
	plt.colorbar(pcm1, ax=axes[0])
	
	pcm1 = axes[1].pcolormesh(grided.point_longitude['data'][0,:,:], grided.point_latitude['data'][0,:,:], grided.fields['RHOHV']['data'][0,:,:])
    axes[1].set_title('RHOHV GRIDDED 0 km')
    axes[1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1].set_ylim([options['ylim_min'], options['ylim_max']])
	axes[1].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);		
	[lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)	
	plt.colorbar(pcm1, ax=axes[1])

    pcm1 = axes[2].pcolormesh(grided.point_longitude['data'][frezlev,:,:], grided.point_latitude['data'][frezlev,:,:], grided.fields['RHOHV']['data'][0,:,:])
    axes[2].set_title('RHOHV GRIDDED (Freezing level) '+str(round(grided.z['data'][frezlev]/1e3,2))+' km)')
    axes[2].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[2].set_ylim([options['ylim_min'], options['ylim_max']])
	axes[2].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200, 240], colors=(['k','k']), linewidths=1.5);			
	[lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[2].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
	plt.colorbar(pcm1, ax=axes[2])

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
    opts = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5, 
	    'ZDRoffset': 4}
    era5_file = '20180208_21_RMA1.grib'
    # Frezing level:     
    ERA5_field = xr.load_dataset(era5_dir+era5_file, engine="cfgrib")	
    elemj      = find_nearest(ERA5_field['latitude'], lat_pfs[0])
    elemk      = find_nearest(ERA5_field['longitude'], lon_pfs[0])
    tfield_ref = ERA5_field['t'][:,elemj,elemk] - 273 # convert to C
    geoph_ref  = (ERA5_field['z'][:,elemj,elemk])/9.80665
    # Covert to geop. height (https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
    Re         = 6371*1e3
    alt_ref    = (Re*geoph_ref)/(Re-geoph_ref) 
		#-------------------------- DONT CHANGE
    radar = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile)
    radar_1 = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile_1)
    radar_2 = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/'+rfile_2)
    #------
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


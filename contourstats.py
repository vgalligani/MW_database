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

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------     
def plot_gmi(fname, options, radar, lon_pfs, lat_pfs, icoi):

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

    elif options['radar_name'] == 'RMA4':
        reflectivity_name = 'TH'   
        nlev = 0 
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]

    elif options['radar_name'] == 'RMA8':
        reflectivity_name = 'TH'   
        nlev = 0 
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]

   elif options['radar_name'] == 'RMA3':
        reflectivity_name = 'TH'   
        nlev = 0 
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]
	
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
    idx2 = (lat_s2_gmi>=options['ylim_min']-5) & (lat_s2_gmi<=options['ylim_max']+5) & (lon_s2_gmi>=options['xlim_min']-5) & (lon_s2_gmi<=options['xlim_max']+5)

    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    S2_sub_lat = np.where(idx2 != False, S2_sub_lat, np.nan) 
    S2_sub_lon = np.where(idx2 != False, S2_sub_lon, np.nan) 
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    for i in range(tb_s2_gmi.shape[2]):
        S2_sub_tb[:,:,i]  = np.where(np.isnan(S2_sub_lon) != 1, tb_s2_gmi[:,:,i], np.nan)	
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)
    
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

    p1 = ax1.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.03])   # [left, bottom, width, height] or Bbox 
    cbar = fig.colorbar(im, cax=ax_cbar, shrink=0.8, ticks=np.arange(50,300,50), extend='both', orientation="horizontal", label='TBV (K)')   

    #fig.savefig(options['fig_dir']+'GMI_basicTBs.png', dpi=300, transparent=False)  
    
    #----------------------------------------------------------------------------------------
    # NEW FIGURE. solo dos paneles: Same as above but plt lowest level y closest to freezing level!
    #----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[7,6])
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    axes.pcolormesh(lons, lats, ZH, cmap=cmap, vmax=vmax, vmin=vmin)
    axes.set_title('Ground Level')
    axes.set_xlim([options['xlim_min'], options['xlim_max']])
    axes.set_ylim([options['ylim_min'], options['ylim_max']])
    # -----
    # CONTORNO CORREGIDO POR PARALAJE Y PODER CORRER LOS ICOIS, simplemente pongo nans fuera del area de interes ... 
    contorno89 = axes.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
	
    datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] )) 
    TB_inds = get_contour_info(contorno89, icoi, datapts)	
    for ii in range(len(icoi)): 
    
        if ii==0:
            #inds_1  = concave_path.contains_points(datapts)
            #axes.plot(S1_sub_lon[inds_1], S1_sub_lat[inds_1], 'o', markersize=10, markerfacecolor='black')
            axes.plot(lon_gmi[:,:][idx1][TB_inds[ii]], lat_gmi[:,:][idx1][TB_inds[ii]], 'o', markersize=10, markerfacecolor='black')
            dummy = axes.plot(np.nan, np.nan, 'o', markersize=20, markerfacecolor='black', label='icoi:'+str(icoi[0]))
        if ii==1:
            #inds_2  = concave_path.contains_points(datapts)
            #axes.plot(S1_sub_lon[inds_2], S1_sub_lat[inds_2], 'o', markersize=10, markerfacecolor='darkblue')
            axes.plot(lon_gmi[:,:][idx1][TB_inds[ii]], lat_gmi[:,:][idx1][TB_inds[ii]], 'o', markersize=10, markerfacecolor='darkblue')
            dummy = axes.plot(np.nan, np.nan, 'o', markersize=20, markerfacecolor='darkblue', label='icoi:'+str(icoi[1]))
        if ii==2:
            #inds_3  = concave_path.contains_points(datapts)
            #axes.plot(S1_sub_lon[inds_3], S1_sub_lat[inds_3], 'o', markersize=10, markerfacecolor='darkred')
            axes.plot(lon_gmi[:,:][idx1][TB_inds[ii]], lat_gmi[:,:][idx1][TB_inds[ii]], 'o', markersize=10, markerfacecolor='darkred')
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
        if ii == 3:
            inds_4   = concave_path.contains_points(datapts_in)
            TB_inds      = [inds_1, inds_2, inds_3, inds_4]   

    return TB_inds



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_scatter_4icois(options, radar, icois, fname):

    # ojo que aca agarro los verdaderos PCTMIN, no los que me pasó Sarah B. que estan 
    # ajustados a TMI footprints. 
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
    if test_this == 1: 
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[14,12])
        if 'TH' in radar.fields.keys():  
            radarTH = radar.fields['TH']['data'][start_index:end_index]
            radarZDR = (radar.fields['TH']['data'][start_index:end_index])-(radar.fields['TV']['data'][start_index:end_index])-options['ZDRoffset']
        elif 'DBZH' in radar.fields.keys():
            radarTH = radar.fields['DBZH']['data'][start_index:end_index]
            radarZDR = radar.fields['DBZH']['data'][start_index:end_index]-radar.fields['DBZV']['data'][start_index:end_index]
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
		
    if len(icois)==1:
        colors_plot = ['k']
        labels_plot = [str('icoi=')+str(icois[0])] 	
	
    if len(icois)==2:
        colors_plot = ['k', 'darkblue']
        labels_plot = [str('icoi=')+str(icois[0]), str('icoi=')+str(icois[1])] 
	
    if len(icois)==3:
        colors_plot = ['k', 'darkblue', 'darkred']
        labels_plot = [str('icoi=')+str(icois[0]), str('icoi=')+str(icois[1]), str('icoi=')+str(icois[2])] 

    if len(icois)==4:
        colors_plot = ['k', 'darkblue', 'darkred', 'darkgreen']
        labels_plot = [str('icoi=')+str(icois[0]), str('icoi=')+str(icois[1]), str('icoi=')+str(icois[2]), str('icoi=')+str(icois[3])] 
		
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
    if test_this == 1: 	
    	fig = plt.figure(figsize=(20,7)) 
    	plt.pcolormesh(lons, lats, radarTH, cmap=cmap, vmin=vmin, vmax=vmax)
    	for ic in range(len(GMI_tbs1_37)):
        	plt.plot(lon_gmi[:,:][idx1][TB_inds[ic]], lat_gmi[:,:][idx1][TB_inds[ic]],'x' );    
        	plt.plot( np.ravel(lons)[RN_inds_parallax[ic]], 	np.ravel(lats)[RN_inds_parallax[ic]], 'om')
    	plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200], colors=(['r']), linewidths=1.5);
    	plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    	plt.xlim([options['xlim_min'], options['xlim_max']]) 
    	plt.ylim([options['ylim_min'], options['ylim_max']])

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
        print('MIN10PCTs: '  +str(np.min(2.5  * TB_s1[:,0] - 1.5  * TB_s1[:,1])) ) 
        print('MIN19PCTs: '  +str(np.min(2.4  * TB_s1[:,2] - 1.4  * TB_s1[:,3])) ) 
        print('MIN37PCTs: '  +str(np.min(2.15 * TB_s1[:,5] - 1.15 * TB_s1[:,6])) ) 
        print('MIN85PCTs: '  +str(np.min(1.7  * TB_s1[:,7] - 0.7  * TB_s1[:,8])) ) 
    plt.grid(True)
    plt.legend()
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
    fig, axes = plt.subplots(nrows=4, ncols=1, constrained_layout=True,
                            figsize=[14,12])
    vmax_sample = [] 
    save_area = []
    save_pixels = []
    save_gates = []
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
        save_area.append(np.round(area89,1))
        save_pixels.append(np.round(gates45dbz,1))
        save_gates.append(np.round(gates45dbz,1))
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
    if len(RN_inds_parallax) == 4:
        br2 = [x + barWidth for x in br1] 
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        plt.bar(br2, MINPCTS_icois[1,:], color='darkred',   width = barWidth, label=barlabels[1])
        plt.bar(br3, MINPCTS_icois[2,:], color='darkgreen', width = barWidth, label=barlabels[2])	
        plt.bar(br4, MINPCTS_icois[3,:], color='black', width = barWidth, label=barlabels[3])	
    plt.ylabel('MINPCT (K)')  
    plt.xticks([r + barWidth for r in range(len(name))], name)   # adjutst! len() 
    plt.legend()
    plt.title('Min. observed PCTs for COI')
    plt.grid(True)


    # GUARDAR EN UN SOLO POOL DATOS DONDE PHAIL>0.5, Y OTRO GRAN POOL DONDE NO ... 
    PCTarray_PHAIL   = [] #np.zeros((len(options['icoi_PHAIL']), 4));
    PCTarray_NOPHAIL = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL']) , 4));
    ZHarray_PHAIL    = [] #np.zeros((len(options['icoi_PHAIL']), np.ravel(radarZDR).shape[0] ));
    ZHarray_NOPHAIL  = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL']), np.ravel(radarZDR).shape[0]));
    ZDRarray_PHAIL   = [] #np.zeros((len(options['icoi_PHAIL']), np.ravel(radarZDR).shape[0]));
    ZDRarray_NOPHAIL = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL']) , np.ravel(radarZDR).shape[0]));
    AREA_PHAIL       = [] #np.zeros((len(options['icoi_PHAIL'])));
    AREA_NOPHAIL     = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL'])));
    PIXELS_PHAIL     = [] #np.zeros((len(options['icoi_PHAIL'])));
    PIXELS_NOPHAIL   = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL']) ));
    GATES_PHAIL      = [] #np.zeros((len(options['icoi_PHAIL'])));
    GATES_NOPHAIL    = [] #np.zeros(( len(RN_inds_parallax)- len(options['icoi_PHAIL'])));

    for ic in range(len(RN_inds_parallax)): 
        if icois[ic] == options['icoi_PHAIL'][0]:
            PCTarray_PHAIL.append( [MINPCTS_icois[ic,:]])
            ZHarray_PHAIL.append( np.ravel(radarTH)[RN_inds_parallax[ic]].data )
            ZDRarray_PHAIL.append( np.ravel(radarTH)[RN_inds_parallax[ic]].data )
            AREA_PHAIL.append( save_area[ic] )
            PIXELS_PHAIL.append( save_pixels[ic] )
            GATES_PHAIL.append( save_gates[ic] )

        else:
            PCTarray_NOPHAIL.append( [MINPCTS_icois[ic,:]])
            ZHarray_NOPHAIL.append( np.ravel(radarTH)[RN_inds_parallax[ic]].data )
            ZDRarray_NOPHAIL.append( np.ravel(radarTH)[RN_inds_parallax[ic]].data )
            AREA_NOPHAIL.append(   save_area[ic] )
            PIXELS_NOPHAIL.append( save_pixels[ic] )
            GATES_NOPHAIL.append(  save_gates[ic] )
		
    # NOPHAIL
    PCTarray_NOPHAIL_out = np.zeros([len(PCTarray_NOPHAIL),4]); PCTarray_NOPHAIL_out[:] = np.nan
    for ilist in range(len(PCTarray_NOPHAIL)): 
       PCTarray_NOPHAIL_out[ilist,:] = PCTarray_NOPHAIL[ilist][0]
    # PHAIL
    PCTarray_PHAIL_out = np.zeros([len(PCTarray_PHAIL),4]); PCTarray_PHAIL_out[:] = np.nan
    for ilist in range(len(PCTarray_PHAIL)): 
       PCTarray_PHAIL_out[ilist,:] = PCTarray_PHAIL[ilist][0]  

    del radar 

    return  PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL

#----------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_scatter_4icois_morethan1OFINTEREST(options, radar, icois, fname):

    # ojo que aca agarro los verdaderos PCTMIN, no los que me pasó Sarah B. que estan 
    # ajustados a TMI footprints. 
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
    if test_this == 1: 
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[14,12])
        if 'TH' in radar.fields.keys():  
            radarTH = radar.fields['TH']['data'][start_index:end_index]
            radarZDR = (radar.fields['TH']['data'][start_index:end_index])-(radar.fields['TV']['data'][start_index:end_index])-options['ZDRoffset']
        elif 'DBZH' in radar.fields.keys():
            radarTH = radar.fields['DBZH']['data'][start_index:end_index]
            radarZDR = radar.fields['DBZH']['data'][start_index:end_index]-radar.fields['DBZV']['data'][start_index:end_index]
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
		
    if len(icois)==1:
        colors_plot = ['k']
        labels_plot = [str('icoi=')+str(icois[0])] 	
	
    if len(icois)==2:
        colors_plot = ['k', 'darkblue']
        labels_plot = [str('icoi=')+str(icois[0]), str('icoi=')+str(icois[1])] 
	
    if len(icois)==3:
        colors_plot = ['k', 'darkblue', 'darkred']
        labels_plot = [str('icoi=')+str(icois[0]), str('icoi=')+str(icois[1]), str('icoi=')+str(icois[2])] 

    if len(icois)==4:
        colors_plot = ['k', 'darkblue', 'darkred', 'darkgreen']
        labels_plot = [str('icoi=')+str(icois[0]), str('icoi=')+str(icois[1]), str('icoi=')+str(icois[2]), str('icoi=')+str(icois[3])] 
		
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
    if test_this == 1: 	
    	fig = plt.figure(figsize=(20,7)) 
    	plt.pcolormesh(lons, lats, radarTH, cmap=cmap, vmin=vmin, vmax=vmax)
    	for ic in range(len(GMI_tbs1_37)):
        	plt.plot(lon_gmi[:,:][idx1][TB_inds[ic]], lat_gmi[:,:][idx1][TB_inds[ic]],'x' );    
        	plt.plot( np.ravel(lons)[RN_inds_parallax[ic]], 	np.ravel(lats)[RN_inds_parallax[ic]], 'om')
    	plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200], colors=(['r']), linewidths=1.5);
    	plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    	plt.xlim([options['xlim_min'], options['xlim_max']]) 
    	plt.ylim([options['ylim_min'], options['ylim_max']])

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
        print('MIN10PCTs: '  +str(np.min(2.5  * TB_s1[:,0] - 1.5  * TB_s1[:,1])) ) 
        print('MIN19PCTs: '  +str(np.min(2.4  * TB_s1[:,2] - 1.4  * TB_s1[:,3])) ) 
        print('MIN37PCTs: '  +str(np.min(2.15 * TB_s1[:,5] - 1.15 * TB_s1[:,6])) ) 
        print('MIN85PCTs: '  +str(np.min(1.7  * TB_s1[:,7] - 0.7  * TB_s1[:,8])) ) 
    plt.grid(True)
    plt.legend()
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
    fig, axes = plt.subplots(nrows=4, ncols=1, constrained_layout=True,
                            figsize=[14,12])
    vmax_sample = [] 
    save_area = []
    save_pixels = []
    save_gates = []
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
        save_area.append(np.round(area89,1))
        save_pixels.append(np.round(gates45dbz,1))
        save_gates.append(np.round(gates45dbz,1))
        #area45 = len(zh45)(240*240)/1000
        if np.any(np.in1d( icois[ic], options['icoi_PHAIL'])): # icois[ic] == options['icoi_PHAIL']:  
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
        if np.any(np.in1d( icois[ic], options['icoi_PHAIL'])):  # icois[ic] == options['icoi_PHAIL']:
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
    if len(RN_inds_parallax) == 4:
        br2 = [x + barWidth for x in br1] 
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        plt.bar(br2, MINPCTS_icois[1,:], color='darkred',   width = barWidth, label=barlabels[1])
        plt.bar(br3, MINPCTS_icois[2,:], color='darkgreen', width = barWidth, label=barlabels[2])	
        plt.bar(br4, MINPCTS_icois[3,:], color='black', width = barWidth, label=barlabels[3])	
    plt.ylabel('MINPCT (K)')  
    plt.xticks([r + barWidth for r in range(len(name))], name)   # adjutst! len() 
    plt.legend()
    plt.title('Min. observed PCTs for COI')
    plt.grid(True)


    # GUARDAR EN UN SOLO POOL DATOS DONDE PHAIL>0.5, Y OTRO GRAN POOL DONDE NO ... 
    PCTarray_PHAIL   = [] #np.zeros((len(options['icoi_PHAIL']), 4));
    PCTarray_NOPHAIL = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL']) , 4));
    ZHarray_PHAIL    = [] #np.zeros((len(options['icoi_PHAIL']), np.ravel(radarZDR).shape[0] ));
    ZHarray_NOPHAIL  = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL']), np.ravel(radarZDR).shape[0]));
    ZDRarray_PHAIL   = [] #np.zeros((len(options['icoi_PHAIL']), np.ravel(radarZDR).shape[0]));
    ZDRarray_NOPHAIL = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL']) , np.ravel(radarZDR).shape[0]));
    AREA_PHAIL       = [] #np.zeros((len(options['icoi_PHAIL'])));
    AREA_NOPHAIL     = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL'])));
    PIXELS_PHAIL     = [] #np.zeros((len(options['icoi_PHAIL'])));
    PIXELS_NOPHAIL   = [] #np.zeros(( len(RN_inds_parallax) - len(options['icoi_PHAIL']) ));
    GATES_PHAIL      = [] #np.zeros((len(options['icoi_PHAIL'])));
    GATES_NOPHAIL    = [] #np.zeros(( len(RN_inds_parallax)- len(options['icoi_PHAIL'])));

    for ic in range(len(RN_inds_parallax)): 
        if np.any(np.in1d( icois[ic], options['icoi_PHAIL'])):  #if icois[ic] == options['icoi_PHAIL'][0]:
            PCTarray_PHAIL.append( [MINPCTS_icois[ic,:]])
            ZHarray_PHAIL.append( np.ravel(radarTH)[RN_inds_parallax[ic]].data )
            ZDRarray_PHAIL.append( np.ravel(radarTH)[RN_inds_parallax[ic]].data )
            AREA_PHAIL.append( save_area[ic] )
            PIXELS_PHAIL.append( save_pixels[ic] )
            GATES_PHAIL.append( save_gates[ic] )

        else:
            PCTarray_NOPHAIL.append( [MINPCTS_icois[ic,:]])
            ZHarray_NOPHAIL.append( np.ravel(radarTH)[RN_inds_parallax[ic]].data )
            ZDRarray_NOPHAIL.append( np.ravel(radarTH)[RN_inds_parallax[ic]].data )
            AREA_NOPHAIL.append(   save_area[ic] )
            PIXELS_NOPHAIL.append( save_pixels[ic] )
            GATES_NOPHAIL.append(  save_gates[ic] )
		
    # NOPHAIL
    PCTarray_NOPHAIL_out = np.zeros([len(PCTarray_NOPHAIL),4]); PCTarray_NOPHAIL_out[:] = np.nan
    for ilist in range(len(PCTarray_NOPHAIL)): 
       PCTarray_NOPHAIL_out[ilist,:] = PCTarray_NOPHAIL[ilist][0]
    # PHAIL
    PCTarray_PHAIL_out = np.zeros([len(PCTarray_PHAIL),4]); PCTarray_PHAIL_out[:] = np.nan
    for ilist in range(len(PCTarray_PHAIL)): 
       PCTarray_PHAIL_out[ilist,:] = PCTarray_PHAIL[ilist][0]  

    del radar 

    return  PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL




#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def run_general_case(options, lat_pfs, lon_pfs, icois):

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
	
    if options['radar_name'] == 'RMA8':
        PHIORIG = radar.fields['PHIDP']['data'].copy() 
        mask = radar.fields['PHIDP']['data'].data.copy()    
        mask[:] = False
        PHIORIG.mask = mask
        radar.add_field_like('PHIDP', 'PHIDP', PHIORIG, replace_existing=True)	
        
    plot_gmi(gmi_dir+options['gfile'], options, radar, lon_pfs, lat_pfs, icois)
    
    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, 
     ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = plot_scatter_4icois_morethan1OFINTEREST(options, radar, icois, gmi_dir+options['gfile'])
    
    gc.collect()

    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, 
	    GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]
  
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

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------      
def main_20180208(): 

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
    lon_pfs  = [-64.80]
    lat_pfs  = [-31.83]
    phail    = [0.534]
    MIN85PCT = [131.1081]
    MIN37PCT = [207.4052]
    MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    MINPCTs  = [270.51, 242.92, 207.42, 131.1081, 198.25, 208.14]
    rfile    = 'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5'  #21UTC
    era5_file = '20180208_21_RMA1.grib'
    reportes_granizo_twitterAPI_geo = [[-31.49, -64.54], [-31.42, -64.50], [-31.42, -64.19]]
    reportes_granizo_twitterAPI_meta = ['SAA (1930UTC)', 'VCP (1942UTC)', 'CDB (24UTC)']
    opts = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5,  'ZDRoffset': 4,
	     'rfile': 'RMA1/'+rfile, 
	    'radar_name':'RMA1',
	    'icoi_PHAIL':[4],
	    'gfile': gfile, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20180208_RMA1/', 
	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail}
    icois_input  = [2,4,5] 

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def main_DOW7_20181214():
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
    lon_pfs  = [-63.11] # [-61.40] [-59.65]
    lat_pfs  = [-31.90] # [-32.30] [-33.90]
    phail    = [0.967] # [0.998] [0.863]
    #MIN85PCT = [71.08] # [45.91] [67.82] 
    #MIN37PCT = [133.99] # [80.12] [151.73] 
    #MINPCTs_labels = ['MIN10PCT', 'MIN19PCT', 'MIN37PCT', 'MIN85PCT', 'MAX85PCT', 'MIN165V']
    #MINPCTs  = [268.73, 224.68, 169.37, 89.09, 199.99, 	194.18] 
    #MINPCTs  = [260.02, 201.87, 133.99, 71.08, 199.84, 212.55]
    #MINPCTs  = [235.52, 130.79, 80.12, 45.91, 199.95, 205.97]
    #MINPCTs  = [274.45, 239.07, 151.73, 67.82, 195.69, 196.58]
    # USE DOW7 for lowest level
    rfile = 'cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc' 
    gfile     = '1B.GPM.GMI.TB2016.20181214-S015009-E032242.027231.V05A.HDF5'
    era5_file = '20181214_03_RMA1.grib'
    reportes_granizo_twitterAPI_geo = [[-32.19, -64.57],[-32.07, -64.54]]
    reportes_granizo_twitterAPI_meta = [['0320UTC','0100']]
    opts = {'xlim_min': -65.3, 'xlim_max': -63.3, 'ylim_min': -32.4, 'ylim_max': -31, 'ZDRoffset': 0,
	    'rfile': 'DOW7/'+rfile, 'gfile': gfile, 
	     'radar_name':'DOW7', 
	     'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181214_RMA1/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [15]}
    icois_input  = [15] 
		
    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	
	
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  
def main_CSPR2_20181111(): 

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
    lon_pfs  = [-64.53]
    lat_pfs  = [-31.83]
    time_pfs = ['1250UTC']
    phail    = [0.653]
    rfile     = 'corcsapr2cmacppiM1.c1.20181111.130003.nc' #'corcsapr2cfrppiM1.a1.20181111.130003.nc'
    gfile     = '1B.GPM.GMI.TB2016.20181111-S113214-E130446.026724.V05A.HDF5'
    era5_file = '20181111_13_RMA1.grib'
    reportes_granizo_twitterAPI_geo = [[]]
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -65.5, 'xlim_max': -63.6, 'ylim_min': -33, 'ylim_max': -31.5, 
    	    'ZDRoffset': 0, 
	    'rfile': 'CSPR2_data/'+rfile, 'gfile': gfile, 
    	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181111am/', 
    	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    	    'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
     	   'icoi_PHAIL': [3], 'radar_name':'CSPR2'}
    icois_input  = [6,5] 
	
    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  	
def RMA1_20190308(): 

    lon_pfs  = [-63.74]
    lat_pfs  = [-30.75]
    time_pfs = ['0204UTC']
    phail    = [0.895]
    rfile    = 'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc'
    gfile     = '1B.GPM.GMI.TB2016.20190308-S004613-E021846.028537.V05A.HDF5'
    era5_file = '20190308_02_RMA1.grib'
    reportes_granizo_twitterAPI_geo = [[]]
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -65.2, 'xlim_max': -62, 'ylim_min': -33, 'ylim_max': -30, 
    	    'ZDRoffset': 0.5, 'rfile': 'RMA1/'+rfile, 'gfile': gfile, 
    	    'window_calc_KDP': 7, 
    	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190308/', 
    	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    	    'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
     	   'icoi_PHAIL': [3], 'radar_name':'RMA1'}
    icois_input  = [3]

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  
def RMA5_20200815(): 
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
    lon_pfs  = [ -54.11 ]
    lat_pfs  = [ -25.28 ]
    time_pfs = ['0215UTC']
    phail    = [ 0.725 ]
    rfile    = 'cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20200815-S015947-E033219.036720.V05A.HDF5'
    era5_file = '20200815_02.grib'
    reportes_granizo_twitterAPI_geo = [[-25.93, -54.57], [-27.03, -55.24]] 
    reportes_granizo_twitterAPI_meta = ['Wanda', 'Jardin de America']
    opts = {'xlim_min': -55.0, 'xlim_max': -52.0, 'ylim_min': -27.5, 'ylim_max': -25.0, 
	    'ZDRoffset': 2, 'rfile': 'RMA5/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20200815_RMA5/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [7], 'radar_name':'RMA5'}
    icois_input  = [7] 
   
    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]		

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  
def RMA3_20190305():
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
    lat_pfs  = [-25.95] 
    lon_pfs  = [-60.57]
    time_pfs = ['1252'] 
    phail    = [0.737]
    rfile     = 'cfrad.20190305_124638.0000_to_20190305_125231.0000_RMA3_0200_01.nc'
    gfile     = '1B.GPM.GMI.TB2016.20190305-S123614-E140847.028498.V05A.HDF5'
    era5_file = '20190305_13.grib'
    #
    # REPORTES TWITTER ... 
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -63, 'xlim_max': -58, 'ylim_min': -27, 'ylim_max': -23, 'ZDRoffset': 3, 
	    'rfile': 'RMA3/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190305_RMA3/', 
	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL':[6], 'radar_name':'RMA3'}
    icois_input  = [6,7] 

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	

#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------- 
def RMA4_20180209():
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    lon_pfs  = [-60.18]
    lat_pfs  = [-27.92]
    time_pfs = ['2005UTC']
    phail    = [0.762]
    rfile    = 'cfrad.20180209_200449.0000_to_20180209_201043.0000_RMA4_0200_01.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20180209-S184820-E202054.022451.V05A.HDF5' 
    era5_file = '20180209_20_RMA4.grib' 
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 
	    'ZDRoffset': -1,   
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20180209_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [17], 'radar_name':'RMA4'}
    icois_input  = [10,17,19] 

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	

#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------- 
def RMA4_20181001(): 
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    lon_pfs  = [-57.16]
    lat_pfs  = [-29.61]
    time_pfs = ['0953UTC']
    phail    = [0.521]
    rfile    = 'cfrad.20181001_095450.0000_to_20181001_100038.0000_RMA4_0200_01.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20181001-S093732-E111006.026085.V05A.HDF5' 
    era5_file = '20181001_09_RMA4.grib' 
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 
	    'ZDRoffset': 1.5,   
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181001_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [37], 'radar_name':'RMA4'}
    icois_input  = [26,36,37] 

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	


#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------- 
def RMA4_20190209(): 

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'		
    lon_pfs  = [-60.28]
    lat_pfs  = [-27.46]
    time_pfs = ['1931UTC']
    phail    = ['0.989']
    rfile    = 'cfrad.20190209_192724.0000_to_20190209_193317.0000_RMA4_0200_01.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20190209-S191744-E205018.028129.V05A.HDF5'
    era5_file = '20190209_20_RMA4.grib' 
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 
	    'ZDRoffset': 1,   
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190209_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [15], 'radar_name':'RMA4'}
    icois_input  = [11,15,16] 

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, time_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	



#----------------------------------------------------------------------------------------------
# ACA TAMBIEN HAY DOS CON PHAIL! 
#---------------------------------------------------------------------------------------------- 
def RMA4_20181218(): 
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
    lon_pfs  = [ -60.31, -59.63]
    lat_pfs  = [-27.98, -28.40]
    time_pfs = ['0115UTC','0115UTC']
    phail    = ['0.964','0.595']
    rfile    = 'cfrad.20181218_014441.0000_to_20181218_015039.0000_RMA4_0200_01.nc' 
    gfile    =  '1B.GPM.GMI.TB2016.20181217-S235720-E012953.027292.V05A.HDF5'
    era5_file = '20181218_01_RMA4.grib' 
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 
	    'ZDRoffset': 3,   
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181218_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [7, 6], 'radar_name':'RMA4'}
    icois_input  = [8, 7, 6]

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	




#----------------------------------------------------------------------------------------------
# ojo que aca los dos son PHAIL ver como agrego! cambio el codigo
#---------------------------------------------------------------------------------------------- 
def RMA4_20181215(): 
 
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    lon_pfs  = [-60.7, -59.01]
    lat_pfs  = [-28.38, -28.38]
    time_pfs = ['0216UTC','0216UTC' ]
    phail    = ['0.930','0.747']
    rfile    = 'cfrad.20181215_021522.0000_to_20181215_022113.0000_RMA4_0200_01.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20181215-S005848-E023122.027246.V05A.HDF5'
    era5_file = '20181215_02_RMA4.grib' 
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 
	    'ZDRoffset': 1.5,   
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181215_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [14,11], 'radar_name':'RMA4'}
    icois_input  = [14, 11] 

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	


#----------------------------------------------------------------------------------------------
# EN ESTE VER QUE HAGO? HAY MUCHOS CONTORNOS FUERA DEL RANGO DEL RADAR ... AGREGO LOS TBS? 
# APARTE MODIFICAR EL CODIGO PORQUE HAY MAS DE UN CONTORNOS CON PHAIL>50% QUE CONTAR! 
#---------------------------------------------------------------------------------------------- 
def RMA4_20181031(): 
	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    # aca solo voy a poner las 3 que me interesan dentro de rango razonable del radar ... 
    #icoi=58//	2018	10	31	01	11	 -26.52	 -57.33	 0.993	267.5031	307.3651	198.3740	111.7183	 56.3188	193.9708	219.0600	1
    #icoi=20//	2018	10	31	01	10	 -28.71	 -58.37	 0.931	276.8601	302.9811	237.4247	151.6656	 67.1172	198.5928	160.6000	1
    #icoi=1//	2018	10	31	01	10	 -28.70	 -60.70	 0.738	274.8905	303.1061	244.4909	174.1353	 99.4944	199.8251	163.2800	1
    time_pfs = ['0110UTC','0110UTC','0110UTC']
    phail    = [0.993, 0.931, 0.738]
    lon_pfs  = [-57.33, -58.37, -60.70]
    lat_pfs  = [-26.52, -28.71, -28.70]
    rfile    = 'cfrad.20181031_010936.0000_to_20181031_011525.0000_RMA4_0200_01.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20181031-S005717-E022950.026546.V05A.HDF5' 
    era5_file = '20181031_01_RMA4.grib' 
    # REPORTES TWITTER ... 
    # CDB capital (varios en base, e.g. https://t.co/Z94Z4z17Ev)
    # VCP (https://twitter.com/icebergdelsur/status/961717942714028032, https://t.co/RJakJjW8sl) gargatuan hail paper!
    # San Antonio de Arredondo (https://t.co/GJwBLvwHVJ ) > 6 cm
    reportes_granizo_twitterAPI_geo = []
    reportes_granizo_twitterAPI_meta = []
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26, 
	    'ZDRoffset': 1.5,   
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 
	    'window_calc_KDP': 7, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181031_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	   'time_pfs':time_pfs[0], 'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [1,58,20], 'radar_name':'RMA4'}
    icois_input  = [1,26,20,58] 

    [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    return [PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]	



#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------- 















#----------------------------------------------------------------------------------------------
# main_20180208 OK 
#----------------------------------------------------------------------------------------------  	
import xarray as xr

[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = main_20180208()

PCTarray_PHAIL_out_mean = []
for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
	PCTarray_PHAIL_out_mean.append(np.nanmean(PCTarray_PHAIL_out[:,ifreq]))

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[ifreq])
		
# And create a netcdf file
RMA1_20180208 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('PCTs'), PCTarray_PHAIL_out_),
                    "PCTarray_NOPHAIL_out": (('icois','PCTs'), PCTarray_NOPHAIL_out),
                    "PCTarray_PHAIL_mean": (('PCTs'),      PCTarray_PHAIL_out_mean),
                    "PCTarray_NOPHAIL_mean": (('PCTs'),    PCTarray_NOPHAIL_out_mean),	
                    "AREA_PHAIL":            (('Nr'),    [np.nanmean(AREA_PHAIL)]),
                    "AREA_NOPHAIL":          (('Nr'),    [np.nanmean(AREA_NOPHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),    [np.nanmean(PIXELS_PHAIL)]), 
                    "PIXELS_NOPHAIL":        (('Nr'),    [np.nanmean(PIXELS_NOPHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),    [np.nanmean(GATES_PHAIL)]), 
                    "GATES_NOPHAIL":         (('Nr'),    [np.nanmean(GATES_NOPHAIL)]),
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_NOPHAIL_1":   (('gates2'), ZHarray_NOPHAIL[0]),
                    "ZDRarray_NOPHAIL_1":  (('gates2'), ZDRarray_NOPHAIL[0]),
                    "ZHarray_NOPHAIL_2":   (('gates3'), ZHarray_NOPHAIL[1]),
                    "ZDRarray_NOPHAIL_2":  (('gates3'), ZDRarray_NOPHAIL[1]),
                    }   )
RMA1_20180208.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA1_20180208.nc', 'w')

del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_, PCTarray_PHAIL_out_mean


#----------------------------------------------------------------------------------------------
# main_DOW7_20181214 OK 
#----------------------------------------------------------------------------------------------  
[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = main_DOW7_20181214()

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])
		
# And create a netcdf file
DOW7_20181214 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('PCTs'), PCTarray_PHAIL_out_),
                    "AREA_PHAIL":            (('Nr'),    [np.nanmean(AREA_PHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),    [np.nanmean(PIXELS_PHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),    [np.nanmean(GATES_PHAIL)]), 
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0]),
                    }   )
DOW7_20181214.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_DOW7_20181214.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_


#----------------------------------------------------------------------------------------------
# main_CSPR2_20181111 OK 
#----------------------------------------------------------------------------------------------  

[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = main_CSPR2_20181111()

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])
if len(PCTarray_NOPHAIL_out) == 1:
	PCTarray_NOPHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_NOPHAIL_out_.append( PCTarray_NOPHAIL_out[0][ifreq])
		
# And create a netcdf file
CSPR2_20181111 = xr.Dataset( {
                    "PCTarray_PHAIL_out":   (('PCTs'),  PCTarray_PHAIL_out_),
                    "PCTarray_NOPHAIL_out": (('PCTs'),  PCTarray_NOPHAIL_out_),
                    "AREA_PHAIL":            (('Nr'),   [np.nanmean(AREA_PHAIL)]),
                    "AREA_NOPHAIL":          (('Nr'),   [np.nanmean(AREA_NOPHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),   [np.nanmean(PIXELS_PHAIL)]), 
                    "PIXELS_NOPHAIL":        (('Nr'),   [np.nanmean(PIXELS_NOPHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),   [np.nanmean(GATES_PHAIL)]), 
                    "GATES_NOPHAIL":         (('Nr'),   [np.nanmean(GATES_NOPHAIL)]),
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_NOPHAIL_1":   (('gates2'), ZHarray_NOPHAIL[0]),
                    "ZDRarray_NOPHAIL_1":  (('gates2'), ZDRarray_NOPHAIL[0])
                    }   )

CSPR2_20181111.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_CSPR2_20181111.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_, PCTarray_NOPHAIL_out_

#----------------------------------------------------------------------------------------------
# RMA1_20190308 OK 
#----------------------------------------------------------------------------------------------  

[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA1_20190308()

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_PHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])
		
# And create a netcdf file
RMA1_20190308 = xr.Dataset( {
                    "PCTarray_PHAIL_out":   (('PCTs'),  PCTarray_PHAIL_out_),
                    "AREA_PHAIL":            (('Nr'),   [np.nanmean(AREA_PHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),   [np.nanmean(PIXELS_PHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),   [np.nanmean(GATES_PHAIL)]), 
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0])
                    }   )

RMA1_20190308.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA1_20190308.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_


#----------------------------------------------------------------------------------------------
# RMA5_20200815 OK
#----------------------------------------------------------------------------------------------  

[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA5_20200815()

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_PHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])
		
# And create a netcdf file
RMA5_20200815 = xr.Dataset( {
                    "PCTarray_PHAIL_out":   (('PCTs'),  PCTarray_PHAIL_out_),
                    "AREA_PHAIL":            (('Nr'),   [np.nanmean(AREA_PHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),   [np.nanmean(PIXELS_PHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),   [np.nanmean(GATES_PHAIL)]), 
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0])
                    }   )

RMA5_20200815.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA5_20200815.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_


#----------------------------------------------------------------------------------------------
# RMA3_20190305  OK
#----------------------------------------------------------------------------------------------  
[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA3_20190305()


if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_PHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])

if len(PCTarray_NOPHAIL_out) == 1:
	PCTarray_NOPHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_NOPHAIL_out_.append( PCTarray_NOPHAIL_out[0][ifreq])
		
# And create a netcdf file
RMA3_20190305 = xr.Dataset( {
                    "PCTarray_PHAIL_out":   (('PCTs'),  PCTarray_PHAIL_out_),
                    "PCTarray_NOPHAIL_out": (('PCTs'),  PCTarray_NOPHAIL_out_),
                    "AREA_PHAIL":            (('Nr'),   [np.nanmean(AREA_PHAIL)]),
                    "AREA_NOPHAIL":          (('Nr'),   [np.nanmean(AREA_NOPHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),   [np.nanmean(PIXELS_PHAIL)]), 
                    "PIXELS_NOPHAIL":        (('Nr'),   [np.nanmean(PIXELS_NOPHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),   [np.nanmean(GATES_PHAIL)]), 
                    "GATES_NOPHAIL":         (('Nr'),   [np.nanmean(GATES_NOPHAIL)]),
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_NOPHAIL_1":   (('gates2'), ZHarray_NOPHAIL[0]),
                    "ZDRarray_NOPHAIL_1":  (('gates2'), ZDRarray_NOPHAIL[0])
                    }   )

RMA3_20190305.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA3_20190305.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_, PCTarray_NOPHAIL_out_


#----------------------------------------------------------------------------------------------
# RMA4_20180209 OK
#----------------------------------------------------------------------------------------------  
[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA4_20180209()

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_PHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])
				
PCTarray_PHAIL_out_mean = []
PCTarray_NOPHAIL_out_mean = []
for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
	PCTarray_PHAIL_out_mean.append(np.nanmean(PCTarray_PHAIL_out[:,ifreq]))
	PCTarray_NOPHAIL_out_mean.append(np.nanmean(PCTarray_NOPHAIL_out[:,ifreq]))

# And create a netcdf file
RMA4_20180209 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('PCTs'), PCTarray_PHAIL_out_),
                    "PCTarray_NOPHAIL_out": (('icois','PCTs'), PCTarray_NOPHAIL_out),
                    "PCTarray_PHAIL_mean": (('PCTs'),      PCTarray_PHAIL_out_mean),
                    "PCTarray_NOPHAIL_mean": (('PCTs'),    PCTarray_NOPHAIL_out_mean),	
                    "AREA_PHAIL":            (('Nr'),    [np.nanmean(AREA_PHAIL)]),
                    "AREA_NOPHAIL":          (('Nr'),    [np.nanmean(AREA_NOPHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),    [np.nanmean(PIXELS_PHAIL)]), 
                    "PIXELS_NOPHAIL":        (('Nr'),    [np.nanmean(PIXELS_NOPHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),    [np.nanmean(GATES_PHAIL)]), 
                    "GATES_NOPHAIL":         (('Nr'),    [np.nanmean(GATES_NOPHAIL)]),
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_NOPHAIL_1":   (('gates2'), ZHarray_NOPHAIL[0]),
                    "ZDRarray_NOPHAIL_1":  (('gates2'), ZDRarray_NOPHAIL[0]),
                    "ZHarray_NOPHAIL_2":   (('gates3'), ZHarray_NOPHAIL[1]),
                    "ZDRarray_NOPHAIL_2":  (('gates3'), ZDRarray_NOPHAIL[1]),
                    }   )
RMA4_20180209.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA4_20180209.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_mean, PCTarray_NOPHAIL_out_mean, PCTarray_PHAIL_out_



#----------------------------------------------------------------------------------------------
# RMA4_20181001 OK
#----------------------------------------------------------------------------------------------  
[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA4_20181001()	

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_PHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])
				
PCTarray_PHAIL_out_mean = []
PCTarray_NOPHAIL_out_mean = []
for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
	PCTarray_PHAIL_out_mean.append(np.nanmean(PCTarray_PHAIL_out[:,ifreq]))
	PCTarray_NOPHAIL_out_mean.append(np.nanmean(PCTarray_NOPHAIL_out[:,ifreq]))

# And create a netcdf file
RMA4_20181001 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('PCTs'), PCTarray_PHAIL_out_),
                    "PCTarray_NOPHAIL_out": (('icois','PCTs'), PCTarray_NOPHAIL_out),
                    "PCTarray_PHAIL_mean": (('PCTs'),      PCTarray_PHAIL_out_mean),
                    "PCTarray_NOPHAIL_mean": (('PCTs'),    PCTarray_NOPHAIL_out_mean),	
                    "AREA_PHAIL":            (('Nr'),    [np.nanmean(AREA_PHAIL)]),
                    "AREA_NOPHAIL":          (('Nr'),    [np.nanmean(AREA_NOPHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),    [np.nanmean(PIXELS_PHAIL)]), 
                    "PIXELS_NOPHAIL":        (('Nr'),    [np.nanmean(PIXELS_NOPHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),    [np.nanmean(GATES_PHAIL)]), 
                    "GATES_NOPHAIL":         (('Nr'),    [np.nanmean(GATES_NOPHAIL)]),
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_NOPHAIL_1":   (('gates2'), ZHarray_NOPHAIL[0]),
                    "ZDRarray_NOPHAIL_1":  (('gates2'), ZDRarray_NOPHAIL[0]),
                    "ZHarray_NOPHAIL_2":   (('gates3'), ZHarray_NOPHAIL[1]),
                    "ZDRarray_NOPHAIL_2":  (('gates3'), ZDRarray_NOPHAIL[1]),
                    }   )
RMA4_20181001.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA4_20181001.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_mean, PCTarray_NOPHAIL_out_mean, PCTarray_PHAIL_out_

#----------------------------------------------------------------------------------------------
# RMA4_20190209 OK
#----------------------------------------------------------------------------------------------
[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA4_20190209()

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_PHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])
				
PCTarray_PHAIL_out_mean = []
PCTarray_NOPHAIL_out_mean = []
for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
	PCTarray_PHAIL_out_mean.append(np.nanmean(PCTarray_PHAIL_out[:,ifreq]))
	PCTarray_NOPHAIL_out_mean.append(np.nanmean(PCTarray_NOPHAIL_out[:,ifreq]))

# And create a netcdf file
RMA4_20190209 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('PCTs'), PCTarray_PHAIL_out_),
                    "PCTarray_NOPHAIL_out": (('icois','PCTs'), PCTarray_NOPHAIL_out),
                    "PCTarray_PHAIL_mean": (('PCTs'),      PCTarray_PHAIL_out_mean),
                    "PCTarray_NOPHAIL_mean": (('PCTs'),    PCTarray_NOPHAIL_out_mean),	
                    "AREA_PHAIL":            (('Nr'),    [np.nanmean(AREA_PHAIL)]),
                    "AREA_NOPHAIL":          (('Nr'),    [np.nanmean(AREA_NOPHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),    [np.nanmean(PIXELS_PHAIL)]), 
                    "PIXELS_NOPHAIL":        (('Nr'),    [np.nanmean(PIXELS_NOPHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),    [np.nanmean(GATES_PHAIL)]), 
                    "GATES_NOPHAIL":         (('Nr'),    [np.nanmean(GATES_NOPHAIL)]),
                    "ZHarray_PHAIL":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_NOPHAIL_1":   (('gates2'), ZHarray_NOPHAIL[0]),
                    "ZDRarray_NOPHAIL_1":  (('gates2'), ZDRarray_NOPHAIL[0]),
                    "ZHarray_NOPHAIL_2":   (('gates3'), ZHarray_NOPHAIL[1]),
                    "ZDRarray_NOPHAIL_2":  (('gates3'), ZDRarray_NOPHAIL[1]),
                    }   )
RMA4_20190209.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA4_20190209.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_mean, PCTarray_NOPHAIL_out_mean, PCTarray_PHAIL_out_

#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------- 
[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA4_20181031()

if len(PCTarray_NOPHAIL_out) == 1:
	PCTarray_NOPHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_NOPHAIL_out_.append( PCTarray_NOPHAIL_out[0][ifreq])
				
PCTarray_PHAIL_out_mean = []
PCTarray_NOPHAIL_out_mean = []
for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
	PCTarray_PHAIL_out_mean.append(np.nanmean(PCTarray_PHAIL_out[:,ifreq]))
	PCTarray_NOPHAIL_out_mean.append(np.nanmean(PCTarray_NOPHAIL_out[:,ifreq]))

# And create a netcdf file
RMA4_20181031 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('icois','PCTs'), PCTarray_PHAIL_out),
                    "PCTarray_NOPHAIL_out": (('PCTs'), PCTarray_NOPHAIL_out_),
                    "PCTarray_PHAIL_mean": (('PCTs'),      PCTarray_PHAIL_out_mean),
                    "PCTarray_NOPHAIL_mean": (('PCTs'),    PCTarray_NOPHAIL_out_mean),	
                    "meanAREA_PHAIL":            (('Nr'),    [np.nanmean(AREA_PHAIL)]),
                    "meanAREA_NOPHAIL":          (('Nr'),    [np.nanmean(AREA_NOPHAIL)]),
                    "meanPIXELS_PHAIL":          (('Nr'),    [np.nanmean(PIXELS_PHAIL)]), 
                    "meanPIXELS_NOPHAIL":        (('Nr'),    [np.nanmean(PIXELS_NOPHAIL)]), 
                    "meanGATES_PHAIL":           (('Nr'),    [np.nanmean(GATES_PHAIL)]), 
                    "meanGATES_NOPHAIL":         (('Nr'),    [np.nanmean(GATES_NOPHAIL)]),
                    "AREA_PHAIL":            (('icois'),    AREA_PHAIL),
                    "AREA_NOPHAIL":          (('noicois'),    AREA_NOPHAIL),
                    "PIXELS_PHAIL":          (('icois'),    PIXELS_PHAIL), 
                    "PIXELS_NOPHAIL":        (('noicois'),    PIXELS_NOPHAIL), 
                    "GATES_PHAIL":           (('icois'),    GATES_PHAIL), 
                    "GATES_NOPHAIL":         (('noicois'),    GATES_NOPHAIL),
                    "ZHarray_PHAIL_1":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL_1":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_PHAIL_2":   (('gates2'), ZHarray_PHAIL[1]),
                    "ZDRarray_PHAIL_2":  (('gates2'), ZDRarray_PHAIL[1]),
                    "ZHarray_PHAIL_3":   (('gates3'), ZHarray_PHAIL[2]),
                    "ZDRarray_PHAIL_3":  (('gates3'), ZDRarray_PHAIL[2]),
                    "ZHarray_NOPHAIL":   (('gates4'), ZHarray_NOPHAIL[0]),
                    "ZDRarray_NOPHAIL":  (('gates4'), ZDRarray_NOPHAIL[0])  }   )



RMA4_20181031.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA4_20181031.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_mean, PCTarray_NOPHAIL_out_, PCTarray_NOPHAIL_out_mean

#----------------------------------------------------------------------------------------------
# RMA4_20181215 (plot_scatter_4icois_morethan1OFINTEREST) ok 
#---------------------------------------------------------------------------------------------- 
[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA4_20181215()
				
PCTarray_PHAIL_out_mean = []
for ifreq in range(PCTarray_PHAIL_out.shape[1]):
	PCTarray_PHAIL_out_mean.append(np.nanmean(PCTarray_PHAIL_out[:,ifreq]))	
	
if len(PCTarray_NOPHAIL_out) == 1:
	PCTarray_NOPHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_NOPHAIL_out_.append( PCTarray_NOPHAIL_out[0][ifreq])
		
# And create a netcdf file
RMA4_20181215 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('icois','PCTs'), PCTarray_PHAIL_out),
                    "PCTarray_PHAIL_mean": (('PCTs'),      PCTarray_PHAIL_out_mean),
                    "meanAREA_PHAIL":            (('Nr'),    [np.nanmean(AREA_PHAIL)]),
                    "meanPIXELS_PHAIL":          (('Nr'),    [np.nanmean(PIXELS_PHAIL)]), 
                    "meanGATES_PHAIL":           (('Nr'),    [np.nanmean(GATES_PHAIL)]), 
                    "AREA_PHAIL":            (('icois'),    AREA_PHAIL),
                    "PIXELS_PHAIL":          (('icois'),    PIXELS_PHAIL), 
                    "GATES_PHAIL":           (('icois'),    GATES_PHAIL), 
                    "ZHarray_PHAIL_1":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL_1":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_PHAIL_2":   (('gates2'), ZHarray_PHAIL[1]),
                    "ZDRarray_PHAIL_2":  (('gates2'), ZDRarray_PHAIL[1])
                    }   )

RMA4_20181215.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA4_20181215.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_PHAIL_out_mean, PCTarray_NOPHAIL_out_

#----------------------------------------------------------------------------------------------
# RMA4_20181218 (plot_scatter_4icois_morethan1OFINTEREST) 
#----------------------------------------------------------------------------------------------
[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = RMA4_20181218()
				
PCTarray_PHAIL_out_mean = []
for ifreq in range(PCTarray_PHAIL_out.shape[1]):
	PCTarray_PHAIL_out_mean.append(np.nanmean(PCTarray_PHAIL_out[:,ifreq]))	

if len(PCTarray_NOPHAIL_out) == 1:
	PCTarray_NOPHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_NOPHAIL_out_.append( PCTarray_NOPHAIL_out[0][ifreq])
		
# And create a netcdf file
RMA4_20181218 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('icois','PCTs'), PCTarray_PHAIL_out),
                    "PCTarray_NOPHAIL_out": (('PCTs'), PCTarray_NOPHAIL_out_),
                    "PCTarray_PHAIL_mean": (('PCTs'),      PCTarray_PHAIL_out_mean),
                    "AREA_PHAIL":            (('Nr'),    [np.nanmean(AREA_PHAIL)]),
                    "AREA_NOPHAIL":          (('Nr'),    [np.nanmean(AREA_NOPHAIL)]),
                    "PIXELS_PHAIL":          (('Nr'),    [np.nanmean(PIXELS_PHAIL)]), 
                    "PIXELS_NOPHAIL":        (('Nr'),    [np.nanmean(PIXELS_NOPHAIL)]), 
                    "GATES_PHAIL":           (('Nr'),    [np.nanmean(GATES_PHAIL)]), 
                    "GATES_NOPHAIL":         (('Nr'),    [np.nanmean(GATES_NOPHAIL)]),
                    "AREA_PHAIL":            (('icois'),    AREA_PHAIL),
                    "AREA_NOPHAIL":          (('noicois'),    AREA_NOPHAIL),
                    "PIXELS_PHAIL":          (('icois'),    PIXELS_PHAIL), 
                    "PIXELS_NOPHAIL":        (('noicois'),    PIXELS_NOPHAIL), 
                    "GATES_PHAIL":           (('icois'),    GATES_PHAIL), 
                    "GATES_NOPHAIL":         (('noicois'),    GATES_NOPHAIL),	
                    "ZHarray_PHAIL_1":       (('gates1'), ZHarray_PHAIL[0]),
                    "ZDRarray_PHAIL_1":      (('gates1'), ZDRarray_PHAIL[0]),
                    "ZHarray_PHAIL_2":   (('gates2'), ZHarray_PHAIL[1]),
                    "ZDRarray_PHAIL_2":  (('gates2'), ZDRarray_PHAIL[1]),
                    "ZHarray_NOPHAIL":   (('gates4'), ZHarray_NOPHAIL[0]),
                    "ZDRarray_NOPHAIL":  (('gates4'), ZDRarray_NOPHAIL[0]),	
                    }   )

RMA4_20181218.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/full_RMA4_20181218.nc', 'w')
del PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL
del PCTarray_NOPHAIL_out_, PCTarray_PHAIL_out_mean



#----------------------------------------------------------------------------------------------
# scatter plots 37/19 AND 37/85 w/ further info
#----------------------------------------------------------------------------------------------

def make_scatterplots_sector3_with3Dvalue(var4title, varTitle, novarTitle, vminn, vmaxx ): 
	
	fig = plt.figure(figsize=(10,10)) 
	gs1 = gridspec.GridSpec(2, 2)
	ax1 = plt.subplot(gs1[0,0])
	plt.scatter(RMA3_20190305['PCTarray_PHAIL_out'].data[1],  RMA3_20190305['PCTarray_PHAIL_out'].data[2], 
		    c=RMA3_20190305[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_PHAIL_out'].data[1],  RMA4_20180209['PCTarray_PHAIL_out'].data[2], 
		    c=RMA4_20180209[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_PHAIL_out'].data[1], RMA4_20181001['PCTarray_PHAIL_out'].data[2],  
		    c=RMA4_20181001[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_PHAIL_out'].data[1],  RMA4_20190209['PCTarray_PHAIL_out'].data[2], 
		    c=RMA4_20190209[varTitle].data[0], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())

	plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181031['PCTarray_PHAIL_out'].data[0,2], 
		    c=RMA4_20181031[varTitle].data[0], marker='o',  s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181031['PCTarray_PHAIL_out'].data[1,2], 
		    c=RMA4_20181031[varTitle].data[1], marker='o',  s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[2,1], RMA4_20181031['PCTarray_PHAIL_out'].data[2,2], 
		    c=RMA4_20181031[varTitle].data[2], marker='o', s=30, vmin=vminn, vmax=vmaxx,norm=matplotlib.colors.LogNorm())

	plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181215['PCTarray_PHAIL_out'].data[0,2], 
		    c=RMA4_20181215[varTitle].data[0], marker='o', s=30, vmin=vminn, vmax=vmaxx,norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181215['PCTarray_PHAIL_out'].data[1,2], 
		    c=RMA4_20181215[varTitle].data[1], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())

	plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181218['PCTarray_PHAIL_out'].data[0,2], 
		    c=RMA4_20181218[varTitle].data[0], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	pcm = plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181218['PCTarray_PHAIL_out'].data[1,2], 
			  c=RMA4_20181218[varTitle].data[1], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	
	plt.scatter(RMA3_20190305['PCTarray_NOPHAIL_out'].data[1],  RMA3_20190305['PCTarray_NOPHAIL_out'].data[2],c=RMA3_20190305[novarTitle].data[0], 
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA3_20190305['PCTarray_NOPHAIL_out'].data[1],  RMA3_20190305['PCTarray_NOPHAIL_out'].data[2], c=RMA3_20190305[novarTitle].data[1],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,2], c=RMA4_20180209[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,2], c=RMA4_20180209[novarTitle].data[1],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,2], c=RMA4_20181001[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,2], c=RMA4_20181001[novarTitle].data[1],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,2], c=RMA4_20190209[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,2], c=RMA4_20190209[novarTitle].data[1],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181031['PCTarray_NOPHAIL_out'].data[1],  RMA4_20181031['PCTarray_NOPHAIL_out'].data[2], c=RMA4_20181031[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181218['PCTarray_NOPHAIL_out'].data[1],  RMA4_20181218['PCTarray_NOPHAIL_out'].data[2], c=RMA4_20181218[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())

	plt.colorbar(pcm)

	plt.legend(fontsize=10)
	plt.grid(True)
	plt.xlabel('MINPCT(19)')
	plt.ylabel('MINPCTT(37)')
	plt.xlim([170,300])
	plt.ylim([80,240])

	ax1 = plt.subplot(gs1[0,1])
	plt.scatter(RMA3_20190305['PCTarray_PHAIL_out'].data[2],  RMA3_20190305['PCTarray_PHAIL_out'].data[3], 
		    c=RMA3_20190305[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_PHAIL_out'].data[2],  RMA4_20180209['PCTarray_PHAIL_out'].data[3], 
		    c=RMA4_20180209[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_PHAIL_out'].data[2], RMA4_20181001['PCTarray_PHAIL_out'].data[3],  
		    c=RMA4_20181001[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_PHAIL_out'].data[2],  RMA4_20190209['PCTarray_PHAIL_out'].data[3],
		    c=RMA4_20190209[varTitle].data[0], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())

	plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[0,3], 
		    c=RMA4_20181031[varTitle].data[0], marker='o',  s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[1,3], 
		    c=RMA4_20181031[varTitle].data[1], marker='o',  s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[2,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[2,3], 
		    c=RMA4_20181031[varTitle].data[2], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())

	plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181215['PCTarray_PHAIL_out'].data[0,3], 
		    c=RMA4_20181215[varTitle].data[0], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181215['PCTarray_PHAIL_out'].data[1,3], 
		    c=RMA4_20181215[varTitle].data[1], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())

	plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181218['PCTarray_PHAIL_out'].data[0,3], 
		    c=RMA4_20181218[varTitle].data[0], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	pcm = plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181218['PCTarray_PHAIL_out'].data[1,3], 
			  c=RMA4_20181218[varTitle].data[1], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	
	plt.colorbar(pcm)


	plt.scatter(RMA3_20190305['PCTarray_NOPHAIL_out'].data[2],  RMA3_20190305['PCTarray_NOPHAIL_out'].data[3],c=RMA3_20190305[novarTitle].data[0], 
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA3_20190305['PCTarray_NOPHAIL_out'].data[2],  RMA3_20190305['PCTarray_NOPHAIL_out'].data[3], c=RMA3_20190305[novarTitle].data[1],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,3], c=RMA4_20180209[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,3], c=RMA4_20180209[novarTitle].data[1],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,3], c=RMA4_20181001[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,3], c=RMA4_20181001[novarTitle].data[1],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,3], c=RMA4_20190209[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,3], c=RMA4_20190209[novarTitle].data[1],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181031['PCTarray_NOPHAIL_out'].data[2],  RMA4_20181031['PCTarray_NOPHAIL_out'].data[3], c=RMA4_20181031[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181218['PCTarray_NOPHAIL_out'].data[2],  RMA4_20181218['PCTarray_NOPHAIL_out'].data[3], c=RMA4_20181218[novarTitle].data[0],
		    s=40, marker='s', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())


	plt.grid(True)
	plt.xlabel('MINPCT(37)')
	plt.ylabel('MINPCTT(85)')
	plt.xlim([80,230])
	plt.ylim([50,170])
	plt.suptitle('RMA3+RMA4 '+ var4title,y=0.9)

	return



make_scatterplots_sector3_with3Dvalue('AREA_PHAIL', 'AREA_PHAIL', 0, 1e4 ) 




#----------------------------------------------------------------------------------------------
# General Stats: 
# 12 cases 
#----------------------------------------------------------------------------------------------
#----- > colores para CORDOBA, MISIONES, RMA3+RMA4 ... 
colors_plot = ['darkred', 'darkgreen', 'darkblue']
	
fig = plt.figure(figsize=(20,7)) 
gs1 = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs1[0,0])
plt.scatter(RMA1_20180208['PCTarray_PHAIL_out'].data[1],  RMA1_20180208['PCTarray_PHAIL_out'].data[2], s=20, marker='*', color=colors_plot[0], label='RMA1_20180208') 
plt.scatter(DOW7_20181214['PCTarray_PHAIL_out'].data[1],  DOW7_20181214['PCTarray_PHAIL_out'].data[2], s=20, marker='>', color=colors_plot[0], label='DOW7_20181214') 
plt.scatter(CSPR2_20181111['PCTarray_PHAIL_out'].data[1], CSPR2_20181111['PCTarray_PHAIL_out'].data[2],s=20, marker='<', color=colors_plot[0], label='CSPR2_20181111') 
plt.scatter(RMA1_20190308['PCTarray_PHAIL_out'].data[1],  RMA1_20190308['PCTarray_PHAIL_out'].data[2], s=20, marker='s', color=colors_plot[0], label='RMA1_20190308') 

plt.scatter(RMA5_20200815['PCTarray_PHAIL_out'].data[1],  RMA5_20200815['PCTarray_PHAIL_out'].data[2], s=20, marker='*', color=colors_plot[1], label='RMA5_20200815') 

plt.scatter(RMA3_20190305['PCTarray_PHAIL_out'].data[1],  RMA3_20190305['PCTarray_PHAIL_out'].data[2], s=20, marker='*', color='blue', label='RMA5_20200815') 

plt.scatter(RMA4_20180209['PCTarray_PHAIL_out'].data[1],  RMA4_20180209['PCTarray_PHAIL_out'].data[2], s=20, marker='>', color=colors_plot[2], label='RMA4_20180209') 
plt.scatter(RMA4_20181001['PCTarray_PHAIL_out'].data[1],  RMA4_20181001['PCTarray_PHAIL_out'].data[2], s=20, marker='<', color=colors_plot[2], label='RMA4_20181001') 
plt.scatter(RMA4_20190209['PCTarray_PHAIL_out'].data[1],  RMA4_20190209['PCTarray_PHAIL_out'].data[2], s=20, marker='s', color=colors_plot[2], label='RMA4_20190209') 

plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181031['PCTarray_PHAIL_out'].data[0,2], s=20, marker='*', color=colors_plot[2], label='RMA4_20181031') 
plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181031['PCTarray_PHAIL_out'].data[1,2], s=60, marker='*', color=colors_plot[2], label='RMA4_20181031') 
plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[2,1],  RMA4_20181031['PCTarray_PHAIL_out'].data[2,2], s=100, marker='*', color=colors_plot[2], label='RMA4_20181031') 

plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181215['PCTarray_PHAIL_out'].data[0,2], s=20, marker='x', color=colors_plot[2], label='RMA4_20181215') 
plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181215['PCTarray_PHAIL_out'].data[1,2], s=80, marker='x', color=colors_plot[2], label='RMA4_20181215') 

plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181218['PCTarray_PHAIL_out'].data[0,2], s=20, marker='X', color=colors_plot[2], label='RMA4_20181218') 
plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181218['PCTarray_PHAIL_out'].data[1,2], s=80, marker='X', color=colors_plot[2], label='RMA4_20181218') 

plt.plot(np.nan, np.nan, '-', color='darkred', label='Cordoba region')
plt.plot(np.nan, np.nan, '-', color='darkgreen', label='Misiones')
plt.plot(np.nan, np.nan, '-', color='darkblue',  label='RMA4+RMA3')

plt.legend()
plt.grid(True)
plt.xlabel('MINPCT(19)')
plt.ylabel('MINPCTT(37)')
plt.title('Contours from case studies w/ Phail>0.5')

ax1 = plt.subplot(gs1[0,1])
plt.scatter(RMA1_20180208['PCTarray_PHAIL_out'].data[2],  RMA1_20180208['PCTarray_PHAIL_out'].data[3], s=20, marker='*', color=colors_plot[0], label='RMA1_20180208') 
plt.scatter(DOW7_20181214['PCTarray_PHAIL_out'].data[2],  DOW7_20181214['PCTarray_PHAIL_out'].data[3], s=20, marker='>', color=colors_plot[0], label='DOW7_20181214') 
plt.scatter(CSPR2_20181111['PCTarray_PHAIL_out'].data[2], CSPR2_20181111['PCTarray_PHAIL_out'].data[3],s=20, marker='<', color=colors_plot[0], label='CSPR2_20181111') 
plt.scatter(RMA1_20190308['PCTarray_PHAIL_out'].data[2],  RMA1_20190308['PCTarray_PHAIL_out'].data[3], s=20, marker='s', color=colors_plot[0], label='RMA1_20190308') 

plt.scatter(RMA5_20200815['PCTarray_PHAIL_out'].data[2],  RMA5_20200815['PCTarray_PHAIL_out'].data[3], s=20, marker='*', color=colors_plot[1], label='RMA5_20200815') 

plt.scatter(RMA3_20190305['PCTarray_PHAIL_out'].data[2],  RMA3_20190305['PCTarray_PHAIL_out'].data[3], s=20, marker='*', color='blue', label='RMA5_20200815') 

plt.scatter(RMA4_20180209['PCTarray_PHAIL_out'].data[2],  RMA4_20180209['PCTarray_PHAIL_out'].data[3], s=20, marker='>', color=colors_plot[2], label='RMA4_20180209') 
plt.scatter(RMA4_20181001['PCTarray_PHAIL_out'].data[2],  RMA4_20181001['PCTarray_PHAIL_out'].data[3], s=20, marker='<', color=colors_plot[2], label='RMA4_20181001') 
plt.scatter(RMA4_20190209['PCTarray_PHAIL_out'].data[2],  RMA4_20190209['PCTarray_PHAIL_out'].data[3], s=20, marker='s', color=colors_plot[2], label='RMA4_20190209') 

plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[0,3], s=20, marker='*', color=colors_plot[2], label='RMA4_20181031') 
plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[1,3], s=60, marker='*', color=colors_plot[2], label='RMA4_20181031') 
plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[2,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[2,3], s=100, marker='*', color=colors_plot[2], label='RMA4_20181031') 

plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181215['PCTarray_PHAIL_out'].data[0,3], s=20, marker='x', color=colors_plot[2], label='RMA4_20181215') 
plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181215['PCTarray_PHAIL_out'].data[1,3], s=80, marker='x', color=colors_plot[2], label='RMA4_20181215') 

plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181218['PCTarray_PHAIL_out'].data[0,3], s=20, marker='X', color=colors_plot[2], label='RMA4_20181218') 
plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181218['PCTarray_PHAIL_out'].data[1,3], s=80, marker='X', color=colors_plot[2], label='RMA4_20181218') 

plt.grid(True)
plt.xlabel('MINPCT(37)')
plt.ylabel('MINPCTT(85)')
plt.title('Contours from case studies w/ Phail>0.5')

#----------------------------------------------------------------------------------------------
# Same as above but colormap w/ Phail ... RMA1
#----------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(10,10)) 
gs1 = gridspec.GridSpec(2, 2)

ax1 = plt.subplot(gs1[0,0])
plt.scatter(RMA1_20180208['PCTarray_PHAIL_out'].data[1],  RMA1_20180208['PCTarray_PHAIL_out'].data[2], c=0.534, s=30, marker='o', vmin=0.5, vmax=1.0)
plt.scatter(DOW7_20181214['PCTarray_PHAIL_out'].data[1],  DOW7_20181214['PCTarray_PHAIL_out'].data[2], c=0.967, s=30, marker='s', vmin=0.5, vmax=1.0)
plt.scatter(CSPR2_20181111['PCTarray_PHAIL_out'].data[1], CSPR2_20181111['PCTarray_PHAIL_out'].data[2], c=0.653, s=30, marker='>', vmin=0.5, vmax=1.0)
pcm = plt.scatter(RMA1_20190308['PCTarray_PHAIL_out'].data[1],  RMA1_20190308['PCTarray_PHAIL_out'].data[2], c=0.895, marker='<', s=30, vmin=0.5, vmax=1.0)
plt.colorbar(pcm)

plt.scatter(np.nan, np.nan, marker='o', color='w', edgecolor='k', label='20180208')
plt.scatter(np.nan, np.nan, marker='s',color='w', edgecolor='k', label='20181214')
plt.scatter(np.nan, np.nan, marker='>',color='w', edgecolor='k',  label='20181111')
plt.scatter(np.nan, np.nan, marker='<',color='w', edgecolor='k',  label='20190308')

# como se compara con los contornos que no tienen Phail? 
plt.scatter(RMA1_20180208['PCTarray_NOPHAIL_out'].data[0,1],  RMA1_20180208['PCTarray_NOPHAIL_out'].data[0,2], s=30, marker='x', color='k')
plt.scatter(RMA1_20180208['PCTarray_NOPHAIL_out'].data[1,1],  RMA1_20180208['PCTarray_NOPHAIL_out'].data[1,2], s=30, marker='x', color='k')

plt.scatter(np.nan, np.nan, marker='x', color='k', label='20180208')

plt.legend(fontsize=10)
plt.grid(True)
plt.xlabel('MINPCT(19)')
plt.ylabel('MINPCTT(37)')
plt.xlim([170,300])
plt.ylim([80,240])

ax1 = plt.subplot(gs1[0,1])
plt.scatter(RMA1_20180208['PCTarray_PHAIL_out'].data[2],  RMA1_20180208['PCTarray_PHAIL_out'].data[3], c=0.534, s=30, marker='o', vmin=0.5, vmax=1.0)
plt.scatter(DOW7_20181214['PCTarray_PHAIL_out'].data[2],  DOW7_20181214['PCTarray_PHAIL_out'].data[3], c=0.967, s=30, marker='s', vmin=0.5, vmax=1.0)
plt.scatter(CSPR2_20181111['PCTarray_PHAIL_out'].data[2], CSPR2_20181111['PCTarray_PHAIL_out'].data[3], c=0.653, s=30, marker='>', vmin=0.5, vmax=1.0)
pcm = plt.scatter(RMA1_20190308['PCTarray_PHAIL_out'].data[2],  RMA1_20190308['PCTarray_PHAIL_out'].data[3], c=0.895, marker='<', s=30, vmin=0.5, vmax=1.0)
plt.colorbar(pcm)
# como se compara con los contornos que no tienen Phail? 
plt.scatter(RMA1_20180208['PCTarray_NOPHAIL_out'].data[0,2],  RMA1_20180208['PCTarray_NOPHAIL_out'].data[0,3], s=30, marker='x', color='k')
plt.scatter(RMA1_20180208['PCTarray_NOPHAIL_out'].data[1,2],  RMA1_20180208['PCTarray_NOPHAIL_out'].data[1,3], s=30, marker='x', color='k')

plt.grid(True)
plt.xlabel('MINPCT(37)')
plt.ylabel('MINPCTT(85)')
plt.xlim([80,230])
plt.ylim([50,170])

plt.suptitle('CORDOBA',y=0.9)

#----------------------------------------------------------------------------------------------
# Same as above but colormap w/ Phail ... RMA5
#----------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(10,10)) 
gs1 = gridspec.GridSpec(2, 2)

ax1 = plt.subplot(gs1[0,0])
pcm = plt.scatter(RMA5_20200815['PCTarray_PHAIL_out'].data[1],  RMA5_20200815['PCTarray_PHAIL_out'].data[2], c=0.727, marker='o', s=30, vmin=0.5, vmax=1.0)
plt.colorbar(pcm)
plt.scatter(np.nan, np.nan, marker='o', color='w', edgecolor='k', label='20200815')

plt.legend(fontsize=10)
plt.grid(True)
plt.xlabel('MINPCT(19)')
plt.ylabel('MINPCTT(37)')
plt.xlim([170,300])
plt.ylim([80,240])

ax1 = plt.subplot(gs1[0,1])
pcm = plt.scatter(RMA5_20200815['PCTarray_PHAIL_out'].data[2],  RMA5_20200815['PCTarray_PHAIL_out'].data[3], c=0.727, marker='o', s=30, vmin=0.5, vmax=1.0)
plt.colorbar(pcm)

plt.grid(True)
plt.xlabel('MINPCT(37)')
plt.ylabel('MINPCTT(85)')
plt.xlim([80,230])
plt.ylim([50,170])
plt.suptitle('RMA5',y=0.9)

#----------------------------------------------------------------------------------------------
# Same as above but colormap w/ Phail ... RMA3+RMA4
#----------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(10,10)) 
gs1 = gridspec.GridSpec(2, 2)

ax1 = plt.subplot(gs1[0,0])
plt.scatter(RMA3_20190305['PCTarray_PHAIL_out'].data[1],  RMA3_20190305['PCTarray_PHAIL_out'].data[2], c=0.737, s=30, marker='o', vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20180209['PCTarray_PHAIL_out'].data[1],  RMA4_20180209['PCTarray_PHAIL_out'].data[2], c=0.762, s=30, marker='s', vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20181001['PCTarray_PHAIL_out'].data[1], RMA4_20181001['PCTarray_PHAIL_out'].data[2],  c=0.965, s=30, marker='>', vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20190209['PCTarray_PHAIL_out'].data[1],  RMA4_20190209['PCTarray_PHAIL_out'].data[2], c=0.989, marker='<', s=30, vmin=0.5, vmax=1.0)

plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181031['PCTarray_PHAIL_out'].data[0,2], c=0.738, marker='p',  s=20, vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181031['PCTarray_PHAIL_out'].data[1,2], c=0.931, marker='p',  s=40, vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[2,1],  RMA4_20181031['PCTarray_PHAIL_out'].data[2,2], c=0.993, marker='p', s=50, vmin=0.5, vmax=1.0)

plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181215['PCTarray_PHAIL_out'].data[0,2], c=0.930, marker='v', s=20, vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181215['PCTarray_PHAIL_out'].data[1,2], c=0.747, marker='v', s=20, vmin=0.5, vmax=1.0)

plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[0,1],  RMA4_20181218['PCTarray_PHAIL_out'].data[0,2], c=0.599, marker='^', s=20, vmin=0.5, vmax=1.0)
pcm = plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[1,1],  RMA4_20181218['PCTarray_PHAIL_out'].data[1,2], c=0.964, marker='^', s=20, vmin=0.5, vmax=1.0)

plt.scatter(RMA3_20190305['PCTarray_NOPHAIL_out'].data[1],  RMA3_20190305['PCTarray_NOPHAIL_out'].data[2], s=30, marker='x', color='k')
plt.scatter(RMA3_20190305['PCTarray_NOPHAIL_out'].data[1],  RMA3_20190305['PCTarray_NOPHAIL_out'].data[2], s=30, marker='x', color='k')
plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,2], s=30, marker='x', color='k')
plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,2], s=30, marker='x', color='k')
plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,2], s=30, marker='x', color='k')
plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,2], s=30, marker='x', color='k')
plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,2], s=30, marker='x', color='k')
plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,2], s=30, marker='x', color='k')
plt.scatter(RMA4_20181031['PCTarray_NOPHAIL_out'].data[1],  RMA4_20181031['PCTarray_NOPHAIL_out'].data[2], s=30, marker='x', color='k')
plt.scatter(RMA4_20181218['PCTarray_NOPHAIL_out'].data[1],  RMA4_20181218['PCTarray_NOPHAIL_out'].data[2], s=30, marker='x', color='k')

plt.colorbar(pcm)

plt.scatter(np.nan, np.nan, marker='o', color='w', edgecolor='k', label='20190305')
plt.scatter(np.nan, np.nan, marker='s',color='w', edgecolor='k', label='20180209')
plt.scatter(np.nan, np.nan, marker='>',color='w', edgecolor='k',  label='20181001')
plt.scatter(np.nan, np.nan, marker='<',color='w', edgecolor='k',  label='20190209')

plt.scatter(np.nan, np.nan, marker='p',color='w', edgecolor='k',  label='20181031')
plt.scatter(np.nan, np.nan, marker='v',color='w', edgecolor='k',  label='20181215')
plt.scatter(np.nan, np.nan, marker='^',color='w', edgecolor='k',  label='20181218')

plt.legend(fontsize=10)
plt.grid(True)
plt.xlabel('MINPCT(19)')
plt.ylabel('MINPCTT(37)')
plt.xlim([170,300])
plt.ylim([80,240])

ax1 = plt.subplot(gs1[0,1])
plt.scatter(RMA3_20190305['PCTarray_PHAIL_out'].data[2],  RMA3_20190305['PCTarray_PHAIL_out'].data[3], c=0.737, s=30, marker='o', vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20180209['PCTarray_PHAIL_out'].data[2],  RMA4_20180209['PCTarray_PHAIL_out'].data[3], c=0.762, s=30, marker='s', vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20181001['PCTarray_PHAIL_out'].data[2], RMA4_20181001['PCTarray_PHAIL_out'].data[3],  c=0.965, s=30, marker='>', vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20190209['PCTarray_PHAIL_out'].data[2],  RMA4_20190209['PCTarray_PHAIL_out'].data[3], c=0.989, marker='<', s=30, vmin=0.5, vmax=1.0)

plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[0,3], c=0.738, marker='p',  s=20, vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[1,3], c=0.931, marker='p',  s=40, vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20181031['PCTarray_PHAIL_out'].data[2,2],  RMA4_20181031['PCTarray_PHAIL_out'].data[2,3], c=0.993, marker='p', s=50, vmin=0.5, vmax=1.0)

plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181215['PCTarray_PHAIL_out'].data[0,3], c=0.930, marker='v', s=20, vmin=0.5, vmax=1.0)
plt.scatter(RMA4_20181215['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181215['PCTarray_PHAIL_out'].data[1,3], c=0.747, marker='v', s=20, vmin=0.5, vmax=1.0)

plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[0,2],  RMA4_20181218['PCTarray_PHAIL_out'].data[0,3], c=0.599, marker='^', s=20, vmin=0.5, vmax=1.0)
pcm = plt.scatter(RMA4_20181218['PCTarray_PHAIL_out'].data[1,2],  RMA4_20181218['PCTarray_PHAIL_out'].data[1,3], c=0.964, marker='^', s=20, vmin=0.5, vmax=1.0)
plt.colorbar(pcm)


plt.scatter(RMA3_20190305['PCTarray_NOPHAIL_out'].data[2],  RMA3_20190305['PCTarray_NOPHAIL_out'].data[3], s=30, marker='x', color='k')
plt.scatter(RMA3_20190305['PCTarray_NOPHAIL_out'].data[2],  RMA3_20190305['PCTarray_NOPHAIL_out'].data[3], s=30, marker='x', color='k')
plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,3], s=30, marker='x', color='k')
plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,3], s=30, marker='x', color='k')
plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,3], s=30, marker='x', color='k')
plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,3], s=30, marker='x', color='k')
plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,3], s=30, marker='x', color='k')
plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,3], s=30, marker='x', color='k')
plt.scatter(RMA4_20181031['PCTarray_NOPHAIL_out'].data[2],  RMA4_20181031['PCTarray_NOPHAIL_out'].data[3], s=30, marker='x', color='k')
plt.scatter(RMA4_20181218['PCTarray_NOPHAIL_out'].data[2],  RMA4_20181218['PCTarray_NOPHAIL_out'].data[3], s=30, marker='x', color='k')

plt.grid(True)
plt.xlabel('MINPCT(37)')
plt.ylabel('MINPCTT(85)')
plt.xlim([80,230])
plt.ylim([50,170])
plt.suptitle('RMA3+RMA4',y=0.9)





















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:39:58 2021

@author: victoria.galligani
"""
import h5py 
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt;
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from os import listdir
from os.path import isfile, join
import datetime as dt
from datetime import datetime
import netCDF4 as nc
import xarray  as xr
import matplotlib.colors as colors_mat
from collections import defaultdict
import pandas as pd
import seaborn as sns
import Plots as PLOTS
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec


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

def plot_TBs_all(files, platform, idir, options, instr):
    
    """Create a 2x3 colormap with BT(37, 89, 157)w. BT contours on the PD maps. Also include ACP alertsa del SMN. 
        Include PF ellipse from database"""

    plt.matplotlib.rc('font', family='DejaVu Sans', size = 12)

    fontsize   = 12
    linewidths = 1

    geo_reg_shp = '/home/victoria.galligani/Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
    geo_reg = shpreader.Reader(geo_reg_shp)
        
    cmaps = GMI_colormap()
    
    countries = shpreader.Reader('/home/victoria.galligani/Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')

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
        
    for ifile in files:
        if ifile.startswith(platform): 
            # Read file
            fname = idir+ifile
            f = h5py.File( fname, 'r')
            if 'MHS' in instr: 
                # MHS only S1: tb.shape = [:,]
                tb = f[u'/S1/Tc'][:,:,:]           
                lon = f[u'/S1/Longitude'][:,:] 
                lat = f[u'/S1/Latitude'][:,:]
 
                # keep domain of interest only by keeping those where the center nadir obs is inside domain
                inside   = np.logical_and(np.logical_and(lon[:,45] >= opts['xlim_min'], lon[:,45] <= opts['xlim_max']), 
                          np.logical_and(lat[:,45] >= opts['ylim_min'], lat[:,45] <= opts['ylim_max']))
                
                lon = lon[inside] 
                lat = lat[inside]
                tb  = tb[inside,:]
                
                print(ifile)
                yoi = int(ifile[25:29])
                moi = int(ifile[29:31]) #"%02d" % int(ifile[26:28])
                doi = int(ifile[31:33]) #"%02d" % int(ifile[28:30])
                print('day of interest: '+str(doi) )
                pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 
                
            
            
            if 'SSMIS' in instr: 
                # SSMIS tiene S1-S4:  
                # S1[:,3] // 19V,H, 22V
                # S2[:,2] // 37V,H
                # S3[:,4] // 150H 183+/-1H 183+/-3H 183+/-7H
                # S4[:,2] // 91V 91H
                tb1 = f[u'/S1/Tc'][:,:,:]           
                tb2 = f[u'/S2/Tc'][:,:,:]           
                tb3 = f[u'/S3/Tc'][:,:,:]           
                tb4 = f[u'/S4/Tc'][:,:,:]           
                lon1 = f[u'/S1/Longitude'][:,:] 
                lat1 = f[u'/S1/Latitude'][:,:]
                lon2 = f[u'/S2/Longitude'][:,:] 
                lat2 = f[u'/S2/Latitude'][:,:]           
                lon3 = f[u'/S3/Longitude'][:,:] 
                lat3 = f[u'/S3/Latitude'][:,:]                          
                lon4 = f[u'/S4/Longitude'][:,:] 
                lat4 = f[u'/S4/Latitude'][:,:]    
                
                # keep domain of interest only by keeping those where the center nadir obs is inside domain
                inside2   = np.logical_and(np.logical_and(lon2[:,45] >= opts['xlim_min'], lon2[:,45] <= opts['xlim_max']), 
                         np.logical_and(lat2[:,45] >= opts['ylim_min'], lat2[:,45] <= opts['ylim_max']))

                lon2 = lon2[inside2] 
                lat2 = lat2[inside2]
                tb2  = tb2[inside2,:]
                              
                inside4 = np.logical_and(np.logical_and(lon4[:,90] >= opts['xlim_min'], lon4[:,90] <= opts['xlim_max']), 
                         np.logical_and(lat4[:,90] >= opts['ylim_min'], lat4[:,90] <= opts['ylim_max']))

                lon4 = lon4[inside4] 
                lat4 = lat4[inside4]
                tb4  = tb4[inside4,:]
                              
                inside3 = np.logical_and(np.logical_and(lon3[:,90] >= opts['xlim_min'], lon3[:,90] <= opts['xlim_max']), 
                         np.logical_and(lat3[:,90] >= opts['ylim_min'], lat3[:,90] <= opts['ylim_max']))

                lon3 = lon3[inside3] 
                lat3 = lat3[inside3]
                tb3  = tb3[inside3,:]
   
                print(ifile)
                yoi = int(ifile[24:28])
                moi = int(ifile[28:30]) #"%02d" % int(ifile[26:28])
                doi = int(ifile[30:32]) #"%02d" % int(ifile[28:30])
                print('day of interest: '+str(doi) )
                pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 
                

            if 'ATMS' in instr: 
                # ATMS tiene S1-S4:  
                # S1[:,1] // K 23.8 GHz
                # S2[:,1] // Ka 31.4 GHz
                # S3[:,1] // W 88.2 GHz
                # S4[:,6] // 165h, 183.31±7, 183.31±4.5, 183.31±3, 183.31±1.8, 183.31±1
                tb1 = f[u'/S1/Tc'][:,:,:]           
                tb2 = f[u'/S2/Tc'][:,:,:]           
                tb3 = f[u'/S3/Tc'][:,:,:]           
                tb4 = f[u'/S4/Tc'][:,:,:]           
                lon1 = f[u'/S1/Longitude'][:,:] 
                lat1 = f[u'/S1/Latitude'][:,:]
                lon2 = f[u'/S2/Longitude'][:,:] 
                lat2 = f[u'/S2/Latitude'][:,:]           
                lon3 = f[u'/S3/Longitude'][:,:] 
                lat3 = f[u'/S3/Latitude'][:,:]                          
                lon4 = f[u'/S4/Longitude'][:,:] 
                lat4 = f[u'/S4/Latitude'][:,:]    

                print(ifile)
                if 'NPP' in ifile:
                    yoi = int(ifile[23:27])
                    moi = int(ifile[27:29]) #"%02d" % int(ifile[26:28])
                    doi = int(ifile[29:31]) #"%02d" % int(ifile[28:30])
                    print('day of interest: '+str(doi) )
                    pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 

                if 'NOAA20' in ifile:
                    yoi = int(ifile[26:30])
                    moi = int(ifile[30:32]) #"%02d" % int(ifile[26:28])
                    doi = int(ifile[32:34]) #"%02d" % int(ifile[28:30])
                    print('day of interest: '+str(doi) )
                    pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 
                    
                # keep domain of interest only by keeping those where the center nadir obs is inside domain
                inside2   = np.logical_and(np.logical_and(lon2[:,48] >= opts['xlim_min'], lon2[:,48] <= opts['xlim_max']), 
                         np.logical_and(lat2[:,48] >= opts['ylim_min'], lat2[:,48] <= opts['ylim_max']))

                lon2 = lon2[inside2] 
                lat2 = lat2[inside2]
                tb2  = tb2[inside2,:]                    
                
                inside3 = np.logical_and(np.logical_and(lon3[:,48] >= opts['xlim_min'], lon3[:,48] <= opts['xlim_max']), 
                         np.logical_and(lat3[:,48] >= opts['ylim_min'], lat3[:,48] <= opts['ylim_max']))

                lon3 = lon3[inside3] 
                lat3 = lat3[inside3]
                tb3  = tb3[inside3,:]                    
                
                                    
                inside4 = np.logical_and(np.logical_and(lon4[:,48] >= opts['xlim_min'], lon4[:,48] <= opts['xlim_max']), 
                         np.logical_and(lat4[:,48] >= opts['ylim_min'], lat4[:,48] <= opts['ylim_max']))

                lon4 = lon4[inside3] 
                lat4 = lat4[inside3]
                tb4  = tb4[inside3,:]                              
                         
            if 'AMSR2' in instr: 
                # AMSR2 tiene S1-S6:  
                # S1[:,] // 2    6.9  GHz V,H   [243]
                # S2[:,] // 2   16.65 GHz V,H   [243]
                # S3[:,] // 2   18.70 GHz V,H   [243]
                # S4[:,] // 2   23.80 GHz V,H   [243]
                # S5[:,] // 2   36.50 GHz V,H   [486]
                # S6[:,] // 2   89.00 GHz V,H   [486]
                tb5 = f[u'/S5/Tc'][:,:,:]           
                tb6 = f[u'/S6/Tc'][:,:,:]           
                lon5 = f[u'/S5/Longitude'][:,:] 
                lat5 = f[u'/S5/Latitude'][:,:]
                lon6 = f[u'/S6/Longitude'][:,:] 
                lat6 = f[u'/S6/Latitude'][:,:]           

                print(ifile)
                yoi = int(ifile[27:31])
                moi = int(ifile[31:33]) #"%02d" % int(ifile[26:28])
                doi = int(ifile[33:35]) #"%02d" % int(ifile[28:30])
                print('day of interest: '+str(doi) )
                pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 
                
                inside5 = np.logical_and(np.logical_and(lon5[:,243] >= opts['xlim_min'], lon5[:,243] <= opts['xlim_max']), 
                         np.logical_and(lat5[:,243] >= opts['ylim_min'], lat5[:,243] <= opts['ylim_max']))

                lon5 = lon5[inside5] 
                lat5 = lat5[inside5]
                tb5  = tb5[inside5,:]      
                
                inside6 = np.logical_and(np.logical_and(lon6[:,243] >= opts['xlim_min'], lon6[:,243] <= opts['xlim_max']), 
                         np.logical_and(lat6[:,243] >= opts['ylim_min'], lat6[:,243] <= opts['ylim_max']))

                lon6 = lon6[inside6] 
                lat6 = lat6[inside6]
                tb6  = tb6[inside6,:]                          
                
            f.close()
            
            # data_tb37 = PLOTS.apply_geofence_on_data (tb[:,:,5], lat, lon, options['ylim_min'], options['ylim_max'], options['xlim_min'],
            #                     options['xlim_max'])  
            # data_tb89 = PLOTS.apply_geofence_on_data (tb[:,:,7], lat, lon, options['ylim_min'], options['ylim_max'], options['xlim_min'],
            #                     options['xlim_max'])  
            # data_tb157 = PLOTS.apply_geofence_on_data (tb[:,:,0], lat, lon, options['ylim_min'], options['ylim_max'], options['xlim_min'],
            #                     options['xlim_max'])  
    
    
            fig = plt.figure(figsize=(12,7)) 
            gs1 = gridspec.GridSpec(2, 3)
            
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
            if 'MHS' in instr:    
                print('do nothin')
            if 'SSMIS' in instr:    
                im = plt.scatter(lon2, lat2, 
                   c=tb2[:,:,1], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])             
            if 'ATMS' in instr:    
                im = plt.scatter(lon2, lat2, 
                   c=tb2[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])      
            if 'AMSR2' in instr:    
                im = plt.scatter(lon5, lat5, 
                   c=tb5[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])      
            plt.title('BT 37 GHz', fontsize=fontsize)
            ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
            ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
            ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize) 
            for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize)                 
            ax1.text(0.05,1.10,'(a)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
            p1 = ax1.get_position().get_points().flatten()
            # For each polygon detected plot a polygon
            for i in range(len(pol_n)):
                coord = pol_n[i]
                coord.append(coord[0]) #repeat the first point to create a 'closed loop'
                ys, xs = zip(*coord) #create lists of x and y values
                plt.plot(xs,ys, '-m') 
            plt.text(-75, -42, 'ACPs durante el dia', color='m')       
        
            
            # BT(89)              
            ax1 = plt.subplot(gs1[0,1], projection=ccrs.PlateCarree())
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
            if 'MHS' in instr:    
                im = plt.scatter(lon[:], lat[:], 
                   c=tb[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
            if 'SSMIS' in instr:    
                im = plt.scatter(lon4, lat4, 
                   c=tb4[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])            
            if 'ATMS' in instr:    
                im = plt.scatter(lon3, lat3, 
                   c=tb3[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])         
            if 'AMSR2' in instr:    
                im = plt.scatter(lon6, lat6, 
                   c=tb6[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])    
                
            plt.title('BT 89 GHz', fontsize=fontsize)
            ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
            ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
            ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax1.xaxis.set_major_formatter(lon_formatter)
            ax1.yaxis.set_major_formatter(lat_formatter)
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize) 
            for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize)                 
            ax1.text(0.05,1.10,'(b)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
            p1 = ax1.get_position().get_points().flatten()
                
            # BT(166)           
            ax1 = plt.subplot(gs1[0,2], projection=ccrs.PlateCarree())
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
            if 'MHS' in instr:        
                im = plt.scatter(lon[:], lat[:], 
                   c=tb[:,:,1], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
                plt.title('BT 157 GHz', fontsize=fontsize)
                
            if 'SSMIS' in instr:    
                im = plt.scatter(lon3, lat3, 
                   c=tb3[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])       
                plt.title('BT 150 GHz', fontsize=fontsize)

            if 'ATMS' in instr:    
                im = plt.scatter(lon4, lat4, 
                   c=tb4[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])     
                plt.title('BT 165 GHz', fontsize=fontsize)
                       
            #divider = make_axes_locatable(ax1)
            #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
            #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
            ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
            ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
            ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax1.xaxis.set_major_formatter(lon_formatter)
            ax1.yaxis.set_major_formatter(lat_formatter)
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
                
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize) 
            for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize)                 
                
            ax1.text(0.05,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
            #ax1.text(-0.1,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
        
            p2 = ax1.get_position().get_points().flatten()
            # #ax_cbar = fig.add_axes([p1[0], 0.45, p2[2]-p1[0], 0.05])
            ax_cbar = fig.add_axes([0.92, 0.55, 0.02, 0.35])
            cbar = fig.colorbar(im,  cax=ax_cbar, shrink=1,ticks=np.arange(50,325,25), 
                                extend='both', orientation="vertical")
            cbar.set_label('BT (K)', fontsize=fontsize)
            
            if 'MHS' in instr:     
                fig.suptitle(instr +' on '+ifile[25:33]+ ' at '+ ifile[34:47] +' UTC' ,fontweight='bold' )
            if 'SSMIS' in instr:     
                fig.suptitle(instr +' on '+ifile[24:32]+ ' at '+ ifile[33:46] +' UTC' ,fontweight='bold' )
            if 'ATMS' in instr:     
                if 'NPP' in ifile:
                    fig.suptitle(instr +' on '+ifile[23:31]+ ' at '+ ifile[33:37] +' UTC' ,fontweight='bold' )
                if 'NOAA20' in ifile:
                    fig.suptitle(instr +' on '+ifile[26:34]+ ' at '+ ifile[36:40] +' UTC' ,fontweight='bold' )
            if 'AMSR2' in instr:     
                fig.suptitle(instr +' on '+ifile[27:35]+ ' at '+ ifile[37:41] +' UTC' ,fontweight='bold' )
            
            #plt.tight_layout()
            #plt.subplots_adjust(top=0.899)
            #plt.savefig(options['path']+'/'+options['name']+'.eps')
            plt.savefig(options['path']+instr+'/'+ifile+'.png')
            plt.close()

    return 

def plot_TBs(files, platform, idir, options, iop, instr):
    
    """Create a 2x3 colormap with BT(37, 89, 157)w. BT contours on the PD maps. Also include ACP alertsa del SMN. 
        Include PF ellipse from database"""

    plt.matplotlib.rc('font', family='DejaVu Sans', size = 12)

    fontsize   = 12
    linewidths = 1

    geo_reg_shp = '/home/victoria.galligani/Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
    geo_reg = shpreader.Reader(geo_reg_shp)
        
    cmaps = GMI_colormap()
    
    countries = shpreader.Reader('/home/victoria.galligani/Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')

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
        
    for ifile in files:
        if ifile.startswith(platform): 
            if iop in ifile:
                # Read file
                fname = idir+ifile
                f = h5py.File( fname, 'r')
                if 'MHS' in instr: 
                    # MHS only S1: tb.shape = [:,]
                    tb = f[u'/S1/Tc'][:,:,:]           
                    lon = f[u'/S1/Longitude'][:,:] 
                    lat = f[u'/S1/Latitude'][:,:]
     
                    # keep domain of interest only by keeping those where the center nadir obs is inside domain
                    inside   = np.logical_and(np.logical_and(lon[:,45] >= opts['xlim_min'], lon[:,45] <= opts['xlim_max']), 
                              np.logical_and(lat[:,45] >= opts['ylim_min'], lat[:,45] <= opts['ylim_max']))
                    
                    lon = lon[inside] 
                    lat = lat[inside]
                    tb  = tb[inside,:]
                    
                    print(ifile)
                    yoi = int(ifile[25:29])
                    moi = int(ifile[29:31]) #"%02d" % int(ifile[26:28])
                    doi = int(ifile[31:33]) #"%02d" % int(ifile[28:30])
                    print('day of interest: '+str(doi) )
                    pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 
                    
                
                
                if 'SSMIS' in instr: 
                    # SSMIS tiene S1-S4:  
                    # S1[:,3] // 19V,H, 22V
                    # S2[:,2] // 37V,H
                    # S3[:,4] // 150H 183+/-1H 183+/-3H 183+/-7H
                    # S4[:,2] // 91V 91H
                    tb1 = f[u'/S1/Tc'][:,:,:]           
                    tb2 = f[u'/S2/Tc'][:,:,:]           
                    tb3 = f[u'/S3/Tc'][:,:,:]           
                    tb4 = f[u'/S4/Tc'][:,:,:]           
                    lon1 = f[u'/S1/Longitude'][:,:] 
                    lat1 = f[u'/S1/Latitude'][:,:]
                    lon2 = f[u'/S2/Longitude'][:,:] 
                    lat2 = f[u'/S2/Latitude'][:,:]           
                    lon3 = f[u'/S3/Longitude'][:,:] 
                    lat3 = f[u'/S3/Latitude'][:,:]                          
                    lon4 = f[u'/S4/Longitude'][:,:] 
                    lat4 = f[u'/S4/Latitude'][:,:]    
                    
                    # keep domain of interest only by keeping those where the center nadir obs is inside domain
                    inside2   = np.logical_and(np.logical_and(lon2[:,45] >= opts['xlim_min'], lon2[:,45] <= opts['xlim_max']), 
                             np.logical_and(lat2[:,45] >= opts['ylim_min'], lat2[:,45] <= opts['ylim_max']))
    
                    lon2 = lon2[inside2] 
                    lat2 = lat2[inside2]
                    tb2  = tb2[inside2,:]
                                  
                    inside4 = np.logical_and(np.logical_and(lon4[:,90] >= opts['xlim_min'], lon4[:,90] <= opts['xlim_max']), 
                             np.logical_and(lat4[:,90] >= opts['ylim_min'], lat4[:,90] <= opts['ylim_max']))
    
                    lon4 = lon4[inside4] 
                    lat4 = lat4[inside4]
                    tb4  = tb4[inside4,:]
                                  
                    inside3 = np.logical_and(np.logical_and(lon3[:,90] >= opts['xlim_min'], lon3[:,90] <= opts['xlim_max']), 
                             np.logical_and(lat3[:,90] >= opts['ylim_min'], lat3[:,90] <= opts['ylim_max']))
    
                    lon3 = lon3[inside3] 
                    lat3 = lat3[inside3]
                    tb3  = tb3[inside3,:]
       
                    print(ifile)
                    yoi = int(ifile[24:28])
                    moi = int(ifile[28:30]) #"%02d" % int(ifile[26:28])
                    doi = int(ifile[30:32]) #"%02d" % int(ifile[28:30])
                    print('day of interest: '+str(doi) )
                    pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 
                    

                if 'ATMS' in instr: 
                    # ATMS tiene S1-S4:  
                    # S1[:,1] // K 23.8 GHz
                    # S2[:,1] // Ka 31.4 GHz
                    # S3[:,1] // W 88.2 GHz
                    # S4[:,6] // 165h, 183.31±7, 183.31±4.5, 183.31±3, 183.31±1.8, 183.31±1
                    tb1 = f[u'/S1/Tc'][:,:,:]           
                    tb2 = f[u'/S2/Tc'][:,:,:]           
                    tb3 = f[u'/S3/Tc'][:,:,:]           
                    tb4 = f[u'/S4/Tc'][:,:,:]           
                    lon1 = f[u'/S1/Longitude'][:,:] 
                    lat1 = f[u'/S1/Latitude'][:,:]
                    lon2 = f[u'/S2/Longitude'][:,:] 
                    lat2 = f[u'/S2/Latitude'][:,:]           
                    lon3 = f[u'/S3/Longitude'][:,:] 
                    lat3 = f[u'/S3/Latitude'][:,:]                          
                    lon4 = f[u'/S4/Longitude'][:,:] 
                    lat4 = f[u'/S4/Latitude'][:,:]    

                    print(ifile)
                    if 'NPP' in ifile:
                        yoi = int(ifile[23:27])
                        moi = int(ifile[27:29]) #"%02d" % int(ifile[26:28])
                        doi = int(ifile[29:31]) #"%02d" % int(ifile[28:30])
                        print('day of interest: '+str(doi) )
                        pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 

                    if 'NOAA20' in ifile:
                        yoi = int(ifile[26:30])
                        moi = int(ifile[30:32]) #"%02d" % int(ifile[26:28])
                        doi = int(ifile[32:34]) #"%02d" % int(ifile[28:30])
                        print('day of interest: '+str(doi) )
                        pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 
                        
                    # keep domain of interest only by keeping those where the center nadir obs is inside domain
                    inside2   = np.logical_and(np.logical_and(lon2[:,48] >= opts['xlim_min'], lon2[:,48] <= opts['xlim_max']), 
                             np.logical_and(lat2[:,48] >= opts['ylim_min'], lat2[:,48] <= opts['ylim_max']))
    
                    lon2 = lon2[inside2] 
                    lat2 = lat2[inside2]
                    tb2  = tb2[inside2,:]                    
                    
                    inside3 = np.logical_and(np.logical_and(lon3[:,48] >= opts['xlim_min'], lon3[:,48] <= opts['xlim_max']), 
                             np.logical_and(lat3[:,48] >= opts['ylim_min'], lat3[:,48] <= opts['ylim_max']))
    
                    lon3 = lon3[inside3] 
                    lat3 = lat3[inside3]
                    tb3  = tb3[inside3,:]                    
                    
                                        
                    inside4 = np.logical_and(np.logical_and(lon4[:,48] >= opts['xlim_min'], lon4[:,48] <= opts['xlim_max']), 
                             np.logical_and(lat4[:,48] >= opts['ylim_min'], lat4[:,48] <= opts['ylim_max']))
    
                    lon4 = lon4[inside3] 
                    lat4 = lat4[inside3]
                    tb4  = tb4[inside3,:]                              
                             
                if 'AMSR2' in instr: 
                    # AMSR2 tiene S1-S6:  
                    # S1[:,] // 2    6.9  GHz V,H   [243]
                    # S2[:,] // 2   16.65 GHz V,H   [243]
                    # S3[:,] // 2   18.70 GHz V,H   [243]
                    # S4[:,] // 2   23.80 GHz V,H   [243]
                    # S5[:,] // 2   36.50 GHz V,H   [486]
                    # S6[:,] // 2   89.00 GHz V,H   [486]
                    tb5 = f[u'/S5/Tc'][:,:,:]           
                    tb6 = f[u'/S6/Tc'][:,:,:]           
                    lon5 = f[u'/S5/Longitude'][:,:] 
                    lat5 = f[u'/S5/Latitude'][:,:]
                    lon6 = f[u'/S6/Longitude'][:,:] 
                    lat6 = f[u'/S6/Latitude'][:,:]           

                    print(ifile)
                    yoi = int(ifile[27:31])
                    moi = int(ifile[31:33]) #"%02d" % int(ifile[26:28])
                    doi = int(ifile[33:35]) #"%02d" % int(ifile[28:30])
                    print('day of interest: '+str(doi) )
                    pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 
                    
                    inside5 = np.logical_and(np.logical_and(lon5[:,243] >= opts['xlim_min'], lon5[:,243] <= opts['xlim_max']), 
                             np.logical_and(lat5[:,243] >= opts['ylim_min'], lat5[:,243] <= opts['ylim_max']))
    
                    lon5 = lon5[inside5] 
                    lat5 = lat5[inside5]
                    tb5  = tb5[inside5,:]      
                    
                    inside6 = np.logical_and(np.logical_and(lon6[:,243] >= opts['xlim_min'], lon6[:,243] <= opts['xlim_max']), 
                             np.logical_and(lat6[:,243] >= opts['ylim_min'], lat6[:,243] <= opts['ylim_max']))
    
                    lon6 = lon6[inside6] 
                    lat6 = lat6[inside6]
                    tb6  = tb6[inside6,:]                          
                    
                f.close()
                
                # data_tb37 = PLOTS.apply_geofence_on_data (tb[:,:,5], lat, lon, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                #                     options['xlim_max'])  
                # data_tb89 = PLOTS.apply_geofence_on_data (tb[:,:,7], lat, lon, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                #                     options['xlim_max'])  
                # data_tb157 = PLOTS.apply_geofence_on_data (tb[:,:,0], lat, lon, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                #                     options['xlim_max'])  
        
        
                fig = plt.figure(figsize=(12,7)) 
                gs1 = gridspec.GridSpec(2, 3)
                
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
                if 'MHS' in instr:    
                    print('do nothin')
                if 'SSMIS' in instr:    
                    im = plt.scatter(lon2, lat2, 
                       c=tb2[:,:,1], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])             
                if 'ATMS' in instr:    
                    im = plt.scatter(lon2, lat2, 
                       c=tb2[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])      
                if 'AMSR2' in instr:    
                    im = plt.scatter(lon5, lat5, 
                       c=tb5[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])      
                plt.title('BT 37 GHz', fontsize=fontsize)
                ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
                ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
                ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
                lon_formatter = LongitudeFormatter(zero_direction_label=True)
                lat_formatter = LatitudeFormatter()
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                for tick in ax1.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize) 
                for tick in ax1.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize)                 
                ax1.text(0.05,1.10,'(a)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
                p1 = ax1.get_position().get_points().flatten()
                # For each polygon detected plot a polygon
                for i in range(len(pol_n)):
                    coord = pol_n[i]
                    coord.append(coord[0]) #repeat the first point to create a 'closed loop'
                    ys, xs = zip(*coord) #create lists of x and y values
                    plt.plot(xs,ys, '-m') 
                plt.text(-75, -42, 'ACPs durante el dia', color='m')       
            
                
                # BT(89)              
                ax1 = plt.subplot(gs1[0,1], projection=ccrs.PlateCarree())
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
                if 'MHS' in instr:    
                    im = plt.scatter(lon[:], lat[:], 
                       c=tb[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
                if 'SSMIS' in instr:    
                    im = plt.scatter(lon4, lat4, 
                       c=tb4[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])            
                if 'ATMS' in instr:    
                    im = plt.scatter(lon3, lat3, 
                       c=tb3[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])         
                if 'AMSR2' in instr:    
                    im = plt.scatter(lon6, lat6, 
                       c=tb6[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])    
                    
                plt.title('BT 89 GHz', fontsize=fontsize)
                ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
                ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
                ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
                lon_formatter = LongitudeFormatter(zero_direction_label=True)
                lat_formatter = LatitudeFormatter()
                ax1.xaxis.set_major_formatter(lon_formatter)
                ax1.yaxis.set_major_formatter(lat_formatter)
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                for tick in ax1.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize) 
                for tick in ax1.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize)                 
                ax1.text(0.05,1.10,'(b)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
                p1 = ax1.get_position().get_points().flatten()
                    
                # BT(166)           
                ax1 = plt.subplot(gs1[0,2], projection=ccrs.PlateCarree())
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
                if 'MHS' in instr:        
                    im = plt.scatter(lon[:], lat[:], 
                       c=tb[:,:,1], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
                    plt.title('BT 157 GHz', fontsize=fontsize)
                    
                if 'SSMIS' in instr:    
                    im = plt.scatter(lon3, lat3, 
                       c=tb3[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])       
                    plt.title('BT 150 GHz', fontsize=fontsize)

                if 'ATMS' in instr:    
                    im = plt.scatter(lon4, lat4, 
                       c=tb4[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])     
                    plt.title('BT 165 GHz', fontsize=fontsize)
                           
                #divider = make_axes_locatable(ax1)
                #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
                #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
                ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
                ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
                ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
                lon_formatter = LongitudeFormatter(zero_direction_label=True)
                lat_formatter = LatitudeFormatter()
                ax1.xaxis.set_major_formatter(lon_formatter)
                ax1.yaxis.set_major_formatter(lat_formatter)
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                    
                for tick in ax1.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize) 
                for tick in ax1.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize)                 
                    
                ax1.text(0.05,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
                #ax1.text(-0.1,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
            
                p2 = ax1.get_position().get_points().flatten()
                # #ax_cbar = fig.add_axes([p1[0], 0.45, p2[2]-p1[0], 0.05])
                ax_cbar = fig.add_axes([0.92, 0.55, 0.02, 0.35])
                cbar = fig.colorbar(im,  cax=ax_cbar, shrink=1,ticks=np.arange(50,325,25), 
                                    extend='both', orientation="vertical")
                cbar.set_label('BT (K)', fontsize=fontsize)
                
                if 'MHS' in instr:     
                    fig.suptitle(instr +' on '+ifile[25:33]+ ' at '+ ifile[34:47] +' UTC' ,fontweight='bold' )
                if 'SSMIS' in instr:     
                    fig.suptitle(instr +' on '+ifile[24:32]+ ' at '+ ifile[33:46] +' UTC' ,fontweight='bold' )
                if 'ATMS' in instr:     
                    if 'NPP' in ifile:
                        fig.suptitle(instr +' on '+ifile[23:31]+ ' at '+ ifile[33:37] +' UTC' ,fontweight='bold' )
                    if 'NOAA20' in ifile:
                        fig.suptitle(instr +' on '+ifile[26:34]+ ' at '+ ifile[36:40] +' UTC' ,fontweight='bold' )
                if 'AMSR2' in instr:     
                    fig.suptitle(instr +' on '+ifile[27:35]+ ' at '+ ifile[37:41] +' UTC' ,fontweight='bold' )
                #plt.tight_layout()
                #plt.subplots_adjust(top=0.899)
                #plt.savefig(options['path']+'/'+options['name']+'.eps')
                #plt.savefig(options['path']+'_plots'+'/'+options['name']+'_ALLGMICHANNELS.png')
                #plt.close()

    return 

if __name__ == '__main__':

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12

    # Domain of interest
    xlim_min = -75; 
    xlim_max = -50; 
    ylim_min = -40; 
    ylim_max = -19; 
    darea_do =  (ylim_max-ylim_min)*(xlim_max-xlim_min) # "in degrees"
    
    opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
            'ylim_min': ylim_min, 'ylim_max': ylim_max, 
            'path': '/datosmunin2/victoria.galligani/DATOS_mw/Plots/'}
    
    ##############################################################################
    prov = genfromtxt("/home/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/home/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')
    
    mhs_dir     = '/home/victoria.galligani/datosmunin2/DATOS_mw/MHS/'
    ssmis_dir   = '/home/victoria.galligani/datosmunin2/DATOS_mw/SSMIS/'
    atms_dir    = '/home/victoria.galligani/datosmunin2/DATOS_mw/ATMS/'
    amsr2_dir    = '/home/victoria.galligani/datosmunin2/DATOS_mw/AMSR2/'

    ##############################################################################
    
    #onlyfiles = [f for f in listdir(mhs_dir) if isfile(join(mhs_dir, f))]
    #plot_TBs(onlyfiles, '1C', mhs_dir, opts, '20181102', 'MHS')

    #onlyfiles = [f for f in listdir(ssmis_dir) if isfile(join(ssmis_dir, f))]
    #plot_TBs(onlyfiles, '1C', ssmis_dir, opts, '20181102', 'SSMIS')

    #onlyfiles = [f for f in listdir(atms_dir) if isfile(join(atms_dir, f))]
    #plot_TBs(onlyfiles, '1C', atms_dir, opts, '20181102', 'ATMS')

    #onlyfiles = [f for f in listdir(amsr2_dir) if isfile(join(amsr2_dir, f))]
    #plot_TBs(onlyfiles, '1C', amsr2_dir, opts, '20181102', 'AMSR2')

    ##############################################################################
    #onlyfiles = [f for f in listdir(mhs_dir) if isfile(join(mhs_dir, f))]
    #plot_TBs_all(onlyfiles, '1C', mhs_dir, opts, 'MHS')

    #onlyfiles = [f for f in listdir(ssmis_dir) if isfile(join(ssmis_dir, f))]
    #plot_TBs_all(onlyfiles, '1C', ssmis_dir, opts, 'SSMIS')

    onlyfiles = [f for f in listdir(atms_dir) if isfile(join(atms_dir, f))]
    plot_TBs_all(onlyfiles, '1C', atms_dir, opts, 'ATMS')

    onlyfiles = [f for f in listdir(amsr2_dir) if isfile(join(amsr2_dir, f))]
    plot_TBs_all(onlyfiles, '1C', amsr2_dir, opts, 'AMSR2')




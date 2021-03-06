#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Created on Wed Feb 03 2021
-------------------------------------------------------------------------------
@purpose : Open GPM overpasses (GPM GMI Common Calibrated Brightness 
Temperatures Collocated L1C 1.5 hours 13 km V05 - GPM_1CGPMGMI - at GES DISC) 
in the region of interest ([-40, -70] and [-20,-50]) and during the 
CACTI-RELAMPAGO field campaign (01/September/2018 – 30/April/2019). For this
selection, https://search.earthdata.nasa.gov shows 916 granules (approx. 25GB)
Downloaded to /Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/GMI
@author  : V. Galligani
@email   : victoria.galligani@cima.fcen.uba.ar
-------------------------------------------------------------------------------
@TODOS(?):
    * Save relevant data to .nc file to save space and handle data. 
    * RR, base de datos de chuntao? BB, conv/strati 
    * use latest ACP
-------------------------------------------------------------------------------
"""
#################################################################
# REMEMBER TO USE conda activate py37 in ernest
#################################################################
# Load libraries
import sys
import numpy as np
import h5py 
import matplotlib.pyplot as plt
import Plots as PlottingGMITools
from os import listdir
from os.path import isfile, join
import shutil
import pandas as pd
import osgeo
import geopandas as gpd
from shapely.ops import cascaded_union
from numpy import genfromtxt;
from math import pi, cos, sin
from pyhdf import SD
from pyproj import Proj, transform
#
# Some matplotlib figure definitions
plt.matplotlib.rc('font', family='serif', size = 10)
plt.rcParams['xtick.labelsize']=10
plt.rcParams['ytick.labelsize']=10
        
#################################################################
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
    PCT10 = 2.5  * tb_s1_gmi[:,:,0] - 1.5  * tb_s1_gmi[:,:,1] 
    PCT19 = 2.4  * tb_s1_gmi[:,:,2] - 1.4  * tb_s1_gmi[:,:,3] 
    PCT37 = 2.15 * tb_s1_gmi[:,:,5] - 1.15 * tb_s1_gmi[:,:,6] 
    PCT89 = 1.7  * tb_s1_gmi[:,:,7] - 0.7  * tb_s1_gmi[:,:,8] 
    
    return PCT10, PCT19, PCT37, PCT89

#################################################################
def read_hdf5(filename):
    """
    -------------------------------------------------------------
    this program is to read from a HDF5 file and write the data to an python dictionary 
    -------------------------------------------------------------
    """
    import h5py 
    dic={}
    with h5py.File(filename, 'r') as h5file:
         for key, item in h5file['/'].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                dic[key] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                dic[key] = recursively_load_dict_contents_from_group(h5file, '/' + key + '/')
    return dic

#################################################################
def read_hdf(filename):
    
    # open the hdf file
    hdf  = SD.SD(filename)
    # Make dictionary and read the HDF file
    ggpf_data = {} 
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        ggpf_data[name] = data
    hdf.end()

    return ggpf_data

#################################################################
def get_ellipse_info(pf_data, select, j):

    u=pf_data['r_lon'][select][j]               #x-position of the center
    v=pf_data['r_lat'][select][j]               #y-position of the center
    a=pf_data['r_minor_degree'][select][j]      #radius on the x-axis
    b=pf_data['r_major_degree'][select][j]      #radius on the y-axis
    t_rot=pf_data['r_orientation'][select][j]   #rotation angle
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
    #2-D rotation matrix
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
            
    return u+Ell_rot[0,:], v+Ell_rot[1,:]

#################################################################
def get_ellipse_info_MAJ(pf_data, select, j):

    u=pf_data['R_LON'][select][j]               #x-position of the center
    v=pf_data['R_LAT'][select][j]               #y-position of the center
    a=pf_data['R_MINOR'][select][j]      #radius on the x-axis
    b=pf_data['R_MAJOR'][select][j]      #radius on the y-axis
    t_rot=pf_data['R_ORIENTATION'][select][j]   #rotation angle
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
    #2-D rotation matrix
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
            
    return u+Ell_rot[0,:], v+Ell_rot[1,:]

#################################################################

def get_ellipse_info_MAJ_km(pf_data, select, j):

    u=pf_data['R_LON'][select][j]               #x-position of the center
    v=pf_data['R_LAT'][select][j]               #y-position of the center
    
    # TESTING THIS PROJ.  math.cos(math.radians(1))
    a=pf_data['R_MINOR'][select][j]/111   # x-axis (1degre == 92 km)
    b=pf_data['R_MAJOR'][select][j]/111# y-axis

    # OLD
    #a=pf_data['R_MINOR'][select][j]/111        #radius on the x-axis
    #b=pf_data['R_MAJOR'][select][j]/111        #radius on the y-axis

    t_rot=pf_data['R_ORIENTATION'][select][j]   #rotation angle
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
    #2-D rotation matrix
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
            
    return u+Ell_rot[0,:], v+Ell_rot[1,:]
        

# Data base limits
#
xlim_min = -75; # (actually used -70 in search)
xlim_max = -50; 
ylim_min = -40; 
ylim_max = -20; 

opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
        'ylim_min': ylim_min, 'ylim_max': ylim_max}

# =============================================================================
#     'Gross' removal of clear sky overpasses inside area of interest 
#     mv HDF5 files that with PCT(89) > 200 K to another folder and deleted
# =============================================================================
# mypath    = '../GMI/'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# for i in onlyfiles:
#     if ".HDF5" in i: 
#         fname_GMI = mypath+i
#         f = h5py.File( fname_GMI, 'r')
#         tb_s1_gmi = f[u'/S1/Tc'][:,:,:]           
#         lon_gmi   = f[u'/S1/Longitude'][:,:] 
#         lat_gmi   = f[u'/S1/Latitude'][:,:]
#         f.close()
#         inside  = np.logical_and(np.logical_and(lon_gmi >= opts['xlim_min'], lon_gmi <= opts['xlim_max']), 
#                              np.logical_and(lat_gmi >= opts['ylim_min'], lat_gmi <= opts['ylim_max']))    
#         [PCT10, PCT19, PCT37, PCT89] = calc_PCTs(tb_s1_gmi)
#         if np.array( np.where(PCT89[inside]<200) ).size == 0:
#             print(fname_GMI[7:] + 'IS EMPTY AND COPY FILE TO TRASH FOLDER')
#             shutil.move(fname_GMI, mypath+'outside_PCT/'+i)
# no_Files = len([f for f in listdir(mypath+'outside_PCT/') if isfile(join(mypath+'outside_PCT/', f))])
# print('This condition leaves out '+str(no_Files)+' orbits and')
# no_Files = len([f for f in listdir(mypath) if isfile(join(mypath, f))])
# print('We have a total of '+str(no_Files)+' orbits with PCT(89) < 200 K')
# =============================================================================

#------------------------------------------------------------------------------
# Open shapefile and plot the area of interest 
# Shapefile de provincias Argentinas (IGN) para buscar datos en provincias
# de inters: Prov. Cordoba, Region litoral, Prov. Mendoza, Prov. Buenos Aires
# Chaco, Formosa, Santiago del Estero, San Luis
Prov = gpd.read_file('/Users/victoria.galligani/Work/Tools/Shapefiles/Provincia/ign_provincia.shp')
fieldName = {}
for i in range(len(Prov['FNA'])):
    fieldName[i] = Prov['FNA'][i]
    
areas_interes = ['Ciudad Autónoma de Buenos Aires', 'Provincia de Mendoza',
             'Provincia de Córdoba', 'Provincia de San Luis',
             'Provincia de Santa Fe', 'Provincia de Entre Ríos', 
             'Provincia del Chaco', 'Provincia de Formosa', 
             'Provincia de Santiago del Estero', 
             'Provincia de Buenos Aires', 'Provincia de Corrientes', 
             'Provincia de Misiones']

shp_total = Prov[ (Prov['FNA'] == areas_interes[0]) |
                  (Prov['FNA'] == areas_interes[1]) |
                  (Prov['FNA'] == areas_interes[2]) |
                  (Prov['FNA'] == areas_interes[3]) |
                  (Prov['FNA'] == areas_interes[4]) |
                  (Prov['FNA'] == areas_interes[5]) |
                  (Prov['FNA'] == areas_interes[6]) |
                  (Prov['FNA'] == areas_interes[7]) |
                  (Prov['FNA'] == areas_interes[8]) |
                  (Prov['FNA'] == areas_interes[9]) |
                  (Prov['FNA'] == areas_interes[10])|
                  (Prov['FNA'] == areas_interes[11]) ]

print('Plot union of polygons(Provincias) de interes')
polygons = shp_total['geometry']
boundary = gpd.GeoSeries(cascaded_union(polygons))
boundary.plot(color = 'red')
plt.close() 

#------------------------------------------------------------------------------
import shapefile as shp
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from shapely.geometry import MultiPoint
from osgeo import ogr

sfile   = shp.Reader("/Users/victoria.galligani/Work/Tools/Shapefiles/Provincia/ign_provincia.shp")
shp     = ogr.Open("/Users/victoria.galligani/Work/Tools/Shapefiles/Provincia/ign_provincia.shp")
layer   = shp.GetLayer()

ix      = []
iy      = []
points  = []
counter = 0

results = {'north': None}

for shape_rec in sfile.shapeRecords():    
    if any(ext in shape_rec.record[3] for ext in areas_interes):
        print(shape_rec.record[3])
        # These pts are for each province:
        pts = shape_rec.shape.points
        for u in range(len(pts)):
            ix.append(pts[u][0])
            iy.append(pts[u][1])
            # para misiones guardar punto mas norte: [esto tarda bastante ... optimizar? closes point to -25?] 
            if "Misiones" in shape_rec.record[5]: 
                if results['north'] == None or results['north'][1] < pts[u][1]:
                    results['north'] = ( pts[u][0], pts[u][1] ) 
                    points.append( (pts[u][0], pts[u][1]) ) 
        if "Misiones" not in shape_rec.record[5]:  
            # Para el resto usar el centroide? 
            feature  = layer[counter]
            geometry = feature.GetGeometryRef()   #area_shape 
            poly = geometry.GetGeometryRef(0)     #area_polygon 
            for p in range(poly.GetPointCount()):
                points.append((poly.GetX(p), poly.GetY(p)))
    counter=counter+1

#------------------------------------------------------------------------------
# Run ConvexHull
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.path import Path

convexhull = ConvexHull(points)
array_points = np.array(points)

#------------------------------------------------------------------------------
# Plot enclosing polygon
prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

fig = plt.figure(figsize=[15,11])  # previously [11,15]
for simplex in convexhull.simplices:
     plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'c')
#plt.plot(array_points[convexhull.vertices, 0], array_points[convexhull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
plt.plot(prov[:,0],prov[:,1],color='k'); 
plt.ylim([-45, -20])
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Polygon of interest')
fig.savefig('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/Plots/EnclosingPolygon.png', dpi=300,transparent=False)        
plt.close()
# =============================================================================
#  Plot all remaining overlaps. Two figures:
#    1) BTs + PCs + ACPS( w/ text w/ area_politica of ACPs)
#    2) PCT + ACPs (w/ text w/ minPCT per channel and area_politica of ACPs)
# =============================================================================
mypath    = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/GMI/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for ifile in onlyfiles:
    if ".HDF5" in ifile: 
        fname_GMI = mypath+ifile
        f = h5py.File( fname_GMI, 'r')
        tb_s1_gmi = f[u'/S1/Tc'][:,:,:]           
        lon_gmi   = f[u'/S1/Longitude'][:,:] 
        lat_gmi   = f[u'/S1/Latitude'][:,:]
        lon_s2_gmi   = f[u'/S2/Longitude'][:,:] 
        lat_s2_gmi   = f[u'/S2/Latitude'][:,:]
        tb_s2_gmi = f[u'/S2/Tc'][:,:,:]                   
        f.close()

        inside2  = np.logical_and(np.logical_and(lon_s2_gmi >= opts['xlim_min'], lon_s2_gmi <= opts['xlim_max']), 
                             np.logical_and(lat_s2_gmi >= opts['ylim_min'], lat_s2_gmi <= opts['ylim_max']))
 
        inside  = np.logical_and(np.logical_and(lon_gmi >= opts['xlim_min'], lon_gmi <= opts['xlim_max']), 
                             np.logical_and(lat_gmi >= opts['ylim_min'], lat_gmi <= opts['ylim_max']))
    
        hull_path    = Path( array_points[convexhull.vertices] )

        x_data  = lon_s2_gmi[inside2]
        y_data  = lat_s2_gmi[inside2]
        z_data  = tb_s2_gmi[inside2,:]
        datapts = np.column_stack((x_data,y_data))
        inds = hull_path.contains_points(datapts)

        #---------------------------------------------------------------------
        # 1) PLOT with clipped data inside polygon and full GMI swath to check method
        #---------------------------------------------------------------------
        #testfolder = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/Plots/IOP_PlotsClippingTest'
        #opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
        #                'ylim_min': ylim_min, 'ylim_max': ylim_max,
        #                'title': str(ifile), 'path': testfolder,
        #                'name':  str(ifile)+'_PLOT_MASKED'} 
        #PlottingGMITools.plot_polygon_GMI(z_data[inds,0], z_data[inds,0]-z_data[inds,1], 
        #                  x_data[inds],  y_data[inds], lon_s2_gmi, lat_s2_gmi, tb_s2_gmi, opts, inds)
        #---------------------------------------------------------------------
        # 2) PLOT GMI DATA WITH ACP POLYGON, CLIPPED INSIDE POLIGON OF INTEREST
        #---------------------------------------------------------------------
        testfolder = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/Plots/IOP_PlotsClippedMaps_wACPS'
        opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
                        'ylim_min': ylim_min, 'ylim_max': ylim_max,
                        'title': str(ifile), 'path': testfolder,
                        'name':  str(ifile)+'_PLOT_MASKED', 'ifile': ifile} 
        PlottingGMITools.plot_GMI(lon_gmi, lat_gmi, tb_s1_gmi, tb_s2_gmi, 
             lon_s2_gmi, lat_s2_gmi, opts, convexhull, array_points, ClippedFlag=1)
        #---------------------------------------------------------------------
        # 3) PLOT GMI POL. CORRECTED TBS with ACP POLYGON, CLIPPED INSIDE POLYGON AND 
        # ADD IN FIGURE MINPCT37 AND MINPCT89
        #---------------------------------------------------------------------
        testfolder = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/Plots/IOP_PlotsClippedMaps_wACPS_PCTs'
        opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
                        'ylim_min': ylim_min, 'ylim_max': ylim_max,
                        'title': str(ifile), 'path': testfolder,
                        'name':  str(ifile)+'_PLOT_MASKED', 'ifile': ifile} 
        [PCT10, PCT19, PCT37, PCT89] = calc_PCTs(tb_s1_gmi)
        PlottingGMITools.plot_PCT(lon_gmi, lat_gmi, PCT10, PCT19, PCT37, PCT89, opts, ifile, 1, 
                                  array_points, convexhull)

# =============================================================================
# GPM Precipitation Feature database by Chuntao Liu
# -------------------------------------------------------
# NOTE: THIS IS THE CONSTELLATION PF (no radar info!)
# from version 1 document (2017): Currently the framework of the GPM PF algorithm
# includes a) combining useful parameters from each individual Ku, Ka, DPR and 
# GMI products; b) collocating GMI high/low frequencies and Ku/Ka radar measurements 
# after the parallax correction of the different geo-location geometry for nadir 
# and conical scans; c) defining precipitation features; d) calculation of 
# parameters representing the convective and precipitation properties inside the PF
# FROM GPM: 1B.GMI TB + 2AGMIGPROF precip. + 2AKuKaDPR Z and R + 
# 2B radar GMI retrievals and latent heating 
# There is also the level 1: “1Z.GPM.DPRGMI.PF.yymmdd-starttime-endtime.orbit.version.HDF"
# that could be available for some interesting DPR parameters. Level 1 is before PF are created
# PFs are either based on the DPR swath or GMI swath
# -------------------------------------------------------
pf_path = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/GPM.PF/'
pf_data = []
 
fields = ['area37lt175', 'area37lt200', 'area37lt225', 'area37lt250', 'area37lt275', 
          'area85lt100', 'area85lt125', 'area85lt150', 'area85lt175', 'area85lt200', 
          'area85lt225', 'area85lt250', 'area85lt275', 'arearain', 'arearain_100mm', 
          'arearain_10mm', 'arearain_1mm', 'arearain_20mm', 'arearain_50mm', 
          'day', 'elev', 'hour', 'icearea', 'icevolume', 'landocean', 'lat', 'lon', 
          'maxiwp', 'maxrainrate', 'minh10', 'minh166', 'minh19', 'minh37', 'minh85', 
          'minlat', 'minlon', 'minpct10', 'minpct10lat', 'minpct10lon', 'minpct19', 
          'minpct19lat', 'minpct19lon', 'minpct37', 'minpct37lat', 'minpct37lon', 
          'minpct85', 'minpct85lat', 'minpct85lon', 'minv10', 'minv166', 'minv1833',
          'minv1837', 'minv19', 'minv22', 'minv37', 'minv85', 'month', 'npixels', 
          'orbit', 'r_flag', 'r_lat', 'r_lon', 'r_major', 'r_major_degree', 'r_minor', 
          'r_minor_degree', 'r_orientation', 'r_solid', 'volrain', 'volrain_100mm', 
          'volrain_10mm', 'volrain_1mm', 'volrain_20mm', 'volrain_50mm', 'year']

# R_LAT                       Latitude center of fitted ellipse   [degree]
# R_LON                       Longitude center of fitted ellipse  [degree]
# R_MAJOR                     Major axis length of fitted ellipse [km]
# R_MAJOR_DEGREE              Major axis length of fitted ellipse [degree]
# R_MINOR                     Minor axis length of fitted ellipse [km]
# R_MINOR_DEGREE              Minor axis length of fitted ellipse [degree]
# R_ORIENTATION               Orientation of fitted ellipse
# R_SOLID                     fraction of area filled with fitted ellipse 

# pf_data has groups for each month 
months  = ['09','10','11','12'] 
for imonth in months:
     pf_data = np.ma.append(pf_data, read_hdf5(pf_path + 'GPM.2018'+imonth+'.HDF5'))

months = ['01','02','03','04']
for imonth in months:
     pf_data = np.ma.append(pf_data, read_hdf5(pf_path + 'GPM.2019'+imonth+'.HDF5'))

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# ALSO TEST THE RADAR PF!  LEVEL 2
# Figure 2 lists current precipitation feature definitions. Kurpf, Karpf, and DPRrpf are
# defined based on the contiguous pixels with non zero precipitation rate (here > 0.1mm/hr
# is used) from Ku, Ka, and DPR retrievals. Kurppf is based on the projected area with any
# radar echo in the column. gpf is based on the GMI precipitation inside Ku swath. Rgpf and 
# kuclconf are designed to assess the performance of retrievals and the convective regions 
# for storms. Ggpf is the GMI precipitation feature in GMI swath. Gpctf is defined with area 
# of GMI 89GHz Polarization Corrected Temperature (PCT, Spence et al., 1989) colder than 250 K

# Here ggpf 
#-------------------------------------------------------------------------- GGPF 
ggpf_path = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/GGPF.PF/'
ggpf_data_main = []
 
# pf_data has groups for each month 
months  = ['09','10','11','12'] 
for imonth in months:
    # open the hdf file
    hdf  = SD.SD(ggpf_path + 'pf_2018'+imonth+'_level2.HDF')
    # Make dictionary and read the HDF file
    ggpf_data = {} 
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        ggpf_data[name] = data
    hdf.end()
    ggpf_data_main = np.ma.append(ggpf_data_main, ggpf_data)

months = ['01','02','03','04']
for imonth in months:
    # open the hdf file
    hdf  = SD.SD(ggpf_path + 'pf_2019'+imonth+'_level2.HDF')
    # Make dictionary and read the HDF file
    ggpf_data = {} 
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        ggpf_data[name] = data
    hdf.end()
    ggpf_data_main = np.ma.append(ggpf_data_main, ggpf_data)

#-------------------------------------------------------------------------- DPRRPF
DPRrpf_path = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/DPRRPF/'
DPRrpf_data_main = []
 
# pf_data has groups for each month 
months  = ['09','10','11','12'] 
for imonth in months:
    # open the hdf file
    hdf  = SD.SD(DPRrpf_path + 'pf_2018'+imonth+'_level2.HDF')
    # Make dictionary and read the HDF file
    DPRrpf_data = {} 
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        DPRrpf_data[name] = data
    hdf.end()
    DPRrpf_data_main = np.ma.append(DPRrpf_data_main, DPRrpf_data)

months = ['01','02','03','04']
for imonth in months:
    # open the hdf file
    hdf  = SD.SD(DPRrpf_path + 'pf_2019'+imonth+'_level2.HDF')
    # Make dictionary and read the HDF file
    DPRrpf_data = {} 
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        DPRrpf_data[name] = data
    hdf.end()
    DPRrpf_data_main = np.ma.append(DPRrpf_data_main, DPRrpf_data)

#-------------------------------------------------------------------------- KuRPF
Kurpf_path = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/KURPF/'
Kurpf_data_main = []
 
# pf_data has groups for each month 
months  = ['09','10','11','12'] 
for imonth in months:
    # open the hdf file
    hdf  = SD.SD(Kurpf_path + 'pf_2018'+imonth+'_level2.HDF')
    # Make dictionary and read the HDF file
    Kurpf_data = {} 
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        Kurpf_data[name] = data
    hdf.end()
    Kurpf_data_main = np.ma.append(Kurpf_data_main, Kurpf_data)

months = ['01','02','03','04']
for imonth in months:
    # open the hdf file
    hdf  = SD.SD(Kurpf_path + 'pf_2019'+imonth+'_level2.HDF')
    # Make dictionary and read the HDF file
    Kurpf_data = {} 
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        Kurpf_data[name] = data
    hdf.end()
    Kurpf_data_main = np.ma.append(Kurpf_data_main, Kurpf_data)


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------   
# TESTING ifile='1C.GPM.GMI.XCAL2016-C.20181228-S081913-E095146.027453.V05A.HDF5'
# which has many ACPs. 
example_test = 0
if example_test == 1: 
    
    import matplotlib
    import matplotlib.gridspec as gridspec


    ifile    = '1C.GPM.GMI.XCAL2016-C.20181228-S081913-E095146.027453.V05A.HDF5'
    dpr_file = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/2A.GPM.DPR.V8-20180723.20181228-S081913-E095146.027453.V06A.HDF5'
    ku_file  = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/2A.GPM.Ku.V8-20180723.20181228-S081913-E095146.027453.V06A.HDF5'
    
    f = h5py.File( dpr_file, 'r')
    # Matched Scan (MS), Normal. Scan (NS) and High Sensitivity scan (HS).
    DPR_NS_LON = f[u'/MS/Longitude'][:,:]          
    DPR_NS_LAT = f[u'/MS/Latitude'][:,:]         
    DPR_z_corr = f[u'/MS/SLV/zFactorCorrected'][:,:,:]  
    DPR_NearSurfPrecip = f[u'/MS/SLV/precipRateNearSurface'][:,:]  
    f.close()

    f = h5py.File( ku_file, 'r')
    # Normal. Scan (NS)
    ku_NS_LON = f[u'/NS/Longitude'][:,:]          
    ku_NS_LAT = f[u'/NS/Latitude'][:,:]         
    ku_z_corr = f[u'/NS/SLV/zFactorCorrected'][:,:,:]  
    ku_NearSurfPrecip = f[u'/NS/SLV/precipRateNearSurface'][:,:]  
    f.close()
    
    fname_GMI = mypath+ifile
    f = h5py.File( fname_GMI, 'r')
    tb_s1_gmi = f[u'/S1/Tc'][:,:,:]           
    lon_gmi   = f[u'/S1/Longitude'][:,:] 
    lat_gmi   = f[u'/S1/Latitude'][:,:]
    lon_s2_gmi   = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi   = f[u'/S2/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tc'][:,:,:]                   
    f.close()

    inside2  = np.logical_and(np.logical_and(lon_s2_gmi >= opts['xlim_min'], lon_s2_gmi <= opts['xlim_max']), 
                         np.logical_and(lat_s2_gmi >= opts['ylim_min'], lat_s2_gmi <= opts['ylim_max']))
 
    inside  = np.logical_and(np.logical_and(lon_gmi >= opts['xlim_min'], lon_gmi <= opts['xlim_max']), 
                         np.logical_and(lat_gmi >= opts['ylim_min'], lat_gmi <= opts['ylim_max']))

    inside_dpr  = np.logical_and(np.logical_and(DPR_NS_LON >= opts['xlim_min'], DPR_NS_LON <= opts['xlim_max']), 
                         np.logical_and(DPR_NS_LAT >= opts['ylim_min'], DPR_NS_LAT <= opts['ylim_max']))

    inside_ku  = np.logical_and(np.logical_and(ku_NS_LON >= opts['xlim_min'], ku_NS_LON <= opts['xlim_max']), 
                         np.logical_and(ku_NS_LAT >= opts['ylim_min'], ku_NS_LAT <= opts['ylim_max']))
        
    
    
    yoi = int(ifile[22:26])
    moi = "%02d" % int(ifile[26:28])
    doi = "%02d" % int(ifile[28:30])
    orbit =  int(ifile[47:53])
    print('month of interest: '+ str(moi) +' and day of interest: '+str(doi) )
    # open the correct monthly file: 
    pf_data = read_hdf5(pf_path + 'GPM.'+str(yoi)+str(moi)+'.HDF5')
    # now select day and hour of interest:  
    # note that hour is UTC hour. 
    min_hour    = int(ifile[32:34])
    min_minutes = int(ifile[34:36])
    max_hour    = int(ifile[40:42])
    max_minutes = int(ifile[42:44])
    # turn into fraction of UTC hour
    hoi_window = [ min_hour+(min_minutes/60),  max_hour+(max_minutes/60) ]
    # Use both hour and geographical region and orbit_number to filter the right data 
    select  = np.logical_and( np.logical_and( np.logical_and(np.logical_and(pf_data['lon'] >= opts['xlim_min'], pf_data['lon'] <= opts['xlim_max']), 
                              np.logical_and(pf_data['lat'] >= opts['ylim_min'], pf_data['lat'] <= opts['ylim_max']), 
                              pf_data['day'] == int(ifile[28:30])), 
                              np.logical_and(pf_data['hour'] >= hoi_window[0], pf_data['hour'] <= hoi_window[1])), 
                              pf_data['orbit'] == orbit  )   
    # test for DPRrpf 
    DPRrpf_data  = read_hdf(DPRrpf_path + 'pf_'+str(yoi)+str(moi)+'_level2.HDF')    
    selectDPRrpf = np.logical_and( np.logical_and( np.logical_and(np.logical_and(DPRrpf_data['LON'] >= opts['xlim_min'], 
                                                                                 DPRrpf_data['LON'] <= opts['xlim_max']), 
                              np.logical_and(DPRrpf_data['LAT'] >= opts['ylim_min'], DPRrpf_data['LAT'] <= opts['ylim_max']), 
                              DPRrpf_data['DAY'] == int(ifile[28:30])), 
                              np.logical_and(DPRrpf_data['HOUR'] >= hoi_window[0], DPRrpf_data['HOUR'] <= hoi_window[1])), 
                              DPRrpf_data['ORBIT'] == orbit  )   

    # test for Kurpf 
    Kurpf_data  = read_hdf(Kurpf_path + 'pf_'+str(yoi)+str(moi)+'_level2.HDF')    
    selectKurpf = np.logical_and( np.logical_and( np.logical_and(np.logical_and(Kurpf_data['LON'] >= opts['xlim_min'], 
                                                                                 Kurpf_data['LON'] <= opts['xlim_max']), 
                              np.logical_and(Kurpf_data['LAT'] >= opts['ylim_min'], Kurpf_data['LAT'] <= opts['ylim_max']), 
                              Kurpf_data['DAY'] == int(ifile[28:30])), 
                              np.logical_and(Kurpf_data['HOUR'] >= hoi_window[0], Kurpf_data['HOUR'] <= hoi_window[1])), 
                              Kurpf_data['ORBIT'] == orbit  )   

    
    #--------------------------------------------------------------------------
    #------------------- FIGURE TESTING 
    # r_lon and r_lat: Longitude/Latitude center of fitted ellipse
    fig = plt.figure(figsize=(12,6))     
    gs1 = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs1[0,0])
    im = plt.scatter(lon_gmi[inside], lat_gmi[inside], 
           c=tb_s1_gmi[inside,7], s=10, cmap=PlottingGMITools.cmaps['turbo_r'])  
    #plt.scatter(pf_data['r_lon'][select], pf_data['r_lat'][select], s=40, facecolors='none', edgecolors='k')
    ax1.set_xlim([-70,-55])
    ax1.set_ylim([-35,-20])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.colorbar(im)
    plt.title('TBV(89 GHz)')
    elems     = np.where(pf_data['r_lon'][select]!=0)
    elems_gmi = np.where(pf_data['r_lon'][select]!=0)

    #convert = []
    counter = 0
    for j in elems[0]:
        counter=counter+1
        #convert[j] = pf_data['r_major_degree'][select][j]/pf_data['r_major'][select][j]
        u, v = get_ellipse_info(pf_data, select, j)
        plt.plot( u , v, 'darkorange' )    #rotated ellipse
        plt.grid(color='lightgray',linestyle='--') 
        t = plt.text(pf_data['r_lon'][select][j], pf_data['r_lat'][select][j]+1, str(counter), fontsize=10)
        t.set_bbox(dict(facecolor='orange', alpha=0.5, edgecolor='orange'))

    # ax1.set_xlim([-70,-57])
    # ax1.set_ylim([-30,-24])    
    elems = np.where(DPRrpf_data['R_LON'][selectDPRrpf]!=0)   # and use pf_data['r_lon'][select][j]    
    for j in elems[0]:
        u, v = get_ellipse_info_MAJ_km(DPRrpf_data, selectDPRrpf, j)
        plt.plot( u , v, 'red' )    #rotated ellipse
    # Add radar swath! 
    plt.plot( DPR_NS_LON[:,0] , DPR_NS_LAT[:,0], '--k' )    #rotated ellipse
    plt.plot( DPR_NS_LON[:,-1] , DPR_NS_LAT[:,-1], '--k' )    #rotated ellipse
    
    plot_pd = 0
    if plot_pd == 1: 
        # PD
        ax1 = plt.subplot(gs1[0,1])
        im = plt.scatter(lon_gmi[inside], lat_gmi[inside], 
               c=tb_s1_gmi[inside,7]-tb_s1_gmi[inside,8], s=10, vmin=0, vmax=16, cmap=PlottingGMITools.discrete_cmap(16,  'rainbow'))  
        plt.scatter(pf_data['r_lon'][select], pf_data['r_lat'][select], s=40, facecolors='none', edgecolors='k')
        ax1.set_xlim([-70,-55])
        ax1.set_ylim([-35,-20])
        plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
        plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
        plt.colorbar(im)
        plt.title('PD(89 GHz)')  
        elems = np.where(pf_data['r_lon'][select]!=0)   # and use pf_data['r_lon'][select][j]    
        for j in elems[0]:
            u, v = get_ellipse_info(pf_data, select, j)
            plt.plot( u , v, 'darkorange' )    #rotated ellipse
            plt.grid(color='lightgray',linestyle='--')    
        elems = np.where(DPRrpf_data['R_LON'][selectDPRrpf]!=0)   # and use pf_data['r_lon'][select][j]    
        for j in elems[0]:
            u, v = get_ellipse_info_MAJ_km(DPRrpf_data, selectDPRrpf, j)
            plt.plot( u , v, 'red' )    #rotated ellipse
        # Add radar swath! 
        plt.plot( DPR_NS_LON[:,0] , DPR_NS_LAT[:,0], '--k' )    #rotated ellipse
        plt.plot( DPR_NS_LON[:,-1] , DPR_NS_LAT[:,-1], '--k' )    #rotated ellipse
        
    
    plot_dpr = 1
    if plot_dpr == 1:     
        # max(Zh_DPR)
        ax1 = plt.subplot(gs1[0,1])
        im = plt.scatter(DPR_NS_LON[inside_dpr], DPR_NS_LAT[inside_dpr], 
               c=np.max(DPR_z_corr[inside_dpr,:],1), vmin=0, vmax=40, s=10)   
        # Add radar swath! 
        plt.plot( DPR_NS_LON[:,0] , DPR_NS_LAT[:,0], '--k' )    #rotated ellipse
        plt.plot( DPR_NS_LON[:,-1] , DPR_NS_LAT[:,-1], '--k' )    #rotated ellipse    
        ax1.set_xlim([-70,-55])
        ax1.set_ylim([-35,-20])
        plt.colorbar(im)
        plt.title('DPR_Zcorr')      
        plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
        plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);           
        elems = np.where(pf_data['r_lon'][select]!=0)   # and use pf_data['r_lon'][select][j]    
        for j in elems[0]:
            u, v = get_ellipse_info(pf_data, select, j)
            plt.plot( u , v, 'darkorange' )    #rotated ellipse
            plt.grid(color='lightgray',linestyle='--')    
        elems = np.where(DPRrpf_data['R_LON'][selectDPRrpf]!=0)   # and use pf_data['r_lon'][select][j]    
        elems_dpr = np.where(DPRrpf_data['R_LON'][selectDPRrpf]!=0)   # and use pf_data['r_lon'][select][j]    
        for j in elems[0]:
            u, v = get_ellipse_info_MAJ_km(DPRrpf_data, selectDPRrpf, j)
            plt.plot( u , v, 'red' )    #rotated ellipse
        # Add radar swath! 
        plt.plot( DPR_NS_LON[:,0] , DPR_NS_LAT[:,0], '--k' )    #rotated ellipse
        plt.plot( DPR_NS_LON[:,-1] , DPR_NS_LAT[:,-1], '--k' )    #rotated ellipse
        
    plt.figtext(0.70, 1, "ellipse: GMI-PF feature", fontsize='large', color='darkorange', ha ='right')
    plt.figtext(0.90, 1, "ellipse: DPR-PF feature", fontsize='large', color='red', ha ='right')
    
    plt.suptitle('orbit file: '+str(ifile),y=0.96)
    
    #--------------------------------------------------------------------------
    #------------------- FIGURE w/ precip to evalute mismatched ellipse? 
    fig = plt.figure(figsize=(12,6))     
    gs1 = gridspec.GridSpec(1, 3)
    ax1 = plt.subplot(gs1[0,0])
    im = plt.scatter(lon_gmi[inside], lat_gmi[inside], 
           c=tb_s1_gmi[inside,7], s=100, cmap=PlottingGMITools.cmaps['turbo_r'])  
    #plt.scatter(pf_data['r_lon'][select], pf_data['r_lat'][select], s=40, facecolors='none', edgecolors='k')
    ax1.set_xlim([-68,-64])
    ax1.set_ylim([-35,-33])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.colorbar(im)
    plt.title('TBV(89 GHz)')
    elems     = np.where(pf_data['r_lon'][select]!=0)
    elems_gmi = np.where(pf_data['r_lon'][select]!=0)

    #convert = []
    counter = 0
    for j in elems[0]:
        counter=counter+1
        #convert[j] = pf_data['r_major_degree'][select][j]/pf_data['r_major'][select][j]
        u, v = get_ellipse_info(pf_data, select, j)
        plt.plot( u , v, 'darkorange' )    #rotated ellipse
        plt.grid(color='lightgray',linestyle='--') 

    # ax1.set_xlim([-70,-57])
    # ax1.set_ylim([-30,-24])    
    elems = np.where(DPRrpf_data['R_LON'][selectDPRrpf]!=0)   # and use pf_data['r_lon'][select][j]    
    for j in elems[0]:
        u, v = get_ellipse_info_MAJ_km(DPRrpf_data, selectDPRrpf, j)
        plt.plot( u , v, 'red' )    #rotated ellipse
    # Add radar swath! 
    plt.plot( DPR_NS_LON[:,0] , DPR_NS_LAT[:,0], '--k' )    #rotated ellipse
    plt.plot( DPR_NS_LON[:,-1] , DPR_NS_LAT[:,-1], '--k' )    #rotated ellipse
    ax1.set_yticklabels([])

    elems = np.where(Kurpf_data['R_LON'][selectKurpf]!=0)   # and use pf_data['r_lon'][select][j]    
    for j in elems[0]:
        u, v = get_ellipse_info_MAJ_km(Kurpf_data, selectKurpf, j)
        plt.plot( u , v, 'magenta' )    #rotated ellipse
        
    # max(Zh_DPR)
    ax1 = plt.subplot(gs1[0,1])
    DPR_NearSurfPrecip[np.where(DPR_NearSurfPrecip<0.1)]=np.nan
    im = plt.scatter(DPR_NS_LON[inside_dpr], DPR_NS_LAT[inside_dpr], 
           c=DPR_NearSurfPrecip[inside_dpr],norm=matplotlib.colors.LogNorm(), vmin=0.1, vmax=100, s=50)   
    # Add radar swath! 
    plt.plot( DPR_NS_LON[:,0] , DPR_NS_LAT[:,0], '--k' )    #rotated ellipse
    plt.plot( DPR_NS_LON[:,-1] , DPR_NS_LAT[:,-1], '--k' )    #rotated ellipse    
    ax1.set_xlim([-68,-64])
    ax1.set_ylim([-35,-33])
    plt.colorbar(im)
    plt.title('DPR_NearSurfPrecip(MS)')      
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);           
    elems = np.where(pf_data['r_lon'][select]!=0)   # and use pf_data['r_lon'][select][j]    
    for j in elems[0]:
        u, v = get_ellipse_info(pf_data, select, j)
        plt.plot( u , v, 'darkorange' )    #rotated ellipse
        plt.grid(color='lightgray',linestyle='--')    
    elems = np.where(DPRrpf_data['R_LON'][selectDPRrpf]!=0)   # and use pf_data['r_lon'][select][j]    
    elems_dpr = np.where(DPRrpf_data['R_LON'][selectDPRrpf]!=0)   # and use pf_data['r_lon'][select][j]    
    for j in elems[0]:
        u, v = get_ellipse_info_MAJ_km(DPRrpf_data, selectDPRrpf, j)
        plt.plot( u , v, 'red' )    #rotated ellipse
    # Add radar swath! 
    plt.plot( DPR_NS_LON[:,0] , DPR_NS_LAT[:,0], '--k' )    #rotated ellipse
    plt.plot( DPR_NS_LON[:,-1] , DPR_NS_LAT[:,-1], '--k' )    #rotated ellipse

    # max(Zh_DPR)
    ax1 = plt.subplot(gs1[0,2])
    ku_NearSurfPrecip[np.where(ku_NearSurfPrecip<0.1)]=np.nan
    im = plt.scatter(ku_NS_LON[inside_ku], ku_NS_LAT[inside_ku], 
           c=ku_NearSurfPrecip[inside_ku],norm=matplotlib.colors.LogNorm(), vmin=0.1, vmax=100, s=50)   
    # Add radar swath! 
    plt.plot( ku_NS_LON[:,0] , ku_NS_LAT[:,0], '--k' )    #rotated ellipse
    plt.plot( ku_NS_LON[:,-1] , ku_NS_LAT[:,-1], '--k' )    #rotated ellipse    
    ax1.set_xlim([-68,-64])
    ax1.set_ylim([-35,-33])
    plt.colorbar(im)
    plt.title('Ku_NearSurfPrecip(NS)')      
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);           
    elems = np.where(pf_data['r_lon'][select]!=0)   # and use pf_data['r_lon'][select][j]    
    for j in elems[0]:
        u, v = get_ellipse_info(pf_data, select, j)
        plt.plot( u , v, 'darkorange' )    #rotated ellipse
        plt.grid(color='lightgray',linestyle='--')    
    elems = np.where(Kurpf_data['R_LON'][selectKurpf]!=0)   # and use pf_data['r_lon'][select][j]    
    for j in elems[0]:
        u, v = get_ellipse_info_MAJ_km(Kurpf_data, selectKurpf, j)
        plt.plot( u , v, 'magenta' )    #rotated ellipse
    # Add radar swath! 
    plt.plot( ku_NS_LON[:,0] , ku_NS_LAT[:,0], '--k' )    #rotated ellipse
    plt.plot( ku_NS_LON[:,-1] , ku_NS_LAT[:,-1], '--k' )    #rotated ellipse
        
    plt.figtext(0.50, 1, "ellipse: GMI-PF feature", fontsize='large', color='darkorange', ha ='right')
    plt.figtext(0.70, 1, "ellipse: DPR-PF feature", fontsize='large', color='red', ha ='right')
    plt.figtext(0.90, 1, "ellipse: Ku-PF feature", fontsize='large', color='magenta', ha ='right')
    ax1.set_yticklabels([])
    
    plt.suptitle('orbit file: '+str(ifile),y=0.96)
    










     
    # BUT BEFORE PLOTTING W/ THE ELLIPSES? WHY NOT FILTER OUT DATA FIRST?
    # BECAUSE WE HAVE 401 FILES SO FAR. LOOK AT CERTAIN CHARACTERISTICS INSIDE POLYGONS! 
    # Use proxies for convection

# =============================================================================
# #  mv HDF5 files outside main polygons of interest: Mendoza, Cordoba, Litoral, 
# #  but also keep Prov. Buenos Aires, Santa Fe, etc. 
# 
# #------------------------------------------------------------------------------
# # So now Re-run the PCT condition inside polygon: TO DO SO I NEED TO RE-DO 
# # THE POLYGON STUFF FOR S1 AND CALCULATE THE PCT89. 
# #------------------------------------------------------------------------------
# from ast import literal_eval
# 




# APPENDIX 
# -----------------------------------------------------------------------------
# Plot all IOP orbits and save
# -----------------------------------------------------------------------------
# testfolder = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/V1/IOP_Plots'
# mypath    = '../GMI/'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# for i in onlyfiles:
#     if ".HDF5" in i:
#         if i[22:26] == '2018':
#             if i[26:28] == '10' or i[26:28] == '11' or i[26:28] == '12':   
#                 fname_GMI = mypath+i
#                 f = h5py.File( fname_GMI, 'r')
#                 tb_s1_gmi = f[u'/S1/Tc'][:,:,:]           
#                 lon_gmi   = f[u'/S1/Longitude'][:,:] 
#                 lat_gmi   = f[u'/S1/Latitude'][:,:]
#                 lon_s2_gmi   = f[u'/S2/Longitude'][:,:] 
#                 lat_s2_gmi   = f[u'/S2/Latitude'][:,:]
#                 tb_s1_gmi = f[u'/S1/Tc'][:,:,:]           
#                 tb_s2_gmi = f[u'/S2/Tc'][:,:,:]                          
#                 f.close()
# 
#                 opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
#                         'ylim_min': ylim_min, 'ylim_max': ylim_max,
#                         'title': str(i), 'path': testfolder,
#                         'name':  str(i)+'_PLOT'} 
#                 PlottingGMITools.plot_GMI(lon_gmi, lat_gmi, tb_s1_gmi, tb_s2_gmi, 
#                           lon_s2_gmi, lat_s2_gmi, opts)
# 
#                 [PCT10, PCT19, PCT37, PCT89] = calc_PCTs(tb_s1_gmi)
#                 opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
#                         'ylim_min': ylim_min, 'ylim_max': ylim_max,
#                         'title': str(i), 'path': testfolder,
#                         'name':  str(i)+'_PCT_PLOT'} 
#                 PlottingGMITools.plot_PCT(lon_gmi, lat_gmi, PCT10, PCT19, PCT37, PCT89, opts)
#                 opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
#                         'ylim_min': ylim_min, 'ylim_max': ylim_max,
#                         'title': str(i), 'path': testfolder,
#                         'name':  str(i)+'_PLOT'} 
#                 PlottingGMITools.plot_GMI(lon_gmi, lat_gmi, tb_s1_gmi, tb_s2_gmi, 
#                           lon_s2_gmi, lat_s2_gmi, opts)
# 
#                 [PCT10, PCT19, PCT37, PCT89] = calc_PCTs(tb_s1_gmi)
#                 opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
#                         'ylim_min': ylim_min, 'ylim_max': ylim_max,
#                         'title': str(i), 'path': testfolder,
#                         'name':  str(i)+'_PCT_PLOT'} 
#                 PlottingGMITools.plot_PCT(lon_gmi, lat_gmi, PCT10, PCT19, PCT37, PCT89, opts)
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------
Created on Thu Feb  4 09:51:40 2021
-----------------------------------------------------------------
@purpose : Plotting tools
@author  : victoria.galligani
-----------------------------------------------------------------
@TODOS(?):
 ----------------------------------------------------------------- 
"""
#################################################################
# Load libraries
#################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import geopandas 
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
from ast import literal_eval
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib import patches
import re
#################################################################
# Shapefiles for cartopy 
geo_reg_shp = '/Users/victoria.galligani/Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
geo_reg = shpreader.Reader(geo_reg_shp)

countries = shpreader.Reader('/Users/victoria.galligani/Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')


states_provinces = cfeature.NaturalEarthFeature(
        category='cultural', 
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',
        edgecolor='black')

# countries = cfeature.NaturalEarthFeature(
#          category='cultural',
#          name='admin_0_countries',
#          scale='110m',
#          facecolor='none')

rivers = cfeature.NaturalEarthFeature(
        category='physical',
        name='rivers_lake_centerlines',
        scale='10m',
        facecolor='none',
        edgecolor='black')

# read the shapefile using geopandas

#################################################################
# GMI colormap 
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
    
#################################################################
# START FUNCTIONS 
#################################################################

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

#################################################################

def apply_geofence_on_data(data, data_lat, data_lon, min_latitude, max_latitude, min_longitude,
                               max_longitude):

    data[data_lat < min_latitude]  = np.nan
    data[data_lat > max_latitude]  = np.nan
    data[data_lon > max_longitude] = np.nan
    data[data_lon < min_longitude] = np.nan

    return data

#################################################################
def read_acps(yoi, moi, doi):
    
    """ Returns all the polygons available for a specific date """

    folder = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/OtherData/'
    file = 'ACPs.xlsx' 

    data_ACPS = pd.ExcelFile(folder+file)
    df1       = data_ACPS.parse(data_ACPS.sheet_names[0])

    #- Now, extract only poligons of interest for a specific date 
    month_mask = df1['FECHA AVISO'].map(lambda x: x.month) == moi
    df1 = df1[month_mask]
    
    day_mask = df1['FECHA AVISO'].map(lambda x: x.day) == doi
    df1 = df1[day_mask]
    
    year_mask = df1['FECHA AVISO'].map(lambda x: x.year) == yoi
    df1 = df1[year_mask]

    numero         = df1['Numero ACP']
    fenomeno       = df1['FENOMENO']
    area_politica  = df1['AREA']
    poligono       = df1[['POLIGONO']].to_numpy()    # pandas.core.series.Series
    fecha          = df1['FECHA AVISO']              # en formato AAAA-MM-DD XX:XX:XX 
    
    # Get all the coordinates for each polygon 
    pol_n = []
    for i in range(poligono.shape[0]):
        pol_n.append( list(literal_eval(poligono[i][0])) )
                
    return pol_n 

#################################################################
def read_acps_FULL(yoi, moi, doi):
    
    """ Returns all the polygons available for a specific date """

    folder = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/OtherData/'
    file = 'ACP_2014_082021-con-area.xlsx' 

    data_ACPS = pd.ExcelFile(folder+file)
    df1       = data_ACPS.parse(data_ACPS.sheet_names[0])

    #- Now, extract only poligons of interest for a specific date 
    month_mask = df1['Fecha y Hora ACP'].map(lambda x: x.month) == moi
    df1 = df1[month_mask]
    
    day_mask = df1['Fecha y Hora ACP'].map(lambda x: x.day) == doi
    df1 = df1[day_mask]
    
    year_mask = df1['Fecha y Hora ACP'].map(lambda x: x.year) == yoi
    df1 = df1[year_mask]


    cols = df1.select_dtypes(include=[np.object]).columns
    df1[cols] = df1[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))


    numero         = df1['Número de Aviso'].to_numpy()
    fenomeno       = df1['Fenómeno'].to_numpy()
    area_politica  = df1[' Area del ACP - provincias, municipio/departamentos '].to_numpy()
    poligono       = df1[['Area ACP - coordenadas']].to_numpy()    # pandas.core.series.Series
    fecha          = df1['Fecha y Hora ACP'].to_numpy()              # en formato AAAA-MM-DD XX:XX:XX 
    
    # Get all the coordinates for each polygon 
    pol_n = []
    area_politica_ = []
    for i in range(poligono.shape[0]):
        try:
            pol_n.append( list(literal_eval(poligono[i][0])) )
            text_area_politica = area_politica[i]
            text_area_politica = text_area_politica.replace('ENTRE RAOS', 'ENTRE RIOS')
            text_area_politica = text_area_politica.replace('CA3RDOBA', 'CORDOBA')
            #print(text_area_politica)
            #--------        
            # Replace <b> </b> w/ latex formating
            #text_area_politica = text_area_politica.replace('<b>', '\textbf{')
            #text_area_politica = text_area_politica.replace('</b>', '}')
            # Remove <br /><br /> 
            #text_area_politica = text_area_politica.replace('<br /><br />', ' ')
            #--------
            # Keep provincias only
            elem1 = [match.start() for match in re.finditer('<b>', text_area_politica)]
            elem2 = [match.start() for match in re.finditer(':</b>', text_area_politica)]
            for i in range(len(elem1)):
                area_politica_.append( text_area_politica[elem1[i]+3:elem2[i]] )
        except (SyntaxError, IndexError) as E:  # specific exceptions
            print('SyntaxError for doi/moi/yoi: ', str(doi)+'_'+str(moi)+'-'+str(yoi))
            continue
        #print(area_politica_)


        
    return pol_n, area_politica_
#################################################################

def plot_GMI(lon_gmi, lat_gmi, tb_s1_gmi, tb_s2_gmi, 
             lon_s2_gmi, lat_s2_gmi, options, convexhull, array_points, ClippedFlag):
    
    """Create a 2x3 GMI colormap with BT(37, 89, 166) and PD(37, 89, 166)
        w. BT contours on the PD maps. Added polygon of interest. Option to keep only
        the clipped area. Also include ACP alertsa del SMN. 
        Include PF ellipse from database"""

    ifile = options['ifile']
    yoi = int(ifile[22:26])
    moi = int(ifile[26:28]) #"%02d" % int(ifile[26:28])
    doi = int(ifile[28:30]) #"%02d" % int(ifile[28:30])
    print('day of interest: '+str(doi) )
    pol_n, area_politica = read_acps_FULL(yoi, moi, doi) 

    # fig = plt.figure(figsize=[15,11])  # previously [11,15]
    # for i in range(len(pol_n)):
    #     coord = pol_n[i]
    #     coord.append(coord[0]) #repeat the first point to create a 'closed loop'
    #     ys, xs = zip(*coord) #create lists of x and y values
    #     plt.plot(xs,ys) 
        
    #
    # Get footprints inside lat/lon region of interest 
    #
    inside  = np.logical_and(np.logical_and(lon_gmi >= options['xlim_min'], lon_gmi <= options['xlim_max']), 
                             np.logical_and(lat_gmi >= options['ylim_min'], lat_gmi <= options['ylim_max']))

    inside2 = np.logical_and(np.logical_and(lon_s2_gmi >= options['xlim_min'], lon_s2_gmi <= options['xlim_max']), 
                             np.logical_and(lat_s2_gmi >= options['ylim_min'], lat_s2_gmi <= options['ylim_max']))

         
    data_tb37 = apply_geofence_on_data (tb_s1_gmi[:,:,5], lat_gmi, lon_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  
    data_tb89 = apply_geofence_on_data (tb_s1_gmi[:,:,7], lat_gmi, lon_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  
    data_tb166 = apply_geofence_on_data (tb_s2_gmi[:,:,0], lat_s2_gmi, lon_s2_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  

    lon  = lon_gmi.view() 
    lat  = lat_gmi.view()
    lon2 = lon_s2_gmi.view() 
    lat2 = lat_s2_gmi.view() 


    lon_gmi    = lon_gmi[inside] 
    lat_gmi    = lat_gmi[inside]
    tb_s1_gmi  = tb_s1_gmi[inside,:]
    tb_s2_gmi  = tb_s2_gmi[inside2,:]
    lon_s2_gmi = lon_s2_gmi[inside2] 
    lat_s2_gmi = lat_s2_gmi[inside2]    

    if ClippedFlag == 1:

        hull_path    = Path( array_points[convexhull.vertices] )

        datapts = np.column_stack((lon_gmi,lat_gmi))
        inds = hull_path.contains_points(datapts)
        lon_gmi    = lon_gmi[inds] 
        lat_gmi    = lat_gmi[inds]
        tb_s1_gmi  = tb_s1_gmi[inds,:]
        
        datapts = np.column_stack((lon_s2_gmi,lat_s2_gmi))
        inds = hull_path.contains_points(datapts)
        tb_s2_gmi  = tb_s2_gmi[inds,:]
        lon_s2_gmi = lon_s2_gmi[inds] 
        lat_s2_gmi = lat_s2_gmi[inds] 
    
                              
    plt.matplotlib.rc('font', family='DejaVu Sans', size = 12)

    fontsize   = 12
    linewidths = 1
    
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
    im = plt.scatter(lon_gmi, lat_gmi, 
           c=tb_s1_gmi[:,5], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
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
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')
    # For each polygon detected plot a polygon
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m') 

    
    
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
    im = plt.scatter(lon_gmi[:], lat_gmi[:], 
           c=tb_s1_gmi[:,7], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
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
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m') 
        
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
    im = plt.scatter(lon_s2_gmi[:], lat_s2_gmi[:], 
           c=tb_s2_gmi[:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('BT 166 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')
        
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m') 
        
    ax1.text(0.05,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    #ax1.text(-0.1,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')

    # p2 = ax1.get_position().get_points().flatten()
    # #ax_cbar = fig.add_axes([p1[0], 0.45, p2[2]-p1[0], 0.05])
    # ax_cbar = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar = fig.colorbar(im, shrink=1,ticks=np.arange(50,325,25), 
                        extend='both', orientation="vertical")
    cbar.set_label('BT (K)', fontsize=fontsize)
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m')        
 
        
    # PD(37)
    ax1 = plt.subplot(gs1[1,0], projection=ccrs.PlateCarree())
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
           c=tb_s1_gmi[:,5]-tb_s1_gmi[:,6], s=10, vmin=0, vmax=16, cmap=discrete_cmap(16,  'rainbow'))  
    plt.title('GMI PD 37 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    CS = plt.contour(lon, lat, 
                        data_tb37,[250], colors=('r'), linewidths=(linewidths));
    labels = ["250K"]
    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])
    #ax1.legend(loc='upper left', fontsize=fontsize)
    ax1.set_xlabel('Latitude', fontsize=fontsize)
    ax1.set_ylabel('Longitude', fontsize=fontsize)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(d)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    p0 = ax1.get_position().get_points().flatten()
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')    
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m') 

    # PD(89)   
    ax1 = plt.subplot(gs1[1,1], projection=ccrs.PlateCarree())
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
           c=tb_s1_gmi[:,7]-tb_s1_gmi[:,8], s=10, vmin=0, vmax=16, cmap=discrete_cmap(16,  'rainbow'))  
    plt.title('PD 89 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    CS = plt.contour(lon, lat, 
                        data_tb89,[180,250], colors=('k','r'), linewidths=(linewidths));
    
    labels = ["180K","250K"]
    if len(CS.collections) == 1:
        for i in range(len(labels)-1):
            CS.collections[i].set_label("250K")        
    else:
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
            
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(e)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    p1 = ax1.get_position().get_points().flatten()
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m') 

        
    # PD(166)       
    ax1 = plt.subplot(gs1[1,2], projection=ccrs.PlateCarree())
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
    im = plt.scatter(lon_s2_gmi[:], lat_s2_gmi[:], 
           c=tb_s2_gmi[:,0]-tb_s2_gmi[:,1], s=10, vmin=0, vmax=12, cmap=discrete_cmap(16,  'rainbow'))  
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('PD 166 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    CS = plt.contour(lon2, lat2, 
                        data_tb166,[180,250], colors=('k','r'), linewidths=(linewidths));
    labels = ["180K","250K"]
    if len(CS.collections) == 1:
        for i in range(len(labels)-1):
            CS.collections[i].set_label("250K")        
    else:
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
    ax1.legend(loc='lower right', fontsize=10)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(f)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')

    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m') 
        
        
    # Now add "area_politica" text: text(x,y, ...)
    text_ACPs = []
    for i in range(len(pol_n)):
        #print('area_politica: ',  area_politica[i])
        text_ACPs.append( area_politica[i] )
    if len(text_ACPs)>0:
        text_ACPs[0] = 'ACPs durante el dia en: '+text_ACPs[0]


    t=plt.figtext(0.1,0.05,text_ACPs,fontsize=9,va='top',ha='left',color='darkred', wrap=True)
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    width = bb.width
    print(width) #figsize(12,6)               
        
    # p2 = ax1.get_position().get_points().flatten()
    # #ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.05])
    # ax_cbar = fig.add_axes([0.92, 0.13, 0.02, 0.35])
    cbar = fig.colorbar(im, shrink=1,ticks=np.arange(0,17,1), extend='both', 
                        orientation="vertical")
    cbar.set_label('PD (K)', fontsize=fontsize)
    
    fig.suptitle(options['title'] ,fontweight='bold' )
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.899)
    #plt.savefig(options['path']+'/'+options['name']+'.eps')
    plt.savefig(options['path']+'/'+options['name']+'_ALLGMICHANNELS.png')
    plt.close()
    
    return 
  
#################################################################





#################################################################

def plot_PCT(lon_gmi, lat_gmi, PCT10, PCT19, PCT37, PCT89, options, ifile, ClippedFlag, array_points, convexhull):      

    """Create a 4x1 GMI-PCT colormap with (10, 19, 37, 89)
        w. BT contours on the PD maps. Added polygon of interest (ONLY CLIPPED ARA)
        Included as text below each subplot is the minPCT.
        Also include ACP alertsa del SMN. 
        and in text what areas are included in the ACPs"""
        
    ifile = options['ifile']
    yoi = int(ifile[22:26])
    moi =  int(ifile[26:28]) # "%02d" % int(ifile[26:28])
    doi = int(ifile[28:30])  # "%02d" % int(ifile[28:30])
    print('day of interest: '+str(doi) )
    pol_n, area_politica = read_acps_FULL(yoi, moi, doi) 

    inside  = np.logical_and(np.logical_and(lon_gmi >= options['xlim_min'], lon_gmi <= options['xlim_max']), 
                             np.logical_and(lat_gmi >= options['ylim_min'], lat_gmi <= options['ylim_max']))

    data_pd10 = apply_geofence_on_data (PCT10, lat_gmi, lon_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  
    data_pd19 = apply_geofence_on_data (PCT19, lat_gmi, lon_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                             options['xlim_max'])     
    data_pd37 = apply_geofence_on_data (PCT37, lat_gmi, lon_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  
    data_pd89 = apply_geofence_on_data (PCT89, lat_gmi, lon_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  
    lon  = lon_gmi.view() 
    lat  = lat_gmi.view()
    
    lon_gmi = lon_gmi[inside] 
    lat_gmi = lat_gmi[inside]
    PCT10   = PCT10[inside]
    PCT19   = PCT19[inside]
    PCT37   = PCT37[inside] 
    PCT89   = PCT89[inside]    

    if ClippedFlag == 1:

        hull_path    = Path( array_points[convexhull.vertices] )

        datapts = np.column_stack((lon_gmi,lat_gmi))
        inds    = hull_path.contains_points(datapts)
        lon_gmi = lon_gmi[inds] 
        lat_gmi = lat_gmi[inds]
        PCT10   = PCT10[inds]
        PCT19   = PCT19[inds]
        PCT37   = PCT37[inds] 
        PCT89   = PCT89[inds]   

# ---------------------------- 
    plt.matplotlib.rc('font', family='DejaVu Sans', size = 12)

    fontsize   = 12
    linewidths = 2
    
    fig = plt.figure(figsize=(24,12)) 
    gs1 = gridspec.GridSpec(1, 4)
    
    # PCT(10)
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
    im = plt.scatter(lon_gmi, lat_gmi, 
           c=PCT10, s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('PCT 10 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    CS = plt.contour(lon, lat, 
                        data_pd10,[180,250], colors=('k','r'), linewidths=(linewidths));
    labels = ["180K","250K"]
    if len(CS.collections) == 1:
        for i in range(len(labels)-1):
            CS.collections[i].set_label("250K")        
    else:
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
    #ax1.legend(loc='upper left', fontsize=fontsize)
    ax1.set_xlabel('Latitude', fontsize=fontsize)
    ax1.set_ylabel('Longitude', fontsize=fontsize)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(a)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    p1 = ax1.get_position().get_points().flatten()    
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')
    if len(PCT10) > 0:
        ax1.text(-50,-44,'[minPCT10='+str("%.2f" % np.nanmin(PCT10))+']',fontsize=fontsize,va='top',ha='right',color='darkred')
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m')         
        
    # PCT(19)   
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
    im = plt.scatter(lon_gmi, lat_gmi, 
           c=PCT19, s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('PCT 19 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    CS = plt.contour(lon, lat, 
                        data_pd19,[180,250], colors=('k','r'), linewidths=(linewidths));
    labels = ["180K","250K"]
    if len(CS.collections) == 1:
        for i in range(len(labels)-1):
            CS.collections[i].set_label("250K")        
    else:
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(b)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')
    if len(PCT10) > 0:
        ax1.text(-50,-44,'[minPCT19='+str("%.2f" % np.nanmin(PCT19))+']',fontsize=fontsize,va='top',ha='right',color='darkred')
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m')         
        
    # PCT(37)       
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
    im = plt.scatter(lon_gmi, lat_gmi, 
           c=PCT37, s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('PCT 37 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    CS = plt.contour(lon, lat, 
                        data_pd37,[180,250], colors=('k','r'), linewidths=(linewidths));
    labels = ["180K","250K"]
    if len(CS.collections) == 1:
        for i in range(len(labels)-1):
            CS.collections[i].set_label("250K")        
    else:
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
    ax1.legend(loc='lower right', fontsize=10)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')
    if len(PCT10) > 0:
        ax1.text(-50,-44,'[minPCT37='+str("%.2f" % np.nanmin(PCT37))+']',fontsize=fontsize,va='top',ha='right',color='darkred')
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m')         
        
    # PCT(89)       
    ax1 = plt.subplot(gs1[0,3], projection=ccrs.PlateCarree())
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
    im = plt.scatter(lon_gmi, lat_gmi, 
           c=PCT89, s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('PCT 89 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    CS = plt.contour(lon, lat, 
                        data_pd89,[180,250], colors=('k','r'), linewidths=(linewidths));
    labels = ["180K","250K"]
    if len(CS.collections) == 1:
        for i in range(len(labels)-1):
            CS.collections[i].set_label("250K")        
    else:
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
    ax1.legend(loc='lower right', fontsize=10)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(d)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'm')

    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m')    
        
    if len(PCT10) > 0:
        ax1.text(-50,-44,'[minPCT89='+str("%.2f" % np.nanmin(PCT89))+']',fontsize=fontsize,va='top',ha='right',color='darkred')
 
    # Now add "area_politica" text: text(x,y, ...)
    text_ACPs = []
    for i in range(len(pol_n)):
        #print('area_politica: ',  area_politica[i])
        text_ACPs.append( area_politica[i] )
    if len(text_ACPs)>0:
        text_ACPs[0] = 'ACPs durante el dia en: '+text_ACPs[0]
    ax1.text(-50,-45,text_ACPs,fontsize=fontsize,va='top',ha='right',color='darkred')
    
        
    p2 = ax1.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[0], 0.2, p1[2]*2.7, 0.05])
    cbar = fig.colorbar(im, cax=ax_cbar, shrink=1,ticks=np.arange(50,310,10), extend='both', orientation="horizontal")
    cbar.set_label('PCT (K)', fontsize=fontsize)
    
    #plt.savefig(options['path']+'/'+options['name']+'.eps')
    plt.savefig(options['path']+'/'+options['name']+'.png')
    plt.close()

    return 


#################################################################
#################################################################


def read_shpProvincias():

    import geopandas as gpd
    Prov = gpd.read_file('Provincia/ign_provincia.shp')
               
    fieldName = {}
    for i in range(len(Prov['FNA'])):
        fieldName[i] = Prov['FNA'][i]
    
    # Prov[Prov['OBJECTID']==427].plot()    # 0 caba
    # Prov[Prov['FNA']=='Provincia de Buenos Aires'].plot()
  
    
    return fieldName

#################################################################
#################################################################

def plot_polygon_GMI(tb_s2_gmi, tb_s2_gmi_PD, 
             lon_s2_gmi, lat_s2_gmi, lon, lat, tb_s2_gmi_FULL, options):
    

    inside2  = np.logical_and(np.logical_and(lon >= options['xlim_min'], lon <= options['xlim_max']), 
                             np.logical_and(lat >= options['ylim_min'], lat <= options['ylim_max']))
            
    plt.matplotlib.rc('font', family='DejaVu Sans', size = 12)

    fontsize   = 12
    linewidths = 2
    
    fig = plt.figure(figsize=(12,6)) 
    
    gs1 = gridspec.GridSpec(2, 2)
    
    
    # BT(166)           
    ax1 = plt.subplot(gs1[0,0], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax1.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax1.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax1.add_feature(states_provinces,linewidth=0.4)
    ax1.add_feature(rivers)
    ax1.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax1.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_s2_gmi, lat_s2_gmi, 
            c=tb_s2_gmi, s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.plot(lon[:,0],lat[:,0],'-k')
    plt.plot(lon[:,220],lat[:,220],'-k')
    plt.title('Clipped BT 166 GHz', fontsize=fontsize)
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
    
    ax1.text(0.05,1.10,'(a)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    cbar = fig.colorbar(im, shrink=0.9,ticks=np.arange(50,325,25), 
                        extend='both', orientation="vertical")
    cbar.set_label('BT (K)', fontsize=fontsize)

    
    # PD(166)       
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
    im = plt.scatter(lon_s2_gmi, lat_s2_gmi, 
           c=tb_s2_gmi_PD, s=10, vmin=0, vmax=12, cmap=discrete_cmap(16,  'rainbow'))  
    plt.plot(lon[:,0],lat[:,0],'-k')
    plt.plot(lon[:,220],lat[:,220],'-k')
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('Clipped PD 166 GHz', fontsize=fontsize)
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
    
        
    cbar = fig.colorbar(im, shrink=0.9,ticks=np.arange(0,17,1), extend='both', 
                        orientation="vertical")
    cbar.set_label('PD (K)', fontsize=fontsize)

    # BT(166)        FULL   
    ax1 = plt.subplot(gs1[1,0], projection=ccrs.PlateCarree())
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
        
    im = plt.scatter(lon[inside2], lat[inside2], 
           c=tb_s2_gmi_FULL[inside2,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.plot(lon[:,0],lat[:,0],'-k')
    plt.plot(lon[:,220],lat[:,220],'-k')
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('BT 166 GHz', fontsize=fontsize)
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
    cbar = fig.colorbar(im, shrink=0.9,ticks=np.arange(50,325,25), 
                        extend='both', orientation="vertical")
    cbar.set_label('BT (K)', fontsize=fontsize)

    
    # PD(166)       FULL       
    ax1 = plt.subplot(gs1[1,1], projection=ccrs.PlateCarree())
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
    im = plt.scatter(lon[inside2], lat[inside2], 
           c=tb_s2_gmi_FULL[inside2,0]-tb_s2_gmi_FULL[inside2,1], s=10, vmin=0, vmax=12, cmap=discrete_cmap(16,  'rainbow'))  
    plt.plot(lon[:,0],lat[:,0],'-k')
    plt.plot(lon[:,220],lat[:,220],'-k')
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('PD 166 GHz', fontsize=fontsize)
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
    ax1.text(0.05,1.10,'(d)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')   
    
    cbar = fig.colorbar(im, shrink=0.9,ticks=np.arange(0,17,1), extend='both', 
                        orientation="vertical")
    cbar.set_label('PD (K)', fontsize=fontsize)

    #plt.tight_layout()
    
    fig.suptitle(options['title'] ,fontweight='bold' )
    plt.tight_layout()
    plt.subplots_adjust(top=0.899)
    #plt.savefig(options['path']+'/'+options['name']+'.eps')
    plt.savefig(options['path']+'/'+options['name']+'.png')
    plt.close()
    
    return 

#################################################################
#################################################################

def plot_polygon_GMI(tb_s2_gmi, tb_s2_gmi_PD, 
             lon_s2_gmi, lat_s2_gmi, lon, lat, tb_s2_gmi_FULL, options):
    

    inside2  = np.logical_and(np.logical_and(lon >= options['xlim_min'], lon <= options['xlim_max']), 
                             np.logical_and(lat >= options['ylim_min'], lat <= options['ylim_max']))
            
    plt.matplotlib.rc('font', family='DejaVu Sans', size = 12)

    fontsize   = 12
    linewidths = 2
    
    fig = plt.figure(figsize=(12,6)) 
    
    gs1 = gridspec.GridSpec(2, 2)
    
    
    # BT(166)           
    ax1 = plt.subplot(gs1[0,0], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax1.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax1.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax1.add_feature(states_provinces,linewidth=0.4)
    ax1.add_feature(rivers)
    ax1.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax1.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_s2_gmi, lat_s2_gmi, 
            c=tb_s2_gmi, s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.plot(lon[:,0],lat[:,0],'-k')
    plt.plot(lon[:,220],lat[:,220],'-k')
    plt.title('Clipped BT 166 GHz', fontsize=fontsize)
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
    
    ax1.text(0.05,1.10,'(a)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    cbar = fig.colorbar(im, shrink=0.9,ticks=np.arange(50,325,25), 
                        extend='both', orientation="vertical")
    cbar.set_label('BT (K)', fontsize=fontsize)

    
    # PD(166)       
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
    im = plt.scatter(lon_s2_gmi, lat_s2_gmi, 
           c=tb_s2_gmi_PD, s=10, vmin=0, vmax=12, cmap=discrete_cmap(16,  'rainbow'))  
    plt.plot(lon[:,0],lat[:,0],'-k')
    plt.plot(lon[:,220],lat[:,220],'-k')
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('Clipped PD 166 GHz', fontsize=fontsize)
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
    
        
    cbar = fig.colorbar(im, shrink=0.9,ticks=np.arange(0,17,1), extend='both', 
                        orientation="vertical")
    cbar.set_label('PD (K)', fontsize=fontsize)

    # BT(166)        FULL   
    ax1 = plt.subplot(gs1[1,0], projection=ccrs.PlateCarree())
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
        
    im = plt.scatter(lon[inside2], lat[inside2], 
           c=tb_s2_gmi_FULL[inside2,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.plot(lon[:,0],lat[:,0],'-k')
    plt.plot(lon[:,220],lat[:,220],'-k')
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('BT 166 GHz', fontsize=fontsize)
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
    cbar = fig.colorbar(im, shrink=0.9,ticks=np.arange(50,325,25), 
                        extend='both', orientation="vertical")
    cbar.set_label('BT (K)', fontsize=fontsize)

    
    # PD(166)       FULL       
    ax1 = plt.subplot(gs1[1,1], projection=ccrs.PlateCarree())
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
    im = plt.scatter(lon[inside2], lat[inside2], 
           c=tb_s2_gmi_FULL[inside2,0]-tb_s2_gmi_FULL[inside2,1], s=10, vmin=0, vmax=12, cmap=discrete_cmap(16,  'rainbow'))  
    plt.plot(lon[:,0],lat[:,0],'-k')
    plt.plot(lon[:,220],lat[:,220],'-k')
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('PD 166 GHz', fontsize=fontsize)
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
    ax1.text(0.05,1.10,'(d)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')   
    cbar = fig.colorbar(im, shrink=0.9,ticks=np.arange(0,17,1), extend='both', 
                        orientation="vertical")
    cbar.set_label('PD (K)', fontsize=fontsize)
    
    
# =============================================================================
#     
#     p2 = ax1.get_position().get_points().flatten()
#     ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.05])
#     cbar = fig.colorbar(im, cax=ax_cbar, shrink=1,ticks=np.arange(0,17,1), extend='both', orientation="horizontal")
#     cbar.set_label('PD (K)', fontsize=fontsize)
# 
# 
# =============================================================================

    plt.tight_layout()
    
    fig.suptitle(options['title'] ,fontweight='bold' )
    plt.tight_layout()
    plt.subplots_adjust(top=0.899)
    #plt.savefig(options['path']+'/'+options['name']+'.eps')
    plt.savefig(options['path']+'/'+options['name']+'.png')
    plt.close()
    
    return 
  
#################################################################
#################################################################

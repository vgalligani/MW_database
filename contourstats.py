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
from copy import deepcopy
from pyart.correct import phase_proc
import xarray as xr
import matplotlib.colors as colors
import wradlib as wrl    
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from cycler import cycler
#import seaborn as sns
import cartopy.io.shapereader as shpreader
import copy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import gc
import math
from pyart.core.transforms import antenna_to_cartesian
from collections import Counter
import alphashape
from descartes import PolygonPatch
import cartopy.feature as cfeature
from matplotlib.path import Path

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import h5py
from csu_radartools import csu_fhc

plt.matplotlib.rc('font', family='serif', size = 12)
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12  

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
            iinds      = [inds_1]
       	if ii == 1:
            inds_2   = concave_path.contains_points(datapts_in)
            iinds      = [inds_1, inds_2]
        if ii == 2:
            inds_3   = concave_path.contains_points(datapts_in)
            iinds      = [inds_1, inds_2, inds_3]
        if ii == 3:
            inds_4   = concave_path.contains_points(datapts_in)
            iinds      = [inds_1, inds_2, inds_3, inds_4]   

    return iinds

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def points_in_polygon(pts, polygon):
    """
    Returns if the points are inside the given polygon,

    Implemented with angle accumulation.
    see: https://en.wikipedia.org/wiki/Point_in_polygon#Winding_number_algorithm

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


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#https://stackoverflow.com/questions/70701788/find-points-that-lie-in-a-concave-hull-of-a-point-cloud
#---------------------------------------------------------------------------------------------
def get_contour_info_IMPROVED(contorno, icois, datapts_in): 

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
	
        array_points = np.array(vertices)
        cond = points_in_polygon(datapts_in, array_points)
        
        if ii == 0:
            inds_1   = points_in_polygon(datapts_in, array_points)
            iinds    = [inds_1]
       	if ii == 1:
            inds_2   = points_in_polygon(datapts_in, array_points)
            iinds    = [inds_1, inds_2]
        if ii == 2:
            inds_3   = points_in_polygon(datapts_in, array_points)
            iinds    = [inds_1, inds_2, inds_3]
        if ii == 3:
            inds_4   = points_in_polygon(datapts_in, array_points)
            iinds    = [inds_1, inds_2, inds_3, inds_4]   

    return iinds

#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def circ_radius(p0,p1,p2):
    """
    Vectorized computation of triangle circumscribing radii.
    See for example https://www.cuemath.com/jee/circumcircle-formulae-trigonometry/
    """
    a = p1-p0
    b = p2-p0

    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    norm_a_b = np.linalg.norm(a-b, axis=1)
    cross_a_b = np.cross(a,b)  # 2 * area of triangles
    return (norm_a*norm_b*norm_a_b) / np.abs(2.0*cross_a_b)

#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def alpha_shape_delaunay_mask(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points and return the Delaunay triangulation and a boolean
    mask for any triangle in the triangulation whether it belongs to the alpha shape.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :return: Delaunay triangulation dt and boolean array is_in_shape, so that dt.simplices[is_in_alpha] contains
    only the triangles that belong to the alpha shape.
    """
    # Modified and vectorized from:
    # https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points/50714300#50714300

    assert points.shape[0] > 3, "Need at least four points"
    dt = Delaunay(points)

    p0 = points[dt.simplices[:,0],:]
    p1 = points[dt.simplices[:,1],:]
    p2 = points[dt.simplices[:,2],:]

    rads = circ_radius(p0, p1, p2)

    is_in_shape = (rads < alpha)

    return dt, is_in_shape

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------    
def in_alpha_shape(p, dt, is_in_alpha):
    simplex_ids = dt.find_simplex(p)
    res = np.full(p.shape[0], False)
    res[simplex_ids >= 0] = is_in_alpha[simplex_ids[simplex_ids >= 0]]  # simplex should be in dt _and_ in alpha
    return res
#----------------------------------------------------------------------------------------------

def get_gmi(fname, options, radar, lon_pfs, lat_pfs, icoi):

    user = platform.system()
    if   user == 'Linux':
        home_dir = '/home/victoria.galligani/'  
    elif user == 'Darwin':
        home_dir = '/Users/victoria.galligani'

    # read file
    f = h5py.File( fname, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()

    return lon_gmi, lat_gmi, tb_s1_gmi 
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
    TB_inds = get_contour_info_IMPROVED(contorno89, icoi, datapts) #get_contour_info(contorno89, icoi, datapts)	
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
from matplotlib import text as mtext

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_gmi_paper_fig5(fname, options, radar, lon_pfs, lat_pfs, icoi, transects, caso):

    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12  

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
        nlev = 0 
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]	
        reflectivity_name = 'corrected_reflectivity'   
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]

    elif options['radar_name'] == 'RMA5':
        reflectivity_name = 'DBZH'; nlev=0;
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]

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



    #----------------------------------------------------------------------------------------
    # NEW FIGURE. solo dos paneles: Same as above but plt lowest level y closest to freezing level!
    #----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True,figsize=[10,5])

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    #ZH[np.where(ZH<20)]=np.nan
    pcm1=axes[0].pcolormesh(lons, lats, ZH, cmap=cmap, vmax=vmax, vmin=vmin)
    axes[0].set_title('Ground Level')
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    plt.colorbar(pcm1, ax=axes[0],label='(dBZ)')


    # -----
    # CONTORNO CORREGIDO POR PARALAJE Y PODER CORRER LOS ICOIS, simplemente pongo nans fuera del area de interes ... 
    contorno89 = axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);


    # LIMITS	 
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    # RADAR RINGS	      
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    CurvedText(
        x = lon_radius[20000:],
        y = lat_radius[20000:],
        text='50 km',#'this this is a very, very long text',
        va = 'bottom', axes=axes[0])

    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 

    # Add labels:
    labels = ["PCT 89-GHz 200 K contour"] 
    for i in range(len(labels)):
        contorno89.collections[i].set_label(labels[i])



    if caso == '20200815': 
        axes[0].set_title(r'RMA5 Zh (15/8/2018 0216UTC), Elev: 0.7$^{o}$')
        axes[0].legend(loc='lower left')
        CurvedText(
           	x = lon_radius[19000:],
           	y = lat_radius[19000:],
           	text='100 km',#'this this is a very, very long text',
           	va = 'bottom', axes=axes[0])
        # Addlabels to icois! 
        axes[0].text(-54.9, -25.2, 'coi=1')
        axes[0].text(-53.4, -25.5, 'coi=2')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        for i in range(len(lon_pfs)):
            axes[0].plot(lon_pfs[i]-0.3, lat_pfs[i], marker='*', markersize=20, markerfacecolor="black",
            markeredgecolor='black', markeredgewidth=1.5)
            
    elif caso == '20181214': 
        axes[0].set_title(r'DOW7 Zh (14/12/2018 0220UTC), Elev: 0.7$^{o}$')
        axes[0].legend(loc='upper left')
        CurvedText(
            x = lon_radius[30000:],
            y = lat_radius[30000:],
            text='50 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[0])
	
    elif caso == '20181111': 
        axes[0].set_title(r'CSPR2 Zh (11/11/2018 1300UTC), Elev: 0.7$^{o}$')

        axes[0].legend(loc='upper left')

        # RADAR RINGS	      
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
        CurvedText(
            x = lon_radius[25000:],
            y = lat_radius[25000:],
            text='50 km',#'this this is a very, very long text',
            va = 'top', axes=axes[0])

        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        CurvedText(
            x = lon_radius[30000:],
            y = lat_radius[30000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[0])

    elif caso == '20180208': 
        axes[0].set_title(r'RMA1 Zh (8/2/2018 2057UTC), Elev: 0.7$^{o}$')
        axes[0].legend(loc='upper left')
        CurvedText(
            x = lon_radius[17000:],
            y = lat_radius[17000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[0])	
        # Addlabels to icois! 
        axes[0].text(-64, -31, 'coi=1')
        axes[0].text(-65.3, -32.0, 'coi=2')
        axes[0].text(-65, -32.7, 'coi=3')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        for i in range(len(lon_pfs)):
            axes[0].plot(lon_pfs[i], lat_pfs[i]-0.05, marker='*', markersize=20, markerfacecolor="black",
            markeredgecolor='black', markeredgewidth=1.5)	

    elif caso == '20181031': 
        axes[0].set_title(r'RMA4 Zh (31/10/2018 0109UTC), Elev: 0.7$^{o}$')
        axes[0].legend(loc='upper left')
        CurvedText(
            x = lon_radius[50000:],
            y = lat_radius[50000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[0])	
        # Addlabels to icois! 
        axes[0].text(-61, -28.2, 'coi=1')
        axes[0].text(-59.5, -28.2, 'coi=2')
        axes[0].text(-57.8, -28.4, 'coi=3')
        axes[0].text(-58.5, -26.4, 'coi=4')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        for i in range(len(lon_pfs)):
            axes[0].plot(lon_pfs[i], lat_pfs[i]-0.05, marker='*', markersize=20, markerfacecolor="black",
            markeredgecolor='black', markeredgewidth=1.5)	


    if len(options['REPORTES_meta'])>0:
        for ireportes in range(len(options['REPORTES_geo'])):
            axes[0].plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, label=options['REPORTES_meta'][ireportes])


    plt.legend()
    if options['radar_name'] == 'DOW7':
       	azimuths = radar.azimuth['data']
    else:
       	azimuths = radar.azimuth['data'][start_index:end_index]
    # TRANSECTAS:
    for itrans in transects:
        target_azimuth = azimuths[itrans]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect     = lons[filas,:]
        lat_transect     = lats[filas,:]
        axes[0].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')
        #plt.title('Transecta Nr:'+ str(test_transect), Fontsize=20)

    if len(options['REPORTES_meta'])>0:
        for ireportes in range(len(options['REPORTES_geo'])):
            axes[0].plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, label=options['REPORTES_meta'][ireportes])
        plt.legend() 



    im = axes[1].scatter(lon_gmi, lat_gmi, c=PCT89, marker='h', s=100, vmin=100, vmax=300, cmap=cmaps['turbo_r'])  
    axes[1].set_title('PCT 89-GHz')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    contorno89 = axes[1].contour(lon_gmi, lat_gmi, PCT89, [200], colors=('k'), linewidths=2);    
    for i in range(len(lon_pfs)):
        axes[1].plot(lon_pfs[i], lat_pfs[i]-0.05, marker='*', markersize=20, markerfacecolor="black",
        markeredgecolor='black', markeredgewidth=1.5)	
    # TRANSECTAS:
    for itrans in transects:
        target_azimuth = azimuths[itrans]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect     = lons[filas,:]
        lat_transect     = lats[filas,:]
        axes[1].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')
        #plt.title('Transecta Nr:'+ str(test_transect), Fontsize=20)

    if len(options['REPORTES_meta'])>0:
        for ireportes in range(len(options['REPORTES_geo'])):
            axes[1].plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, label=options['REPORTES_meta'][ireportes])
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')	
    axes[1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1].set_ylim([options['ylim_min'], options['ylim_max']])	
    #fig.savefig(options['fig_dir']+'GMI_icois_onZH.png', dpi=300, transparent=False)  
    #plt.close()
    plt.colorbar(im, ax=axes[1],label='(K)')





    return

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def plot_gmi_paper(fname, options, radar, lon_pfs, lat_pfs, icoi, transects, caso):

    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12  

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
        reflectivity_name = 'DBZH'; nlev=0;
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]
	
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
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True,figsize=[5,10])
	
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    #ZH[np.where(ZH<20)]=np.nan
    pcm1=axes[0].pcolormesh(lons, lats, ZH, cmap=cmap, vmax=vmax, vmin=vmin)
    axes[0].set_title('Ground Level')
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    plt.colorbar(pcm1, ax=axes[0],label='(dBZ)')


    # -----
    # CONTORNO CORREGIDO POR PARALAJE Y PODER CORRER LOS ICOIS, simplemente pongo nans fuera del area de interes ... 
    contorno89 = axes[0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200,250], colors=(['k']), linewidths=1.5);
	
    
    # LIMITS	 
    axes[0].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[0].set_ylim([options['ylim_min'], options['ylim_max']])
    # RADAR RINGS	      
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    CurvedText(
        x = lon_radius[20000:],
        y = lat_radius[20000:],
        text='50 km',#'this this is a very, very long text',
        va = 'bottom', axes=axes[0])

    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
	
    # Add labels:
    labels = ["PCT 89-GHz 200 K contour","PCT 89-GHz 250 K contour"] 
    for i in range(len(labels)):
        contorno89.collections[i].set_label(labels[i])


	
    if caso == '20200815': 
        axes[0].set_title(r'RMA5 Zh (15/8/2018 0216UTC), Elev: 0.7$^{o}$')
        axes[0].legend(loc='lower left')
        CurvedText(
           	x = lon_radius[19000:],
           	y = lat_radius[19000:],
           	text='100 km',#'this this is a very, very long text',
           	va = 'bottom', axes=axes[0])
        # Addlabels to icois! 
        axes[0].text(-54.9, -25.2, 'coi=1')
        axes[0].text(-53.4, -25.5, 'coi=2')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        for i in range(len(lon_pfs)):
            axes[0].plot(lon_pfs[i]-0.3, lat_pfs[i], marker='*', markersize=20, markerfacecolor="black",
            markeredgecolor='black', markeredgewidth=1.5)

    elif caso == '20180208': 
        axes[0].set_title(r'RMA1 Zh (8/2/2018 2057UTC), Elev: 0.7$^{o}$')
        axes[0].legend(loc='upper left')
        CurvedText(
            x = lon_radius[17000:],
            y = lat_radius[17000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[0])	
        # Addlabels to icois! 
        axes[0].text(-64, -31, 'coi=1')
        axes[0].text(-65.3, -32.0, 'coi=2')
        axes[0].text(-65, -32.7, 'coi=3')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        for i in range(len(lon_pfs)):
            axes[0].plot(lon_pfs[i], lat_pfs[i]-0.05, marker='*', markersize=20, markerfacecolor="black",
            markeredgecolor='black', markeredgewidth=1.5)	
 
    elif caso == '20181031': 
        axes[0].set_title(r'RMA4 Zh (31/10/2018 0109UTC), Elev: 0.7$^{o}$')
        axes[0].legend(loc='upper left')
        CurvedText(
            x = lon_radius[50000:],
            y = lat_radius[50000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[0])	
        # Addlabels to icois! 
        axes[0].text(-61, -28.2, 'coi=1')
        axes[0].text(-59.5, -28.2, 'coi=2')
        axes[0].text(-57.8, -28.4, 'coi=3')
        axes[0].text(-58.5, -26.4, 'coi=4')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        for i in range(len(lon_pfs)):
            axes[0].plot(lon_pfs[i], lat_pfs[i]-0.05, marker='*', markersize=20, markerfacecolor="black",
            markeredgecolor='black', markeredgewidth=1.5)	
	
	
    if len(options['REPORTES_meta'])>0:
        for ireportes in range(len(options['REPORTES_geo'])):
            axes[0].plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, label=options['REPORTES_meta'][ireportes])
    

    plt.legend()
    azimuths = radar.azimuth['data'][start_index:end_index]
    # TRANSECTAS:
    for itrans in transects:
        target_azimuth = azimuths[itrans]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect     = lons[filas,:]
        lat_transect     = lats[filas,:]
        axes[0].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')
        #plt.title('Transecta Nr:'+ str(test_transect), Fontsize=20)
	
    if len(options['REPORTES_meta'])>0:
        for ireportes in range(len(options['REPORTES_geo'])):
            axes[0].plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, label=options['REPORTES_meta'][ireportes])
        plt.legend() 
	
	

    im = axes[1].scatter(lon_gmi, lat_gmi, c=PCT89, marker='h', s=100, vmin=100, vmax=300, cmap=cmaps['turbo_r'])  
    axes[1].set_title('PCT 89-GHz')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    contorno89 = axes[1].contour(lon_gmi, lat_gmi, PCT89, [200,225], colors=('k'), linewidths=2);    
    for i in range(len(lon_pfs)):
        axes[1].plot(lon_pfs[i], lat_pfs[i]-0.05, marker='*', markersize=20, markerfacecolor="black",
        markeredgecolor='black', markeredgewidth=1.5)	
    azimuths = radar.azimuth['data'][start_index:end_index]
    # TRANSECTAS:
    for itrans in transects:
        target_azimuth = azimuths[itrans]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect     = lons[filas,:]
        lat_transect     = lats[filas,:]
        axes[1].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')
        #plt.title('Transecta Nr:'+ str(test_transect), Fontsize=20)
	
    if len(options['REPORTES_meta'])>0:
        for ireportes in range(len(options['REPORTES_geo'])):
            axes[1].plot( options['REPORTES_geo'][ireportes][1],  options['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, label=options['REPORTES_meta'][ireportes])
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')	
    axes[1].set_xlim([options['xlim_min'], options['xlim_max']])
    axes[1].set_ylim([options['ylim_min'], options['ylim_max']])	
    #fig.savefig(options['fig_dir']+'GMI_icois_onZH.png', dpi=300, transparent=False)  
    #plt.close()
    plt.colorbar(im, ax=axes[1],label='(K)')

    return



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------





def correct_PHIDP_KDP(radar, options, nlev, azimuth_ray, diff_value, tfield_ref, alt_ref):

    if 'TH' in radar.fields.keys():  
        sys_phase = get_sys_phase_simple(radar)
    elif 'DBZHCC' in radar.fields.keys():
        sys_phase = get_sys_phase_simple_dow7(radar)
    elif 'DBZH' in radar.fields.keys():
        sys_phase = get_sys_phase_simple(radar)	
    #------------
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
    calculated_KDP = wrl.dp.kdp_from_phidp(corr_phidp, winlen=7, dr=(radar.range['data'][1]-radar.range['data'][0])/1e3, 
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
    # ACA, RN_inds_parallax es pixels del radar (dato original) dentro del contorno
    RN_inds_parallax =  get_contour_info(contorno89_FIX, icois, datapts_RADAR_NATIVE)
    # tambien me va a interesar el grillado a diferentes resoluciones 
	
    # y usando contornos de DBZ sumados? en vez del contorno de sarah. 

	
    GMI_tbs1_37 = []
    GMI_tbs1_85 = [] 	
    # aca agrego V,H 
    GMI_tbs1_37H = []
    GMI_tbs1_85H = [] 
    GMI_tbs1_19 = []
    GMI_tbs1_19H = [] 
	
    S1_sub_tb_v2 = S1_sub_tb[:,:,:][idx1]		

    for ii in range(len(TB_inds)): 
     	GMI_tbs1_37.append( S1_sub_tb_v2[TB_inds[ii],5] ) 
     	GMI_tbs1_85.append( S1_sub_tb_v2[TB_inds[ii],7] ) 
		
     	GMI_tbs1_37H.append( S1_sub_tb_v2[TB_inds[ii],6] ) 
     	GMI_tbs1_85H.append( S1_sub_tb_v2[TB_inds[ii],8] ) 
     	GMI_tbs1_19.append( S1_sub_tb_v2[TB_inds[ii],2] ) 
     	GMI_tbs1_19H.append( S1_sub_tb_v2[TB_inds[ii],3] ) 
	
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
            if (rho_h[j]<0.8) or (zh_h[j]<35):
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
def GET_TBVH_250ICOIS(options, fname, changui):

    home_dir = '/home/victoria.galligani/'  	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    r_dir    = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'
    radar = pyart.io.read(r_dir+options['rfile'])
	
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

    # ESTO ACA REDUJE A 1! ESTABA EN 5. para tener menos contornos!  para el caso 1 en CDB es 1!!! 
    idx1 = (lat_gmi>=options['ylim_min']-changui) & (lat_gmi<=options['ylim_max']+changui) & (lon_gmi>=options['xlim_min']-changui) & (lon_gmi<=options['xlim_max']+changui)

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
	# USAR CONTORNO DE 250 !!!
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
        contorno89 = plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [250] , colors=(['r']), linewidths=1.5);
        contorno89_FIX = plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:] , [250], colors=(['k']), linewidths=1.5);
        axes.set_xlim([options['xlim_min'], options['xlim_max']]) 
        axes.set_ylim([options['ylim_min'], options['ylim_max']])

    datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] )) 

    # Get vertices of these polygon type shapes
    for ii in range(len(contorno89.collections[0].get_paths())): 
        X1 = []; Y1 = []; vertices = []
        for ik in range(len(contorno89.collections[0].get_paths()[ii].vertices)): 
            X1.append(contorno89.collections[0].get_paths()[ii].vertices[ik][0])
            Y1.append(contorno89.collections[0].get_paths()[ii].vertices[ik][1])
            vertices.append([contorno89.collections[0].get_paths()[ii].vertices[ik][0], 
                                        contorno89.collections[0].get_paths()[ii].vertices[ik][1]])
        array_points = np.array(vertices)
        alpha = 0.95 * alphashape.optimizealpha(array_points)
        hull_pts_CONCAVE   = alphashape.alphashape(array_points, alpha)
        hull_coors_CONCAVE = hull_pts_CONCAVE.exterior.coords.xy
        check_points = np.vstack((hull_coors_CONCAVE)).T
        concave_path = Path(check_points)
        inds   = concave_path.contains_points(datapts)
        plt.plot(lon_gmi[:,:][idx1][inds], lat_gmi[:,:][idx1][inds], marker='o', label=str(ii))
        plt.legend()

    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def GET_TBVH_250_TBVHplots(options, icois, fname, changui):

    home_dir = '/home/victoria.galligani/'  	
    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    r_dir    = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'
    radar = pyart.io.read(r_dir+options['rfile'])
	
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

    # ESTO ACA REDUJE A 1! ESTABA EN 5. para tener menos contornos! 
    idx1 = (lat_gmi>=options['ylim_min']-changui) & (lat_gmi<=options['ylim_max']+changui) & (lon_gmi>=options['xlim_min']-changui) & (lon_gmi<=options['xlim_max']+changui)

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
	# USAR CONTORNO DE 250 !!!
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
        contorno89 = plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [250] , colors=(['r']), linewidths=1.5);
        contorno89_FIX = plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:] , [250], colors=(['k']), linewidths=1.5);
        axes.set_xlim([options['xlim_min'], options['xlim_max']]) 
        axes.set_ylim([options['ylim_min'], options['ylim_max']])

	
    colores_in = ['k','darkgreen','darkred','darkblue','cyan']
    datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] ))
    TB_inds = get_contour_info(contorno89, icois, datapts)
    lines = []
    for ii in range(len(TB_inds)):
        lines += axes.plot(lon_gmi[:,:][idx1][TB_inds[ii]] , lat_gmi[:,:][idx1][TB_inds[ii]], 'o', markersize=10, markerfacecolor=colores_in[ii], 
			   markeredgecolor=colores_in[ii], label=str(icois[ii])) 
    plt.legend()



    # tambien me va a interesar el grillado a diferentes resoluciones 	
    GMI_tbs1_37 = []
    GMI_tbs1_85 = [] 	
    # aca agrego V,H 
    GMI_tbs1_37H = []
    GMI_tbs1_85H = [] 
    GMI_tbs1_19 = []
    GMI_tbs1_19H = [] 
    GMI_latlat = []
    GMI_lonlon = []

    S1_sub_tb_v2 = S1_sub_tb[:,:,:][idx1]		

    for ii in range(len(TB_inds)): 
        GMI_tbs1_37.append( S1_sub_tb_v2[TB_inds[ii],5] ) 
        GMI_tbs1_85.append( S1_sub_tb_v2[TB_inds[ii],7] ) 
        GMI_tbs1_37H.append( S1_sub_tb_v2[TB_inds[ii],6] ) 
        GMI_tbs1_85H.append( S1_sub_tb_v2[TB_inds[ii],8] ) 
        GMI_tbs1_19.append( S1_sub_tb_v2[TB_inds[ii],2] ) 
        GMI_tbs1_19H.append( S1_sub_tb_v2[TB_inds[ii],3] ) 
        GMI_latlat.append( lat_gmi[:,:][idx1][TB_inds[ii]] )
        GMI_lonlon.append( lon_gmi[:,:][idx1][TB_inds[ii]] )
	
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

    if test_this == 1: 
        fig = plt.figure(figsize=(30,10)) 
       	gs1 = gridspec.GridSpec(1, 4)

       	ax1 = plt.subplot(gs1[0,0], projection=ccrs.PlateCarree())

        crs_latlon = ccrs.PlateCarree()
        ax1.set_extent([options['xlim_min'], options['xlim_max'], options['ylim_min'], options['ylim_max']], crs=crs_latlon)
        ax1.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax1.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
        ax1.add_feature(states_provinces,linewidth=0.4)
        ax1.add_feature(rivers)
        ax1.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
        im  = ax1.scatter(S1_sub_lon, S1_sub_lat, c= S1_sub_tb[:,:,2]- S1_sub_tb[:,:,3] , s=450, marker='h', vmin=0, vmax=20, cmap='jet'); 
	#im = plt.scatter(lon_gmi, lat_gmi, GMI_tbs1_19-GMI_tbs1_19H, s=s_sizes, marker='h', vmin=0, vmax=20, cmap=cmaps['turbo_r'])  
        plt.title('BT(V-H) 19  GHz')
        ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
        ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max'],1), crs=crs_latlon)
        ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        plt.contour(lon_gmi, lat_gmi, PCT89, [200, 250], colors=('m'), linewidths=2);



       	ax2 = plt.subplot(gs1[0,1], projection=ccrs.PlateCarree())
        crs_latlon = ccrs.PlateCarree()
        ax2.set_extent([options['xlim_min'], options['xlim_max'], options['ylim_min'], options['ylim_max']], crs=crs_latlon)
        ax2.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax2.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
        edgecolor="black", facecolor='none')
        ax2.add_feature(states_provinces,linewidth=0.4)
        ax2.add_feature(rivers)
        ax2.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
        edgecolor="black", facecolor='none')
        im  = ax2.scatter(S1_sub_lon, S1_sub_lat, c= S1_sub_tb[:,:,5]- S1_sub_tb[:,:,6] , s=450, marker='h', vmin=0, vmax=20, cmap='jet'); 
        plt.title('BT(V-H) 37  GHz')
        ax2.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
        ax2.set_yticks(np.arange(options['ylim_min'], options['ylim_max'],1), crs=crs_latlon)
        ax2.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        plt.contour(lon_gmi, lat_gmi, PCT89, [200, 250], colors=('m'), linewidths=2);


       	ax3 = plt.subplot(gs1[0,2], projection=ccrs.PlateCarree())
        crs_latlon = ccrs.PlateCarree()
        ax3.set_extent([options['xlim_min'], options['xlim_max'], options['ylim_min'], options['ylim_max']], crs=crs_latlon)
        ax3.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax3.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
        edgecolor="black", facecolor='none')
        ax3.add_feature(states_provinces,linewidth=0.4)
        ax3.add_feature(rivers)
        ax3.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
        edgecolor="black", facecolor='none')
        im  = ax3.scatter(S1_sub_lon, S1_sub_lat, c= S1_sub_tb[:,:,7]- S1_sub_tb[:,:,8] , s=450, marker='h', vmin=0, vmax=20, cmap='jet'); 
        plt.title('BT(V-H) 85  GHz')
        ax3.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
        ax3.set_yticks(np.arange(options['ylim_min'], options['ylim_max'],1), crs=crs_latlon)
        ax3.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        plt.contour(lon_gmi, lat_gmi, PCT89, [200, 250], colors=('m'), linewidths=2);
        
        p1 = ax1.get_position().get_points().flatten()
        p2 = ax3.get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.03])   # [left, bottom, width, height] or Bbox 
        cbar = fig.colorbar(im, cax=ax_cbar, shrink=0.8, ticks=np.arange(0,20,2), extend='both', orientation="horizontal", label='TBVH (K)')



       	ax4 = plt.subplot(gs1[0,3], projection=ccrs.PlateCarree())
        crs_latlon = ccrs.PlateCarree()
        ax4.set_extent([options['xlim_min'], options['xlim_max'], options['ylim_min'], options['ylim_max']], crs=crs_latlon)
        ax4.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax4.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
        edgecolor="black", facecolor='none')
        ax4.add_feature(states_provinces,linewidth=0.4)
        ax4.add_feature(rivers)
        ax4.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
        edgecolor="black", facecolor='none')
        im  = ax4.scatter(S1_sub_lon, S1_sub_lat, c= S1_sub_tb[:,:,7], s=450, marker='h', vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
        plt.title('BTV 85  GHz')
        ax4.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
        ax4.set_yticks(np.arange(options['ylim_min'], options['ylim_max'],1), crs=crs_latlon)
        ax4.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+2,2), crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        plt.contour(lon_gmi, lat_gmi, PCT89, [200, 250], colors=('m'), linewidths=2);



    return  GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H

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
            if (rho_h[j]<0.8) or (zh_h[j]<35):
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
            if (rho_h[j]<0.7) or (zh_h[j]<30):  # 0.7 y 0.3
                phiphi[i,j]  = np.nan 
                rho[i,j]     = np.nan

		
    phiphi[:,0:80]  = np.nan 
    rho[:,0:80]    = np.nan 
	
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
    #phi_cor[rho<0.8] = np.nan
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
                ajuste=np.polyoptionsfit(x,y,1)
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

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#---------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------- 
def firstNonNan(listfloats):
  for item in listfloats:
    if math.isnan(item) == False:
      return item
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
            if firstNonNan(PHIDP[radial,60:]):
                phases.append(firstNonNan(PHIDP[radial,60:])) #FOR RMA1 :60, SINO :30?
        phases_nlev.append(np.median(phases))
    phases_out = np.nanmedian(phases_nlev) 

    return phases_out
# --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
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


def add_43prop_field(radar):
	
	radar_height = get_z_from_radar(radar)
	radar = add_field_to_radar_object(radar_height, radar, field_name = 'height')    
	iso0 = np.ma.mean(radar.fields['height']['data'][np.where(np.abs(radar.fields['sounding_temperature']['data']) < 0)])
	radar.fields['height_over_iso0'] = deepcopy(radar.fields['height'])
	radar.fields['height_over_iso0']['data'] -= iso0 
	
	return radar 


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
        #dzv_  = radar.fields['DBZVCC']['data'].copy()
        dZDR  = radar.fields['ZDRC']['data'].copy()
    drho_ = radar.fields['RHOHV']['data'].copy()
    dkdp_ = radar.fields['KDP']['data'].copy()

    # ESTO DE ACA ABAJO PROBADO PARA RMA3:  
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
                #dzv_[i,j]  = np.nan
                drho_[i,j]  = np.nan
                dkdp_[i,j]  = np.nan

    scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=dZDR - options['ZDRoffset'], 
					     rho=drho_, kdp=dkdp_, 
                                             use_temp=True, band='C', T=radar_T)

    RHIs_nlev = np.argmax(scores, axis=0) + 1 
    radar.add_field_like('KDP','HID', RHIs_nlev, replace_existing=True)

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



#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def plot_icois_HIDinfo(options, radar, icois, fname):

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
        #plt.close()
	
	
	

    #------------------------------------------------------
    # histogram de HID dentro de cada contorno
    #------------------------------------------------------
    alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel( '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'+options['era5_file'], 
							    options['lat_pfs'], options['lon_pfs']) 
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    radar = add_43prop_field(radar)   

    if options['radar_name'] == 'DOW7':
        radar = DOW7_NOcorrect_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        gridded_iz  = 41
        gridded_ixy = 355 

    elif options['radar_name'] == 'CSPR2':
        radar = CSPR2_correct_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)	
        gridded_iz  = 41
        gridded_ixy = 440 

    else:
        radar = correct_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
        gridded_iz  = 40
        gridded_ixy = 940 
	
	
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
    #--------------------------------------------------------------------------------------
    datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] )) 
    datapts_RADAR_NATIVE = np.column_stack(( np.ravel(lons),np.ravel(lats) ))
    #--------------------------------------------------------------------------------------
    TB_inds = get_contour_info_IMPROVED(contorno89, icois, datapts)  #get_contour_info(contorno89, icois, datapts)
    RN_inds_parallax = get_contour_info_IMPROVED(contorno89_FIX, icois, datapts_RADAR_NATIVE) #get_contour_info(contorno89_FIX, icois, datapts_RADAR_NATIVE)	
    # tambien me va a interesar el grillado a diferentes resoluciones 
    # y usando contornos de DBZ sumados? en vez del contorno de sarah.
    #--------------------------------------------------------------------------------------	
    # FIGURE CHECK CONTORNOS
    if test_this == 1: 	
    	fig = plt.figure(figsize=(20,7)) 
    	plt.pcolormesh(lons, lats, radarTH, cmap=cmap, vmin=vmin, vmax=vmax)
    	for ic in range(len(icois)):
        	plt.plot(lon_gmi[:,:][idx1][TB_inds[ic]], lat_gmi[:,:][idx1][TB_inds[ic]],'x' );    
        	plt.plot( np.ravel(lons)[RN_inds_parallax[ic]], 	np.ravel(lats)[RN_inds_parallax[ic]], 'om')
    	plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200], colors=(['r']), linewidths=1.5);
    	plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
    	plt.xlim([options['xlim_min'], options['xlim_max']]) 
    	plt.ylim([options['ylim_min'], options['ylim_max']])
    #------------------------------------------------------

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

    #------------------------------------------------------
    name = ['Drizzle','Rain', 'Ice C.', 'Agg.', 'WS', 'V. Ice', 'LD Gr.', 'HD Gr.', 'Hail', 'BD']
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid       = colors.ListedColormap(hid_colors)
    barWidth = 0.15 
    # ahora si, HID por contorno! y por sweep
    HIDs_coi_nlev = np.zeros((radar.nsweeps, len(RN_inds_parallax), 10)); HIDs_coi_nlev[:]=np.nan
    for nlev in range(radar.nsweeps):
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats    = radar.gate_latitude['data'][start_index:end_index]
        lons    = radar.gate_longitude['data'][start_index:end_index]
        radarTH = radar.fields[THNAME]['data'][start_index:end_index]

        datapts_RADAR_NATIVE = np.column_stack(( np.ravel(lons),np.ravel(lats) ))
        RN_inds_parallax =  get_contour_info_IMPROVED(contorno89_FIX, icois, datapts_RADAR_NATIVE)  #get_contour_info(contorno89_FIX, icois, datapts_RADAR_NATIVE)	
        fig = plt.figure(figsize=(20,7)) 
        plt.pcolormesh(lons, lats, radarTH, cmap=cmap, vmin=vmin, vmax=vmax)
        #for ic in range(len(icois)):
        #    plt.plot( lon_gmi[:,:][idx1][TB_inds[ic]], lat_gmi[:,:][idx1][TB_inds[ic]],'x' );  
        #    plt.plot( np.ravel(lons)[RN_inds_parallax[ic]], 	np.ravel(lats)[RN_inds_parallax[ic]], 'om')
        #plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200], colors=(['r']), linewidths=1.5);
        plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
        plt.xlim([options['xlim_min'], options['xlim_max']]) 
        plt.ylim([options['ylim_min'], options['ylim_max']])
        plt.title('Elevation nlev '+str(nlev))
        # GUARDAR CONTORNOS. 
        fig.savefig(options['fig_dir']+'ZH_nlev_'+str(nlev)+'contours.png', dpi=300, transparent=False)  
        plt.close()


        RHIs_nlev = radar.fields['HID']['data'][start_index:end_index]
        #---- plot hid ppi  
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
        pcm1 = axes.pcolormesh(lons, lats, RHIs_nlev, cmap = cmaphid, vmin=0.2, vmax=10)
        axes.set_title('HID nlev '+str(nlev)+' PPI')
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
	#for ic in range(len(icois)):
        # 	plt.plot(lon_gmi[:,:][idx1][TB_inds[ic]], lat_gmi[:,:][idx1][TB_inds[ic]],'x' );    
        #	plt.plot( np.ravel(lons)[RN_inds_parallax[ic]], p.ravel(lats)[RN_inds_parallax[ic]], 'om')	
        plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5);
        fig.savefig(options['fig_dir']+'RHIs_nlev_'+str(nlev)+'contours.png', dpi=300, transparent=False)  
        plt.close()

       	# Entonces plot hid ppi  
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
        pcm1 = axes.pcolormesh(lons, lats, RHIs_nlev, cmap = cmaphid, vmin=0.2, vmax=10)
        axes.set_title('HID nlev '+str(nlev)+' PPI')
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
        for ic in range(len(icois)):
            plt.plot(lon_gmi[:,:][idx1][TB_inds[ic]], lat_gmi[:,:][idx1][TB_inds[ic]],'x' );    
            plt.plot( np.ravel(lons)[RN_inds_parallax[ic]], 	np.ravel(lats)[RN_inds_parallax[ic]], 'om')
        # Entonces get HIDs por contorno en \: 
        HIDs_coi = np.zeros((len(RN_inds_parallax), 10)); HIDs_coi[:]=np.nan
        for ic in range(len(icois)):
            HIDS = np.ravel(RHIs_nlev)[RN_inds_parallax[ic]]
            n, bins, patches = plt.hist(x=HIDS, bins=np.arange(0,11,1))		
            HIDs_coi[ic,:] = n
            del n, bins, patches
        plt.close()
        HIDs_coi_nlev[nlev,:,:] = HIDs_coi[:,:]
        # And barplot ... 
        fig = plt.figure(figsize=(8,3)) 
        barlabels = []
    	# Set position of bar on X axis   
       	br1 = np.arange(len(name)) #---- adjutst!
       	bar1 = plt.bar(br1, HIDs_coi[0,:], color='darkblue',  width = barWidth, label='icoi: '+str(icois[0]))
        if len(RN_inds_parallax) == 2:
            br2 = [x + barWidth for x in br1] 
            bar2 = plt.bar(br2, HIDs_coi[1,:], color='darkred',   width = barWidth, label='icoi: '+str(icois[1]))
            for rect in bar1 + bar2:
                height = rect.get_height()
                if height < 20:
                    height=np.nan;
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')
        if len(RN_inds_parallax) == 3:
            br2 = [x + barWidth for x in br1]
            br3 = [x + barWidth for x in br2]	
            bar2 = plt.bar(br2, HIDs_coi[1,:], color='darkred',   width = barWidth, label='icoi: '+str(icois[1]))
            bar3 = plt.bar(br3, HIDs_coi[2,:], color='darkgreen', width = barWidth, label='icoi: '+str(icois[2]))
            for rect in bar1 + bar2 + bar3 :
                height = rect.get_height()
                if height < 20:
                    height=np.nan;
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')
        if len(RN_inds_parallax) == 4:
            br2 = [x + barWidth for x in br1] 
            br3 = [x + barWidth for x in br2]
            br4 = [x + barWidth for x in br3]
            bar2 = plt.bar(br2, HIDs_coi[1,:], color='darkred',   width = barWidth, label='icoi: '+str(icois[1]))
            bar3 = plt.bar(br3, HIDs_coi[2,:], color='darkgreen', width = barWidth, label='icoi: '+str(icois[2]))
            bar4 = plt.bar(br4, HIDs_coi[3,:], color='black', width = barWidth, label='icoi: '+str(icois[3]))
            # Add counts above the two bar graph
            for rect in bar1 + bar2 + bar3 + bar4:
                height = rect.get_height()
                if height < 20:
                    height=np.nan;
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')
        if len(RN_inds_parallax) == 1:
            for rect in bar1:
                height = rect.get_height()
                if height < 20:
                    height=np.nan;
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')
        plt.xlabel('HID')  
        plt.xlabel('HID counts')  
        plt.xticks([r + barWidth for r in range(len(name))], name)   # adjutst! len() 
        plt.legend()
        plt.title('HID for nsweep Nr. ' + str(nlev))
        plt.grid(True)
        
        fig.savefig(options['fig_dir']+'RHIs_BARPLOT_nlev_'+str(nlev)+'contours.png', dpi=300, transparent=False)  
        plt.close()

    # Ahora con la grid for nlev = 0km, 1km, 2km, 5km, 8km, 10km
    PlotGRIDlevels = [0, 2, 4, 10, 16, 20] 
    alt_z = [0, 1, 2, 5, 8, 10] 
    # 
    # 500m grid! 
    grided  = pyart.map.grid_from_radars(radar, grid_shape=(gridded_iz, gridded_ixy, gridded_ixy), grid_limits=((0.,20000,),   #20,470,470 is for 1km
      		(-np.max(radar.range['data']), np.max(radar.range['data'])),(-np.max(radar.range['data']), np.max(radar.range['data']))), roi_func='dist', min_radius=500.0, weighting_function='BARNES2')  
    gc.collect()
    check_resolxy = grided.point_x['data'][0,0,1]-grided.point_x['data'][0,0,0]
    check_resolz  = grided.point_y['data'][0,1,0]-grided.point_y['data'][0,0,0]
    #
    datapts_RADAR_GRID = np.column_stack(( np.ravel(grided.point_longitude['data'][0,:,:]),np.ravel(grided.point_latitude['data'][0,:,:]) ))
    RNgrid_inds_parallax = get_contour_info_IMPROVED(contorno89_FIX, icois, datapts_RADAR_GRID) #get_contour_info(contorno89_FIX, icois, datapts_RADAR_GRID)		  
    gc.collect()
    #HIDs_coi_zgrid = np.zeros((len(PlotGRIDlevels), len(RNgrid_inds_parallax), 10)); HIDs_coi_zgrid[:]=np.nan
    #------------------------------------------------------	
    for nlev in range(len(PlotGRIDlevels)):
       	# Entonces plot hid GRID  
       	thislev =  PlotGRIDlevels[nlev]
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
        pcm1 = axes.pcolormesh(grided.point_longitude['data'][thislev,:,:], grided.point_latitude['data'][thislev,:,:], 
			       grided.fields['HID']['data'][thislev,:,:], cmap = cmaphid, vmin=0.2, vmax=10)
        axes.set_title('HID gridded at '+str(alt_z[nlev])+' km')
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
        for ic in range(len(icois)):
            plt.plot(lon_gmi[:,:][idx1][TB_inds[ic]], lat_gmi[:,:][idx1][TB_inds[ic]],'x' );    
            plt.plot( np.ravel(grided.point_longitude['data'][0,:,:])[RNgrid_inds_parallax[ic]], np.ravel(grided.point_latitude['data'][0,:,:])[RNgrid_inds_parallax[ic]], 'om')
        # Entonces get HIDs por contorno en HIDs_coi: 
        HIDs_coi_GRID = np.zeros((len(RNgrid_inds_parallax), 10)); HIDs_coi_GRID[:]=np.nan
        for ic in range(len(icois)):
            HIDS = np.ravel(grided.fields['HID']['data'][thislev,:,:])[RNgrid_inds_parallax[ic]]
            n, bins, patches = plt.hist(x=HIDS, bins=np.arange(0,11,1))		
            HIDs_coi_GRID[ic,:] = n
            del n, bins, patches
        fig.savefig(options['fig_dir']+'GRIDDEDPPI_'+str(alt_z[nlev])+'km_contours.png', dpi=300, transparent=False) 	
        plt.close()
	#HIDs_coi_zgrid[nlev,:,:] = HIDs_coi_GRID[:,:]
        # And barplot ... 
        fig = plt.figure(figsize=(8,3)) 
        barlabels = []
    	# Set position of bar on X axis   
       	br1 = np.arange(len(name)) #---- adjutst!
       	bar1 = plt.bar(br1, HIDs_coi_GRID[0,:], color='darkblue',  width = barWidth, label='icoi: '+str(icois[0]))
        if len(RNgrid_inds_parallax) == 2:
            br2 = [x + barWidth for x in br1] 
            bar2 = plt.bar(br2, HIDs_coi_GRID[1,:], color='darkred',   width = barWidth, label='icoi: '+str(icois[1]))
            for rect in bar1 + bar2:
                height = rect.get_height()
                if height < 20:
                    height=np.nan;
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')
        if len(RNgrid_inds_parallax) == 3:
            br2 = [x + barWidth for x in br1]
            br3 = [x + barWidth for x in br2]	
            bar2 = plt.bar(br2, HIDs_coi_GRID[1,:], color='darkred',   width = barWidth, label='icoi: '+str(icois[1]))
            bar3 = plt.bar(br3, HIDs_coi_GRID[2,:], color='darkgreen', width = barWidth, label='icoi: '+str(icois[2]))
            for rect in bar1 + bar2 + bar3 :
                height = rect.get_height()
                if height < 20:
                    height=np.nan;
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')
        if len(RNgrid_inds_parallax) == 4:
            br2 = [x + barWidth for x in br1] 
            br3 = [x + barWidth for x in br2]
            br4 = [x + barWidth for x in br3]
            bar2 = plt.bar(br2, HIDs_coi_GRID[1,:], color='darkred',   width = barWidth, label='icoi: '+str(icois[1]))
            bar3 = plt.bar(br3, HIDs_coi_GRID[2,:], color='darkgreen', width = barWidth, label='icoi: '+str(icois[2]))
            bar4 = plt.bar(br4, HIDs_coi_GRID[3,:], color='black', width = barWidth, label='icoi: '+str(icois[3]))
            # Add counts above the two bar graph
            for rect in bar1 + bar2 + bar3 + bar4:
                height = rect.get_height()
                if height < 20:
                    height=np.nan;
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')
        if len(RNgrid_inds_parallax) == 1:
            for rect in bar1:
                height = rect.get_height()
                if height < 20:
                    height=np.nan;
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')
        plt.xlabel('HID')  
        plt.xlabel('HID counts')  
        plt.xticks([r + barWidth for r in range(len(name))], name)   # adjutst! len() 
        plt.legend()
        plt.title('HID for gridded radar at ' + str(alt_z[nlev])+' km')
        plt.grid(True)
        fig.savefig(options['fig_dir']+'RHIs_BARPLOT_gridded_'+str(alt_z[nlev])+'km_contours.png', dpi=300, transparent=False)  	
        #plt.close()
        
	
	
    # ----------------------------------------------------------------- 
    # Ahora con la grid entera pero sin graficos ... 
    #
    HIDs_coi_zgrid = np.zeros((len(grided.point_z['data'][:,0,0]), len(RNgrid_inds_parallax), 10)); HIDs_coi_zgrid[:]=np.nan
    #------------------------------------------------------	
    for nlev in range(len(grided.point_z['data'][:,0,0])):
        HIDs_coi_GRID = np.zeros((len(RNgrid_inds_parallax), 10)); HIDs_coi_GRID[:]=np.nan
        for ic in range(len(icois)):
            HIDS = np.ravel(grided.fields['HID']['data'][nlev,:,:])[RNgrid_inds_parallax[ic]]
            n, bins, patches = plt.hist(x=HIDS, bins=np.arange(0,11,1))		
            HIDs_coi_zgrid[nlev,ic,:] = n
            del n, bins, patches

	
    return check_resolxy, check_resolz, HIDs_coi_zgrid, HIDs_coi_nlev, grided.point_z['data'][:,0,0]



#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def CalcandPlot_HID_fraction(HIDs_coi_zgrid, icois_NR, grided_point_z, options):

    # AND ADD SUBPLOT WITH TBS DISTRIBUTION?
    f = h5py.File( gmi_dir+options['gfile'], 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]          
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    f.close()
    S1_sub_lat  = lat_gmi.copy()
    S1_sub_lon  = lon_gmi.copy()
    S1_sub_tb = tb_s1_gmi.copy()
	#
    idx1 = (lat_gmi>=options['ylim_min']-5) & (lat_gmi<=options['ylim_max']+5) & (lon_gmi>=options['xlim_min']-5) & (lon_gmi<=options['xlim_max']+5)
    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    #
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)
    #------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[14,12])
    contorno89 = plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [200] , colors=(['r']), linewidths=1.5);
    axes.set_xlim([options['xlim_min'], options['xlim_max']])
    axes.set_ylim([options['ylim_min'], options['ylim_max']])
    plt.close()
    #--------------------------------------------------------------------------------------
    datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] )) 
    #--------------------------------------------------------------------------------------
	
    linecolors = ['LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
           'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
           
    HID_fraction = np.zeros((HIDs_coi_zgrid.shape[0], HIDs_coi_zgrid.shape[1], HIDs_coi_zgrid.shape[2] ));  HID_fraction[:]=np.nan
    for nlev in range(HIDs_coi_zgrid.shape[0]):
        for icois in range(HIDs_coi_zgrid.shape[1]):
            for hid_id in range(HIDs_coi_zgrid.shape[2]):
                HID_fraction[nlev,icois,hid_id] = HIDs_coi_zgrid[nlev,icois,hid_id]/np.sum(HIDs_coi_zgrid[nlev,icois,:])
                
    for icois in range(HIDs_coi_zgrid.shape[1]):
        fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8,6))	
	
        axes[0].plot(HID_fraction[:,icois,8]*100, grided_point_z/1e3, linewidth=1.4, color=linecolors[8], label='Hail')
        axes[0].plot(HID_fraction[:,icois,7]*100+HID_fraction[:,icois,6]*100, grided_point_z/1e3, linewidth=1.4, color=linecolors[6], label='HD+LD graupel')
        axes[0].plot(HID_fraction[:,icois,5]*100+HID_fraction[:,icois,2]*100, grided_point_z/1e3, linewidth=1.4, color=linecolors[5], label='Ice')
        axes[0].plot(HID_fraction[:,icois,3]*100, grided_point_z/1e3, linewidth=1.4, color=linecolors[2], label='Agg.')
        axes[0].plot(HID_fraction[:,icois,9]*100, grided_point_z/1e3, linewidth=1.4, color=linecolors[9], label='BD')
        axes[0].set_ylabel('Altitude (km)')
        axes[0].set_xlabel('Hydrometeor Fraction')
        axes[0].set_title('Frozen-phase')
        axes[0].legend()
        axes[0].set_ylim([0,20])
        axes[0].set_xlim([0,40])
        axes[0].grid(True)


        axes[1].plot(HID_fraction[:,icois,4]*100, grided_point_z/1e3, linewidth=1.4, color=linecolors[4], label='Wet Snow')
        axes[1].plot(HID_fraction[:,icois,1]*100, grided_point_z/1e3, linewidth=1.4, color=linecolors[1], label='Rain')     
        axes[1].set_title('Liquid-Phase')
        axes[1].legend()
        axes[1].set_xlim([0,100])
        axes[1].set_ylim([0,20])
        axes[1].grid(True)
        #
        plt.subplots_adjust(wspace=0, hspace=0)
        axes[1].set_yticklabels([])
	
	#suptitle
        if icois_NR[icois] == options['icoi_PHAIL'][0]:
            plt.suptitle('icoi Nr. '+str(icois_NR[icois])  +', Phail: '+str(options['phail']) ) 

        else:
            plt.suptitle('icoi Nr. '+str(icois_NR[icois]))  
        plt.show()
	
    TB_inds = get_contour_info(contorno89, icois_NR, datapts)

    MIN19PCT = []
    MIN37PCT = []
    MIN89PCT = []

    for icois in range(HIDs_coi_zgrid.shape[1]):
        contour_pct19 = PCT19[idx1][TB_inds[icois]]
        contour_pct37 = PCT37[idx1][TB_inds[icois]]
        contour_pct89 = PCT89[idx1][TB_inds[icois]]
        
        n19, bins, patches19 = plt.hist(x=contour_pct19, bins=np.arange(0,300,1))	
        n37, bins, patches37 = plt.hist(x=contour_pct37, bins=np.arange(0,300,1))		
        n89, bins, patches89 = plt.hist(x=contour_pct89, bins=np.arange(0,300,1))		
        plt.close()

        # And barplot ... 
        fig = plt.figure(figsize=(8,3)) 
        barlabels = []
        barWidth = 1
        #
       	bar1 = plt.bar(bins[1:], n19, color='darkblue',  width = barWidth, label='PCT19')
       	bar2 = plt.bar(bins[1:], n37, color='darkred',   width = barWidth, label='PCT37')
        bar3 = plt.bar(bins[1:], n89, color='darkgreen', width = barWidth, label='PCT89')
        #
        plt.xlabel('PCT')  
        plt.ylabel('Counts')  
        plt.legend()
        plt.grid(True)
        #suptitle
        if icois_NR[icois] == options['icoi_PHAIL'][0]:
            plt.suptitle('icoi Nr. '+str(icois_NR[icois])  +', Phail: '+str(options['phail']) ) 

        else:
            plt.suptitle('icoi Nr. '+str(icois_NR[icois])) 
        plt.show()
        MIN19PCT.append(np.nanmin(contour_pct19))
        MIN37PCT.append(np.nanmin(contour_pct37))
        MIN89PCT.append(np.nanmin(contour_pct89))

    fig = plt.figure(figsize=(8,3)) 
    linestyle_coi = ['solid','dotted','dashed','dashdot']
    for icois in range(HIDs_coi_zgrid.shape[1]):
    	plt.plot(np.nan, np.nan, 'k' ,linestyle=linestyle_coi[icois], label='icoi Nr.'+str(icois_NR[icois]))
    plt.legend()
    plt.text(105,0.1,'MIN19PCT', color='darkblue',bbox=dict(boxstyle="square,pad=0.3", fc='None')) 
    plt.text(105,0.2,'MIN37PCT', color='darkred',bbox=dict(boxstyle="square,pad=0.3",fc='None'))
    plt.text(105,0.3,'MIN85PCT', color='darkgreen',bbox=dict(boxstyle="square,pad=0.3",fc='None'))

    for icois in range(HIDs_coi_zgrid.shape[1]):
        plt.axvline(x=MIN19PCT[icois],color='darkblue',linestyle=linestyle_coi[icois])
        plt.axvline(x=MIN37PCT[icois],color='darkred',linestyle=linestyle_coi[icois])
        plt.axvline(x=MIN89PCT[icois],color='darkgreen',linestyle=linestyle_coi[icois])
    plt.xlabel('PCT (K)')
    plt.ylim([0,1])
    plt.xlim([100, 280])

    return

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

    if options['radar_name'] == 'DOW7':
        alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel( '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'+options['era5_file'], options['lat_pfs'], options['lon_pfs']) 
        radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
        radar = stack_ppis(radar, options['files_list'], options, freezing_lev, radar_T, tfield_ref, alt_ref)
		

	
    plot_gmi(gmi_dir+options['gfile'], options, radar, lon_pfs, lat_pfs, icois)

   
    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, 
    # ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = plot_scatter_4icois_morethan1OFINTEREST(options, radar, icois, gmi_dir+options['gfile'])
    [ check_resolxy, check_resolz, HIDs_coi_zgrid, HIDs_coi_nlev, zgrid_alt] = plot_icois_HIDinfo(options, radar, icois, gmi_dir+options['gfile'])
	
    gc.collect()



    return [ check_resolxy, check_resolz, HIDs_coi_zgrid, HIDs_coi_nlev, zgrid_alt] 


    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, 
    #	    GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]
  
#------------------------------------------------------------------------------
def run_general_paper(options, lat_pfs, lon_pfs, icois, transects, labels_PHAIL,  xlims_mins_input, xlims_xlims_input):

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

    if options['radar_name'] == 'DOW7':
        alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel( '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'+options['era5_file'], options['lat_pfs'], options['lon_pfs']) 
        radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
        radar = stack_ppis(radar, options['files_list'], options, freezing_lev, radar_T, tfield_ref, alt_ref)
		

	
    plot_gmi_paper(gmi_dir+options['gfile'], options, radar, lon_pfs, lat_pfs, icois, transects, options['caso'])
    alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel(era5_dir, era5_file, lat_pfs, lon_pfs) 
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    radar = add_43prop_field(radar)     
    radar = correct_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)

    # 500m grid! 
    grided  = pyart.map.grid_from_radars(radar, grid_shape=(40, 940, 940), grid_limits=((0.,20000,),   #20,470,470 is for 1km
      		(-np.max(radar.range['data']), np.max(radar.range['data'])),(-np.max(radar.range['data']), np.max(radar.range['data']))),
					 roi_func='dist', min_radius=100.0, weighting_function='BARNES2')  
    gc.collect()
    #make_pseudoRHISfromGrid_4(grided, radar, azimuths_oi, labels_PHAIL, xlims_mins_input, xlims_xlims_input, alt_ref, tfield_ref, options)
    make_pseudoRHISfromGrid(grided, radar, transects, labels_PHAIL, xlims_mins_input, xlims_xlims_input, alt_ref, tfield_ref, options)
    gc.collect()
    return
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def run_general_paper_Figure_onlyHID(options, lat_pfs, lon_pfs, icois, transects, labels_PHAIL,  xlims_mins_input, xlims_xlims_input):

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

    if options['radar_name'] == 'DOW7':
	alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel( '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/',options['era5_file'],options['lat_pfs'], options['lon_pfs']) 
        radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
        radar = stack_ppis(radar, options['files_list'], options, freezing_lev, radar_T, tfield_ref, alt_ref)
		

	
    plot_gmi_paper_fig5(gmi_dir+options['gfile'], options, radar, lon_pfs, lat_pfs, icois, transects, options['caso'])
    alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel(era5_dir, era5_file, lat_pfs, lon_pfs) 
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    radar = add_43prop_field(radar)     
    radar = correct_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)

    # 500m grid! 
    grided  = pyart.map.grid_from_radars(radar, grid_shape=(40, 940, 940), grid_limits=((0.,20000,),   #20,470,470 is for 1km
      		(-np.max(radar.range['data']), np.max(radar.range['data'])),(-np.max(radar.range['data']), np.max(radar.range['data']))),
					 roi_func='dist', min_radius=100.0, weighting_function='BARNES2')  
    gc.collect()
    #make_pseudoRHISfromGrid_4(grided, radar, azimuths_oi, labels_PHAIL, xlims_mins_input, xlims_xlims_input, alt_ref, tfield_ref, options)
    make_pseudoRHISfromGrid(grided, radar, transects, labels_PHAIL, xlims_mins_input, xlims_xlims_input, alt_ref, tfield_ref, options)
    gc.collect()
    return

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def make_pseudoRHISfromGrid_4(gridded_radar, radar, azi_oi, titlecois, xlims_xlims_mins, xlims_xlims, alt_ref, tfield_ref, options): 

    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12  
    from scipy.ndimage import gaussian_filter
    
    if 'TH' in radar.fields.keys():  
            THname= 'TH'
            TVname= 'TV'
            KDPname='corrKDP'
            RHOHVname = 'RHOHV'
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
           RHOHVname   = 'RHOHV'  
		
    nlev = 0 
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    azimuths    = radar.azimuth['data'][start_index:end_index]
	
    #radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, gridded_radar)

    fig, axes = plt.subplots(nrows=4, ncols=4, constrained_layout=True, figsize=[15,10])

    for iz in range(len(azi_oi)):
        target_azimuth = azimuths[options['alternate_azi'][iz]]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()		
        if options['radar_name'] == 'CSPR2':
            del filas
            target_azimuth = azimuths[options['alternate_azi'][iz]]		
            filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
            axescheck.plot(lons[filas,:], lats[filas,:],'-k')
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
            #grid_HID[:,i]   = gridded_radar.fields['HID']['data'][:,xloc,yloc]
            # CALCUALTE HID FROM THESE GRIDDED FIELDS:
            scores = csu_fhc.csu_fhc_summer(dz=grid_TVTV[:,i], zdr=grid_ZDR[:,i] - options['ZDRoffset'], rho=grid_RHO[:,i], kdp=grid_KDP[:,i], 
					    use_temp=True, band='C', T=gridded_radar.fields['sounding_temperature']['data'][:,xloc,yloc])
            grid_HID[:,i] = np.argmax(scores, axis=0) + 1 
		
        ni = grid_HID.shape[0]
        nj = grid_HID.shape[1]
        for i in range(ni):
            rho_h = grid_RHO[i,:]
            zh_h = grid_THTH[i,:]
            for j in range(nj):
                if (rho_h[j]<0.7) or (zh_h[j]<10):
                    grid_THTH[i,j]  = np.nan
                    grid_TVTV[i,j]  = np.nan
                    grid_RHO[i,j]  = np.nan	
                    grid_ZDR[i,j]  = np.nan
                    grid_HID[i,j]  = np.nan


        #---- plot hid ppi  
        hid_colors = ['White', 'LightBlue','MediumBlue', 'DarkOrange', 'LightPink',
                'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
        cmaphid = colors.ListedColormap(hid_colors)
        # Figure
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        im_TH  = axes[0,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_THTH, cmap=cmap, vmin=vmin, vmax=vmax)

        #im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, (grid_THTH-grid_TVTV)-options['ZDRoffset'], cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)
        im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, (grid_ZDR)-options['ZDRoffset'], cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)

        im_RHO = axes[2,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_RHO, cmap=pyart.graph.cm.RefDiff , vmin=0.7, vmax=1.)

        im_HID = axes[3,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=0.2, vmax=10)

        axes[0,iz].set_title(titlecois[iz]) 
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
            axes[0,0].set_ylabel('Altitude (km)', fontsize=12)
            axes[1,0].set_ylabel('Altitude (km)', fontsize=12)
            axes[2,0].set_ylabel('Altitude (km)', fontsize=12)
            axes[3,0].set_ylabel('Altitude (km)', fontsize=12)
            axes[0,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[1,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[2,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,0].set_xlabel('Range (km)', fontsize=12)
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
            cbar    = fig.colorbar(im_RHO, cax=ax_cbar, shrink=0.9, ticks=np.arange(0.7,1.01,0.1), label='RHOHV')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.03, 0.02, 0.2])  
            cbar    = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            cbar = adjust_fhc_colorbar_for_pyart(cbar)
            #cbar.cmap.set_under('white')

            pm2    = axes[3,iz-1].get_position().get_points().flatten()


    #- savefile
    fig.savefig(options['fig_dir']+'PseudoRHIS_GRIDDED'+'.png', dpi=300,transparent=False)   
    #plt.close()
    del grid_THTH, grid_RHO, grid_TVTV, grid_HID
    return
	
	
#------------------------------------------------------------------------------
def make_pseudoRHISfromGrid(gridded_radar, radar, azi_oi, titlecois, xlims_xlims_mins, xlims_xlims, alt_ref, tfield_ref, options): 

    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12  
    from scipy.ndimage import gaussian_filter
    
    if 'TH' in radar.fields.keys():  
            THname= 'TH'
            TVname= 'TV'
            KDPname='corrKDP'
            RHOHVname = 'RHOHV'
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
           RHOHVname   = 'RHOHV'  
		
    nlev = 0 
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    azimuths    = radar.azimuth['data'][start_index:end_index]
	
    #radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, gridded_radar)

    fig, axes = plt.subplots(nrows=4, ncols=3, constrained_layout=True, figsize=[15,10])

    figcheck, axescheck = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[15,10])

    for iz in range(len(azi_oi)):
        target_azimuth = azimuths[options['alternate_azi'][iz]]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()		
        if options['radar_name'] == 'CSPR2':
            del filas
            target_azimuth = azimuths[options['alternate_azi'][iz]]		
            filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
            axescheck.plot(lons[filas,:], lats[filas,:],'-k')
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
            #grid_HID[:,i]   = gridded_radar.fields['HID']['data'][:,xloc,yloc]
            # CALCUALTE HID FROM THESE GRIDDED FIELDS:
            scores = csu_fhc.csu_fhc_summer(dz=grid_TVTV[:,i], zdr=grid_ZDR[:,i] - options['ZDRoffset'], rho=grid_RHO[:,i], kdp=grid_KDP[:,i], 
					    use_temp=True, band='C', T=gridded_radar.fields['sounding_temperature']['data'][:,xloc,yloc])
            grid_HID[:,i] = np.argmax(scores, axis=0) + 1 
		

		
        ni = grid_HID.shape[0]
        nj = grid_HID.shape[1]
        for i in range(ni):
            rho_h = grid_RHO[i,:]
            zh_h = grid_THTH[i,:]
            for j in range(nj):
                if (rho_h[j]<0.7) or (zh_h[j]<10):
                    grid_THTH[i,j]  = np.nan
                    grid_TVTV[i,j]  = np.nan
                    grid_RHO[i,j]  = np.nan	
                    grid_ZDR[i,j]  = np.nan
                    grid_HID[i,j]  = np.nan
            #grid_HID[i,:] = gaussian_filter(grid_HID[i,:], sigma=1) 
            #grid_ZDR[i,:] = gaussian_filter(grid_ZDR[i,:], sigma=1) 
            #grid_RHO[i,:] = gaussian_filter(grid_RHO[i,:], sigma=1) 
            #grid_THTH[i,:] = gaussian_filter(grid_THTH[i,:], sigma=1) 

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

        axes[0,iz].set_title(titlecois[iz]) 
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
            axes[0,0].set_ylabel('Altitude (km)', fontsize=12)
            axes[1,0].set_ylabel('Altitude (km)', fontsize=12)
            axes[2,0].set_ylabel('Altitude (km)', fontsize=12)
            axes[3,0].set_ylabel('Altitude (km)', fontsize=12)
            axes[0,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[1,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[2,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,0].set_xlabel('Range (km)', fontsize=12)
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
            cbar    = fig.colorbar(im_RHO, cax=ax_cbar, shrink=0.9, ticks=np.arange(0.7,1.01,0.1), label='RHOHV')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

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
	
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def run_general_case_startconv(options, lat_pfs, lon_pfs, icois):

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

    if options['radar_name'] == 'DOW7':
        alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel( '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'+options['era5_file'], options['lat_pfs'], options['lon_pfs']) 
        radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
        radar = stack_ppis(radar, options['files_list'], options, freezing_lev, radar_T, tfield_ref, alt_ref)
		

	
    plot_gmi_paper(gmi_dir+options['gfile'], options, radar, lon_pfs, lat_pfs, icois)

   
    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, 
    # ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = plot_scatter_4icois_morethan1OFINTEREST(options, radar, icois, gmi_dir+options['gfile'])
    [ check_resolxy, check_resolz, HIDs_coi_zgrid, HIDs_coi_nlev, zgrid_alt] = plot_icois_HIDinfo(options, radar, icois, gmi_dir+options['gfile'])
	
    gc.collect()



    return [ check_resolxy, check_resolz, HIDs_coi_zgrid, HIDs_coi_nlev, zgrid_alt] 

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def run_general_case_RELAMPAGO(options, lat_pfs, lon_pfs, icois):

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    r_dir    = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'
    radar = pyart.io.read(r_dir+options['rfile'])
            
    plot_gmi(gmi_dir+options['gfile'], options, radar, lon_pfs, lat_pfs, icois)
   
    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, 
    # ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = plot_scatter_4icois_morethan1OFINTEREST(options, radar, icois, gmi_dir+options['gfile'])
    [ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi] = plot_icois_HIDinfo(options, radar, icois, gmi_dir+options['gfile'])
	
    gc.collect()



    return [ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi] 

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, 
    #	    GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]
  
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


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------     
def get_median(y, x, tbvbin):

    TBV_bin  = np.arange(50,300,tbvbin)
    TBVH_bin = np.arange(-5,25,1)
    figfig = plt.figure(figsize=(30,10))
    [counts, xedges, yedges, _] = plt.hist2d( x,
        y, bins=[TBV_bin, TBVH_bin], norm=matplotlib.colors.LogNorm())
    xbins = np.digitize( x, xedges[1:-1])
    running_median = [np.median(y[xbins==k]) for k in range(len(xedges))]
    del counts, xedges, yedges
    plt.close()
    return [running_median]

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def run_general_paper_Figure_FIG5(opts_CSPR2, opts_DOW7, opts_RMA1, opts_RMA5):

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'	
    r_dir    = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'

    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12  

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

    #----------------------------------------------------------------------------------------
    # NEW FIGURE. solo dos paneles: Same as above but plt lowest level y closest to freezing level!
    #----------------------------------------------------------------------------------------
    #---- PRIMERO INFO DE TBS
    fname = gmi_dir+opts_CSPR2['gfile']
    #
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
    idx1 = (lat_gmi>=opts_CSPR2['ylim_min']-5) & (lat_gmi<=opts_CSPR2['ylim_max']+5) & (lon_gmi>=opts_CSPR2['xlim_min']-5) & (lon_gmi<=opts_CSPR2['xlim_max']+5)
    idx2 = (lat_s2_gmi>=opts_CSPR2['ylim_min']-5) & (lat_s2_gmi<=opts_CSPR2['ylim_max']+5) & (lon_s2_gmi>=opts_CSPR2['xlim_min']-5) & (lon_s2_gmi<=opts_CSPR2['xlim_max']+5)
    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    S2_sub_lat = np.where(idx2 != False, S2_sub_lat, np.nan) 
    S2_sub_lon = np.where(idx2 != False, S2_sub_lon, np.nan) 
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    for i in range(tb_s2_gmi.shape[2]):
        S2_sub_tb[:,:,i]  = np.where(np.isnan(S2_sub_lon) != 1, tb_s2_gmi[:,:,i], np.nan)	
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)

    # ---- FIGFIG ---------------------------------------------------------------
    fig, axes = plt.subplots(nrows=4, ncols=2, constrained_layout=True,figsize=[8,11])
    # ---- ZH ---------------------------------------------------------------
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    nlev = 0 
    # CASO 20181111 #----------------------------------------------------
    # ---- ZH ---------------------------------------------------------------
    radar = pyart.io.read(r_dir+opts_CSPR2['rfile'])
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]	
    reflectivity_name = 'corrected_reflectivity'   
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]
    ZH[np.where(ZH<20)]=np.nan
    #
    pcm1=axes[0,0].pcolormesh(lons, lats, ZH, cmap=cmap, vmax=vmax, vmin=vmin)
    axes[0,0].set_title('Ground Level')
    axes[0,0].set_xlim([opts_CSPR2['xlim_min'], opts_CSPR2['xlim_max']])
    axes[0,0].set_ylim([opts_CSPR2['ylim_min'], opts_CSPR2['ylim_max']])
    plt.colorbar(pcm1, ax=axes[0,0],label='Zh (dBZ)')
    # CONTORNO CORREGIDO POR PARALAJE Y PODER CORRER LOS ICOIS, simplemente pongo nans fuera del area de interes ... 
    contorno89 = axes[0,0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5); 
    axes[0,0].set_xlim([opts_CSPR2['xlim_min'], opts_CSPR2['xlim_max']])
    axes[0,0].set_ylim([opts_CSPR2['ylim_min'], opts_CSPR2['ylim_max']])
    # Add labels:
    labels = ["PCT 89-GHz 200 K contour"] 
    for i in range(len(labels)):
        contorno89.collections[i].set_label(labels[i])
    # Titles and legend:
    axes[0,0].set_title(r'CSPR2 11/11/2018')
    #axes[0,0].legend(loc='upper left')
    # RADAR RINGS	      
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    CurvedText(
            x = lon_radius[25000:],
            y = lat_radius[25000:],
            text='50 km',#'this this is a very, very long text',
            va = 'top', axes=axes[0,0])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    CurvedText(
            x = lon_radius[30000:],
            y = lat_radius[30000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[0,0])
    if len(opts_CSPR2['REPORTES_meta'])>0:
        for ireportes in range(len(opts_CSPR2['REPORTES_geo'])):
            axes[0,0].plot( opts_CSPR2['REPORTES_geo'][ireportes][1],  opts_CSPR2['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, label=opts_DOW7['REPORTES_meta'][ireportes])

    # ---- PCT ---------------------------------------------------------------
    im = axes[0,1].scatter(lon_gmi, lat_gmi, c=PCT89, marker='h', s=150, vmin=100, vmax=300, cmap=cmaps['turbo_r'])  
    axes[0,1].set_title('PCT 89-GHz')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    contorno89 = axes[0,1].contour(lon_gmi, lat_gmi, PCT89, [200], colors=('k'), linewidths=2);    
    for i in range(len(opts_CSPR2['lon_pfs'])):
        axes[0,1].plot(opts_CSPR2['lon_pfs'][i], opts_CSPR2['lat_pfs'][i]-0.05, marker='*', markersize=20, markerfacecolor="black", 
		       markeredgecolor='black', markeredgewidth=1.5)	
        axes[0,0].plot(opts_CSPR2['lon_pfs'][i], opts_CSPR2['lat_pfs'][i]-0.05, marker='*', markersize=20, markerfacecolor="black", 
		       markeredgecolor='black', markeredgewidth=1.5)		
    if len(opts_CSPR2['REPORTES_meta'])>0:
        for ireportes in range(len(opts_CSPR2['REPORTES_geo'])):
            axes[0,1].plot( opts_CSPR2['REPORTES_geo'][ireportes][1],  opts_CSPR2['REPORTES_geo'][ireportes][0], 'D', 
			   markeredgecolor='black', markerfacecolor='none', markersize=10, label=opts_CSPR2['REPORTES_meta'][ireportes])
    axes[0,1].set_xlabel('Longitude')
    axes[0,1].set_ylabel('Latitude')	
    axes[0,1].set_xlim([opts_CSPR2['xlim_min'], opts_CSPR2['xlim_max']])
    axes[0,1].set_ylim([opts_CSPR2['ylim_min'], opts_CSPR2['ylim_max']])	
    #fig.savefig(opts_CSPR2['fig_dir']+'GMI_icois_onZH.png', dpi=300, transparent=False)  
    #plt.close()
    plt.colorbar(im, ax=axes[0,1],label='(K)')
    # ---- in both:
    azimuths = radar.azimuth['data'][start_index:end_index]	
    for itrans in opts_CSPR2['transects']:
        target_azimuth = azimuths[itrans]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect     = lons[filas,:]
        lat_transect     = lats[filas,:]
        axes[0,0].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')
        axes[0,1].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')

    # CASO 20181214 #----------------------------------------------------
    fname = gmi_dir+opts_DOW7['gfile']
    #
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
    idx1 = (lat_gmi>=opts_DOW7['ylim_min']-5) & (lat_gmi<=opts_DOW7['ylim_max']+5) & (lon_gmi>=opts_DOW7['xlim_min']-5) & (lon_gmi<=opts_DOW7['xlim_max']+5)
    idx2 = (lat_s2_gmi>=opts_DOW7['ylim_min']-5) & (lat_s2_gmi<=opts_DOW7['ylim_max']+5) & (lon_s2_gmi>=opts_DOW7['xlim_min']-5) & (lon_s2_gmi<=opts_DOW7['xlim_max']+5)
    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    S2_sub_lat = np.where(idx2 != False, S2_sub_lat, np.nan) 
    S2_sub_lon = np.where(idx2 != False, S2_sub_lon, np.nan) 
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    for i in range(tb_s2_gmi.shape[2]):
        S2_sub_tb[:,:,i]  = np.where(np.isnan(S2_sub_lon) != 1, tb_s2_gmi[:,:,i], np.nan)	
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)

    # ---- ZH ---------------------------------------------------------------
    radar = pyart.io.read(r_dir+opts_DOW7['rfile'])
    reflectivity_name = 'DBZHCC'   
    lats        = radar.gate_latitude['data']
    lons        = radar.gate_longitude['data']
    ZH          = radar.fields[reflectivity_name]['data']
    ZH[np.where(ZH<20)]=np.nan
    #
    pcm1=axes[1,0].pcolormesh(lons, lats, ZH, cmap=cmap, vmax=vmax, vmin=vmin)
    axes[1,0].set_title('Ground Level')
    axes[1,0].set_xlim([opts_DOW7['xlim_min'], opts_DOW7['xlim_max']])
    axes[1,0].set_ylim([opts_DOW7['ylim_min'], opts_DOW7['ylim_max']])
    plt.colorbar(pcm1, ax=axes[1,0],label='Zh (dBZ)')
    # CONTORNO CORREGIDO POR PARALAJE Y PODER CORRER LOS ICOIS, simplemente pongo nans fuera del area de interes ... 
    contorno89 = axes[1,0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5); 
    # Add labels:
    labels = ["PCT 89-GHz 200 K contour"] 
    for i in range(len(labels)):
        contorno89.collections[i].set_label(labels[i])
    # Titles and legend:
    axes[1,0].set_title(r'DOW7 14/12/2018')
    #axes[1,0].legend(loc='upper left')
    # RADAR RINGS	      
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    CurvedText(
            x = lon_radius[25000:],
            y = lat_radius[25000:],
            text='50 km',#'this this is a very, very long text',
            va = 'top', axes=axes[1,0])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    CurvedText(
            x = lon_radius[30000:],
            y = lat_radius[30000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[1,0])
    if len(opts_DOW7['REPORTES_meta'])>0:
        for ireportes in range(len(opts_DOW7['REPORTES_geo'])):
            axes[1,0].plot( opts_DOW7['REPORTES_geo'][ireportes][1],  opts_DOW7['REPORTES_geo'][ireportes][0], 'D', 
			   markeredgecolor='black', markerfacecolor='none', markersize=10, label=opts_DOW7['REPORTES_meta'][ireportes])
    # ---- PCT ---------------------------------------------------------------
    im = axes[1,1].scatter(lon_gmi, lat_gmi, c=PCT89, marker='h', s=150, vmin=100, vmax=300, cmap=cmaps['turbo_r'])  
    axes[1,1].set_title('PCT 89-GHz')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    contorno89 = axes[1,1].contour(lon_gmi, lat_gmi, PCT89, [200], colors=('k'), linewidths=2);    
    print(opts_DOW7['lat_pfs'])


    axes[1,1].plot(opts_DOW7['lon_pfs'][0]-1, opts_DOW7['lat_pfs'][0]+0.5, marker='*', markersize=20, markerfacecolor="black", markeredgecolor='black', markeredgewidth=1.5)	
    axes[1,0].plot(opts_DOW7['lon_pfs'][0]-1, opts_DOW7['lat_pfs'][0]+0.5, marker='*', markersize=20, markerfacecolor="black", markeredgecolor='black', markeredgewidth=1.5)

    if len(opts_DOW7['REPORTES_meta'])>0:
        for ireportes in range(len(opts_DOW7['REPORTES_geo'])):
            axes[1,1].plot( opts_DOW7['REPORTES_geo'][ireportes][1],  opts_DOW7['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, label=opts_DOW7['REPORTES_meta'][ireportes])
    axes[1,1].set_xlim([opts_DOW7['xlim_min'], opts_DOW7['xlim_max']])
    axes[1,1].set_ylim([opts_DOW7['ylim_min'], opts_DOW7['ylim_max']])	
    plt.colorbar(im, ax=axes[1,1],label='(K)')
    # ---- in both:
    azimuths = radar.azimuth['data']
    for itrans in opts_DOW7['transects']:
        target_azimuth = azimuths[itrans]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect     = lons[filas,:]
        lat_transect     = lats[filas,:]
        axes[1,0].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')
        axes[1,1].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')

    # CASO 20190308 #----------------------------------------------------
    del fname
    fname = gmi_dir+opts_RMA1['gfile'][0]
    #
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
    idx1 = (lat_gmi>=opts_RMA1['ylim_min']-5) & (lat_gmi<=opts_RMA1['ylim_max']+5) & (lon_gmi>=opts_RMA1['xlim_min']-5) & (lon_gmi<=opts_RMA1['xlim_max']+5)
    idx2 = (lat_s2_gmi>=opts_RMA1['ylim_min']-5) & (lat_s2_gmi<=opts_RMA1['ylim_max']+5) & (lon_s2_gmi>=opts_RMA1['xlim_min']-5) & (lon_s2_gmi<=opts_RMA1['xlim_max']+5)
    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    S2_sub_lat = np.where(idx2 != False, S2_sub_lat, np.nan) 
    S2_sub_lon = np.where(idx2 != False, S2_sub_lon, np.nan) 
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    for i in range(tb_s2_gmi.shape[2]):
        S2_sub_tb[:,:,i]  = np.where(np.isnan(S2_sub_lon) != 1, tb_s2_gmi[:,:,i], np.nan)	
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)

    # ---- ZH ---------------------------------------------------------------
    radar = pyart.io.read(r_dir+opts_RMA1['rfile'])
    reflectivity_name = 'TH'   
    nlev = 0 
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]
    ZH[np.where(ZH<20)]=np.nan
    #
    pcm1=axes[2,0].pcolormesh(lons, lats, ZH, cmap=cmap, vmax=vmax, vmin=vmin)
    axes[2,0].set_title('Ground Level')
    axes[2,0].set_xlim([opts_RMA1['xlim_min'], opts_RMA1['xlim_max']])
    axes[2,0].set_ylim([opts_RMA1['ylim_min'], opts_RMA1['ylim_max']])
    plt.colorbar(pcm1, ax=axes[2,0],label='Zh (dBZ)')
    # CONTORNO CORREGIDO POR PARALAJE Y PODER CORRER LOS ICOIS, simplemente pongo nans fuera del area de interes ... 
    contorno89 = axes[2,0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5); 
    axes[2,0].set_xlim([opts_RMA1['xlim_min'], opts_RMA1['xlim_max']])
    axes[2,0].set_ylim([opts_RMA1['ylim_min'], opts_RMA1['ylim_max']])
    # Add labels:
    labels = ["PCT 89-GHz 200 K contour"] 
    for i in range(len(labels)):
        contorno89.collections[i].set_label(labels[i])
    # Titles and legend:
    axes[2,0].set_title(r'RMA1 03/08/2019')
    #axes[2,0].legend(loc='lower left')
    # RADAR RINGS	      
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[2,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    CurvedText(
            x = lon_radius[25000:],
            y = lat_radius[25000:],
            text='50 km',#'this this is a very, very long text',
            va = 'top', axes=axes[2,0])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[2,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    CurvedText(
            x = lon_radius[30000:],
            y = lat_radius[30000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[2,0])
    if len(opts_RMA1['REPORTES_meta'])>0:
        for ireportes in range(len(opts_RMA1['REPORTES_geo'])):
            axes[2,0].plot( opts_RMA1['REPORTES_geo'][ireportes][1],  opts_RMA1['REPORTES_geo'][ireportes][0], 'D', 
			   markeredgecolor='black', markerfacecolor='none', markersize=10, label=opts_RMA1['REPORTES_meta'][ireportes])

    # ---- PCT ---------------------------------------------------------------
    im = axes[2,1].scatter(lon_gmi, lat_gmi, c=PCT89, marker='h', s=100, vmin=100, vmax=300, cmap=cmaps['turbo_r'])  
    axes[2,1].set_title('PCT 89-GHz')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[2,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[2,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    contorno89 = axes[2,1].contour(lon_gmi, lat_gmi, PCT89, [200], colors=('k'), linewidths=2);    
    for i in range(len(opts_RMA1['lon_pfs'])):
        axes[2,0].plot(opts_RMA1['lon_pfs'][i], opts_RMA1['lat_pfs'][i]-0.05, marker='*', markersize=20, markerfacecolor="black", markeredgecolor='black', markeredgewidth=1.5)
        axes[2,1].plot(opts_RMA1['lon_pfs'][i], opts_RMA1['lat_pfs'][i]-0.05, marker='*', markersize=20, markerfacecolor="black", 
		       markeredgecolor='black', markeredgewidth=1.5)	
    if len(opts_RMA1['REPORTES_meta'])>0:
        for ireportes in range(len(opts_RMA1['REPORTES_geo'])):
            axes[2,1].plot( opts_RMA1['REPORTES_geo'][ireportes][1],  opts_RMA1['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', markerfacecolor='none', markersize=10, 
			   label=opts_RMA1['REPORTES_meta'][ireportes])
    axes[2,1].set_xlim([opts_RMA1['xlim_min'], opts_RMA1['xlim_max']])
    axes[2,1].set_ylim([opts_RMA1['ylim_min'], opts_RMA1['ylim_max']])	
    plt.colorbar(im, ax=axes[2,1],label='(K)')
    # ---- in both:
    azimuths = radar.azimuth['data'][start_index:end_index]
    for itrans in opts_RMA1['transects']:
        target_azimuth = azimuths[itrans]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect     = lons[filas,:]
        lat_transect     = lats[filas,:]
        axes[2,0].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')
        axes[2,1].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')


    # CASO 20200815 #----------------------------------------------------
    fname = gmi_dir+opts_RMA5['gfile']
    #
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
    idx1 = (lat_gmi>=opts_RMA5['ylim_min']-5) & (lat_gmi<=opts_RMA5['ylim_max']+5) & (lon_gmi>=opts_RMA5['xlim_min']-5) & (lon_gmi<=opts_RMA5['xlim_max']+5)
    idx2 = (lat_s2_gmi>=opts_RMA5['ylim_min']-5) & (lat_s2_gmi<=opts_RMA5['ylim_max']+5) & (lon_s2_gmi>=opts_RMA5['xlim_min']-5) & (lon_s2_gmi<=opts_RMA5['xlim_max']+5)
    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    S2_sub_lat = np.where(idx2 != False, S2_sub_lat, np.nan) 
    S2_sub_lon = np.where(idx2 != False, S2_sub_lon, np.nan) 
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	
    for i in range(tb_s2_gmi.shape[2]):
        S2_sub_tb[:,:,i]  = np.where(np.isnan(S2_sub_lon) != 1, tb_s2_gmi[:,:,i], np.nan)	
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)

    # ---- ZH ---------------------------------------------------------------
    radar = pyart.io.read(r_dir+opts_RMA5['rfile'])
    nlev = 0
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
    reflectivity_name = 'DBZH'   
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    ZH          = radar.fields[reflectivity_name]['data'][start_index:end_index]
    ZH[np.where(ZH<20)]=np.nan
    #
    pcm1=axes[3,0].pcolormesh(lons, lats, ZH, cmap=cmap, vmax=vmax, vmin=vmin)
    axes[3,0].set_title('Ground Level')
    axes[3,0].set_xlim([opts_RMA5['xlim_min'], opts_RMA5['xlim_max']])
    axes[3,0].set_ylim([opts_RMA5['ylim_min'], opts_RMA5['ylim_max']])
    plt.colorbar(pcm1, ax=axes[3,0],label='Zh (dBZ)')
    # CONTORNO CORREGIDO POR PARALAJE Y PODER CORRER LOS ICOIS, simplemente pongo nans fuera del area de interes ... 
    contorno89 = axes[3,0].contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:], [200], colors=(['k']), linewidths=1.5); 
    axes[3,0].set_xlim([opts_RMA5['xlim_min'], opts_RMA5['xlim_max']])
    axes[3,0].set_ylim([opts_RMA5['ylim_min'], opts_RMA5['ylim_max']])
    # Add labels:
    labels = ["PCT 89-GHz 200 K contour"] 
    for i in range(len(labels)):
        contorno89.collections[i].set_label(labels[i])
    # Titles and legend:
    axes[3,0].set_title(r'RMA5 15/08/2020')
    axes[3,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=False, ncol=1)
    # RADAR RINGS	      
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[3,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    CurvedText(
            x = lon_radius[25000:],
            y = lat_radius[25000:],
            text='50 km',#'this this is a very, very long text',
            va = 'top', axes=axes[3,0])
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[3,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    CurvedText(
            x = lon_radius[30000:],
            y = lat_radius[30000:],
            text='100 km',#'this this is a very, very long text',
            va = 'bottom', axes=axes[3,0])
    if len(opts_RMA5['REPORTES_meta'])>0:
        for ireportes in range(len(opts_RMA5['REPORTES_geo'])):
            axes[3,0].plot( opts_RMA5['REPORTES_geo'][ireportes][1],  opts_RMA5['REPORTES_geo'][ireportes][0], 'D', 
			   markeredgecolor='black', markerfacecolor='none', markersize=10, label=opts_RMA5['REPORTES_meta'][ireportes])

    # ---- PCT ---------------------------------------------------------------
    im = axes[3,1].scatter(lon_gmi, lat_gmi, c=PCT89, marker='h', s=100, vmin=100, vmax=300, cmap=cmaps['turbo_r'])  
    axes[3,1].set_title('PCT 89-GHz')
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[3,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[3,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8) 
    contorno89 = axes[3,1].contour(lon_gmi, lat_gmi, PCT89, [200], colors=('k'), linewidths=2);    
    for i in range(len(opts_RMA5['lon_pfs'])):
        axes[3,0].plot(opts_RMA5['lon_pfs'][i], opts_RMA5['lat_pfs'][i]-0.05, marker='*', markersize=20, markerfacecolor="black", markeredgecolor='black', markeredgewidth=1.5)
        axes[3,1].plot(opts_RMA5['lon_pfs'][i], opts_RMA5['lat_pfs'][i]-0.05, marker='*', markersize=20, markerfacecolor="black", 
		       markeredgecolor='black', markeredgewidth=1.5)	
    if len(opts_RMA5['REPORTES_meta'])>0:
        for ireportes in range(len(opts_RMA5['REPORTES_geo'])):
            axes[3,1].plot( opts_RMA5['REPORTES_geo'][ireportes][1],  opts_RMA5['REPORTES_geo'][ireportes][0], 'D', markeredgecolor='black', 
			   markerfacecolor='none', markersize=10, label=opts_RMA5['REPORTES_meta'][ireportes])
    axes[3,1].set_xlim([opts_RMA5['xlim_min'], opts_RMA5['xlim_max']])
    axes[3,1].set_ylim([opts_RMA5['ylim_min'], opts_RMA5['ylim_max']])	
    plt.colorbar(im, ax=axes[3,1],label='(K)')
    # ---- in both:
    azimuths = radar.azimuth['data'][start_index:end_index]
    for itrans in opts_RMA5['transects']:
        target_azimuth = azimuths[itrans]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect     = lons[filas,:]
        lat_transect     = lats[filas,:]
        axes[3,0].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')
        axes[3,1].plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k', linestyle='--')


    axes[3,0].set_xlabel('Longitude')
    axes[3,0].set_ylabel('Latitude')	

    return

#----------------------------------------------------------------------------------------------     
#----------------------------------------------------------------------------------------------     
def main_fig5():

    gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
    era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'

    # CSPR2
    lon_pfs  = [-64.53]
    lat_pfs  = [-31.83]
    time_pfs = ['1250UTC']
    phail    = [0.653]
    rfile     = 'corcsapr2cmacppiM1.c1.20181111.130003.nc' #'corcsapr2cfrppiM1.a1.20181111.130003.nc'
    gfile     = '1B.GPM.GMI.TB2016.20181111-S113214-E130446.026724.V05A.HDF5'
    era5_file = '20181111_13_RMA1.grib'
    reportes_granizo_twitterAPI_geo = [[]]
    reportes_granizo_twitterAPI_meta = []
    opts_cspr2 = {'xlim_min': -65, 'xlim_max': -63.6, 'ylim_min': -32.6, 'ylim_max': -31.5, 
	    'azimuth_ray': 0,
	    'rfile': 'CSPR2_data/'+rfile, 
   	    'era5_file': era5_file,'caso':'20181111',
	    'gfile': gfile,
    	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181111am/', 
	    'alternate_azi':[30], 'transects': [205],
	    'ZDRoffset': 0,
    	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
     	   'icoi_PHAIL': [3], 'radar_name':'CSPR2', 'xlims_xlims_input' : [60], 'xlims_mins_input' : [20]}
    del lon_pfs, lat_pfs, phail, rfile, gfile, era5_file, reportes_granizo_twitterAPI_geo, reportes_granizo_twitterAPI_meta
	
    # DOW		
    lon_pfs  = [-63.11] # [-61.40] [-59.65]
    lat_pfs  = [-31.90] # [-32.30] [-33.90]
    phail    = [0.967] # [0.998] [0.863]
    rfile = 'cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc' 
    gfile     = '1B.GPM.GMI.TB2016.20181214-S015009-E032242.027231.V05A.HDF5'
    era5_file = '20181214_03_RMA1.grib'
    reportes_granizo_twitterAPI_geo = [[-32.19, -64.57],[-32.07, -64.54]]
    reportes_granizo_twitterAPI_meta = [['0320UTC'],['0100']]
    opts_DOW7 = {'xlim_min': -65.3, 'xlim_max': -63.3, 'ylim_min': -32.4, 'ylim_max': -31, 'ZDRoffset': 0, 'caso':'20181214',
	    'rfile': 'DOW7/'+rfile, 'gfile': gfile, 'azimuth_ray': 0,
	     'radar_name':'DOW7', 'era5_file': era5_file,'alternate_azi':[30], 'ZDRoffset': 0, 'transects': [270],
	     'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181214_RMA1/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	     'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [15], 'files_list':files_list, 'xlims_xlims_input' : [80], 'xlims_mins_input' : [20]}
    del lon_pfs, lat_pfs, phail, rfile, gfile, era5_file, reportes_granizo_twitterAPI_geo, reportes_granizo_twitterAPI_meta

    # RMA1
    lon_pfs  = [-63.74]
    lat_pfs  = [-30.75]
    time_pfs = ['0204UTC']
    phail    = [0.895]
    rfile    = 'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc'
    gfile     = '1B.GPM.GMI.TB2016.20190308-S004613-E021846.028537.V05A.HDF5'
    era5_file = '20190308_02_RMA1.grib'
    reportes_granizo_twitterAPI_geo = [[]]
    reportes_granizo_twitterAPI_meta = []
    opts_RMA1 = {'xlim_min': -65.2, 'xlim_max': -62, 'ylim_min': -32, 'ylim_max': -30, 
    	    'ZDRoffset': 0.5, 'rfile': 'RMA1/'+rfile, 'gfile': gfile, 
    	    'window_calc_KDP': 7, 'era5_file': era5_file, 'azimuth_ray': 50,
    	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190308/', 'transects': [50],
    	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
     	   'icoi_PHAIL': [3], 'radar_name':'RMA1', 'xlims_xlims_input' : [200], 'xlims_mins_input' : [0]}
    del lon_pfs, lat_pfs, phail, rfile, gfile, era5_file, reportes_granizo_twitterAPI_geo, reportes_granizo_twitterAPI_meta


    # rma5		
    lon_pfs  = [ -54.11 ]
    lat_pfs  = [ -25.28 ]
    time_pfs = ['0215UTC']
    phail    = [ 0.725 ]
    rfile    = 'cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc' 
    gfile    = '1B.GPM.GMI.TB2016.20200815-S015947-E033219.036720.V05A.HDF5'
    era5_file = '20200815_02.grib'
    reportes_granizo_twitterAPI_geo = [[-25.93, -54.57], [-27.03, -55.24], [-25.62, -53.77], [-26.07,-53.05],
				       [-25.67,-53.81],[-25.66,-53.55],[-26.24,-52.71],[-25.28,-53.95]] 
    reportes_granizo_twitterAPI_meta = ['Wanda', 'Jardin de America', '','','','','','']
    opts_RMA5 = {'xlim_min': -55.0, 'xlim_max': -52.0, 'ylim_min': -27.5, 'ylim_max': -25.0, 
	    'ZDRoffset': 2, 'rfile': 'RMA5/'+rfile, 'gfile': gfile, 'azimuth_ray': 50,
	    'window_calc_KDP': 7, 'era5_file': era5_file,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20200815_RMA5/','transects': [331],
    	    'alternate_azi':[331],'caso':'20200815',
	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [7], 'radar_name':'RMA5','xlims_xlims_input' : [190], 'xlims_mins_input' : [0]}
    del lon_pfs, lat_pfs, phail, rfile, gfile, era5_file, reportes_granizo_twitterAPI_geo, reportes_granizo_twitterAPI_meta

	
    run_general_paper_Figure_FIG5(opts_cspr2, opts_DOW7, opts_RMA1, opts_RMA5)
	
	
	

	
	
	return

def main_main(): 
	
	main_20180208() 
	#main_DOW7_20181214()  redo pq no hay correccion
	#main_CSPR2_20181111()   redo pq no hay correccion
	RMA1_20190308()
	RMA5_20200815()
	RMA3_20190305()
	RMA4_20180209()
	RMA4_20181001()
	RMA4_20190209()
	RMA4_20181218()
	RMA4_20181215()
	RMA4_20181031() ## OJO QUE ACA TENGO QUE CAMBIAR CONTORNO_FIX ... sin correccion por paralaje?	
	
	
	# run [lon_gmi, lat_gmi, tb_s1_gmi]=  GET_TBVH_4icois(opts, icois_input,  gmi_dir+opts['gfile']) for each! to 
	
	return


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------      
def main_20180208(): 
    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12  

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
	    'azimuth_ray': 210,
	    'rfile': 'RMA1/'+rfile, 
	    'era5_file': era5_file,
	    'radar_name':'RMA1',
	    'icoi_PHAIL':[4], 
	    'alternate_azi':[356,220,192],
	    'gfile': gfile, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20180208_RMA1/', 
	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':MINPCTs_labels,'MINPCTs':MINPCTs, 'phail': phail}
    icois_input  = [2,4,5] 
    labels_PHAIL = ['coi=3','icoi=2 (Phail=53.4%)','icoi=1 (Phail=9.1%)'] 
    xlims_xlims_input  = [60, 100, 150] 
    xlims_mins_input  = [10, 40, 60]		
	
    run_general_paper(opts, lat_pfs, lon_pfs, [2,4,5], [356,220,192], labels_PHAIL,  xlims_mins_input, xlims_xlims_input)
    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [3,4]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, 
     GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)
    
    colores_in = ['darkred','darkblue']
    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]))
		print('MAX(TBVH 37:', np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]))
		print('MAX(TBVH 89:', np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]))
		
    tbvbin = 10;	
    coi_250_LABELS = ['coi=1','coi=2 + coi=3']
    TBV_bin  = np.arange(50,300,tbvbin)
    fig = plt.figure(figsize=(5,5))
    for i in range(len(coi_250)):
		#plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i],tbvbin)
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=coi_250_LABELS[i])
    plt.xlabel('89-GHz TBV (K)')
    plt.ylabel('89-GHz Polarization Difference (K)')
    plt.title('Case 08/02/2018')
    plt.xlim([50,300])
    plt.ylim([0,15])
    plt.legend()
    plt.grid(True)
	
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i],tbvbin)
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])	
		      
	      
		      
    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20180208.nc', 'w')
	
    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20180208.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)

	
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def main_DOW7_20181214():
	
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
    reportes_granizo_twitterAPI_meta = [['0320UTC'],['0100']]
    opts = {'xlim_min': -65.3, 'xlim_max': -63.3, 'ylim_min': -32.4, 'ylim_max': -31, 'ZDRoffset': 0, 'caso':'20181214',
	    'rfile': 'DOW7/'+rfile, 'gfile': gfile, 'azimuth_ray': 0,
	     'radar_name':'DOW7', 'era5_file': era5_file,'alternate_azi':[30], 'ZDRoffset': 0,
	     'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181214_RMA1/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	     'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [15], 'files_list':files_list}
    icois_input  = [15] 
    labels_PHAIL = ['Phail=X%'] 
    xlims_xlims_input  = [80] 
    xlims_mins_input  = [20]		

    run_general_paper_Figure_onlyHID(opts, lat_pfs, lon_pfs, [15], [205], labels_PHAIL,  xlims_mins_input, xlims_xlims_input)






    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],3)

    coi_250 =  [0]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],3)
    tbvbin = 10;	
    coi_250_LABELS = ['coi=1','coi=2 + coi=3']
    TBV_bin  = np.arange(50,300,tbvbin)
    fig = plt.figure(figsize=(5,5))
    for i in range(len(coi_250)):
		#plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i],tbvbin)
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=coi_250_LABELS[i])
    plt.xlabel('89-GHz TBV (K)')
    plt.ylabel('89-GHz Polarization Difference (K)')
    plt.title('Case 14/12/2018')
    plt.xlim([50,300])
    plt.ylim([0,15])
    plt.legend()
    plt.grid(True)

	
	
	
	
	
	
	
    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]))
		print('MAX(TBVH 37:', np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]))
		print('MAX(TBVH 89:', np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]))	
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])










    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])	






		
    [ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    contourstats = xr.Dataset( {
                    "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
                    "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    	            "gridz": (('altitude'), gridz) })
    contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181214.nc', 'w')

    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181214.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]

	
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
    opts = {'xlim_min': -65, 'xlim_max': -63.6, 'ylim_min': -32.6, 'ylim_max': -31.5, 
	    'azimuth_ray': 0,
	    'rfile': 'CSPR2_data/'+rfile, 
   	    'era5_file': era5_file,'caso':'20181111',
	    'gfile': gfile,
    	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181111am/', 
	    'alternate_azi':[30],
	    'ZDRoffset': 0,
    	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
     	   'icoi_PHAIL': [3], 'radar_name':'CSPR2'}
    icois_input  = [3] 
    labels_PHAIL = ['Phail=65.3%'] 
    xlims_xlims_input  = [60] 
    xlims_mins_input  = [0]		


    run_general_paper_Figure_onlyHID(opts, lat_pfs, lon_pfs, [3], [205], labels_PHAIL,  xlims_mins_input, xlims_xlims_input)




    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [10]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)

    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])	





    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181111.nc', 'w')

    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181111.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)	
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]



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
    	    'window_calc_KDP': 7, 'era5_file': era5_file, 'azimuth_ray': 50,
    	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190308/', 
    	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
    	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
     	   'icoi_PHAIL': [3], 'radar_name':'RMA1'}
    icois_input  = [3]



    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [6]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)
    tbvbin = 10;	
    coi_250_LABELS = ['coi=1','coi=2 + coi=3']
    TBV_bin  = np.arange(50,300,tbvbin)
    fig = plt.figure(figsize=(5,5))
    for i in range(len(coi_250)):
		#plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i],tbvbin)
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=coi_250_LABELS[i])
    plt.xlabel('89-GHz TBV (K)')
    plt.ylabel('89-GHz Polarization Difference (K)')
    plt.title('Case 8/3/2019')
    plt.xlim([50,300])
    plt.ylim([0,15])
    plt.legend()
    plt.grid(True)





    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])	



    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20190308.nc', 'w')

    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20190308.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)		
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]


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
    reportes_granizo_twitterAPI_geo = [[-25.93, -54.57], [-27.03, -55.24], [-25.62, -53.77], [-26.07,-53.05],
				       [-25.67,-53.81],[-25.66,-53.55],[-26.24,-52.71],[-25.28,-53.95]] 
    reportes_granizo_twitterAPI_meta = ['Wanda', 'Jardin de America', '','','','','','']
    opts = {'xlim_min': -55.0, 'xlim_max': -52.0, 'ylim_min': -27.5, 'ylim_max': -25.0, 
	    'ZDRoffset': 2, 'rfile': 'RMA5/'+rfile, 'gfile': gfile, 'azimuth_ray': 50,
	    'window_calc_KDP': 7, 'era5_file': era5_file,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20200815_RMA5/',
    	    'alternate_azi':[331,335,10],'caso':'20200815',
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [7], 'radar_name':'RMA5'}
    icois_input  = [7] 
    labels_PHAIL = ['coi=1 (Phail=72.5%)','coi=1 (Phail=72.5%)','coi=2 (Phail=46.6%)' ] 
    xlims_xlims_input  = [190, 190, 150] 
    xlims_mins_input  = [0, 0, 0]	 
	
	
	
    run_general_paper(opts, lat_pfs, lon_pfs, [7], [331,335,10], labels_PHAIL,  xlims_mins_input, xlims_xlims_input)

	
    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [4]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)

    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
	
	
	
    tbvbin = 10;	
    coi_250_LABELS = ['coi=1']
    TBV_bin  = np.arange(50,300,tbvbin)
    fig = plt.figure(figsize=(5,5))
    for i in range(len(coi_250)):
		#plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i],tbvbin)
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=coi_250_LABELS[i])
    plt.xlabel('89-GHz TBV (K)')
    plt.ylabel('89-GHz Polarization Difference (K)')
    plt.title('Case 15/08/2020')
    plt.xlim([50,300])
    plt.ylim([0,15])
    plt.legend()
    plt.grid(True)
	
	
	
	
	
	
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])	



    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20200815.nc', 'w')

    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20200815.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)	
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]
	

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
	    'window_calc_KDP': 7, 'era5_file': era5_file, 'azimuth_ray': 210,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190305_RMA3/', 
	    'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL':[6], 'radar_name':'RMA3'}
    icois_input  = [6,7] 

	
    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [0,8]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)

    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])		
	
    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20190305.nc', 'w')

    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20190305.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)	
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]


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
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 'azimuth_ray': 210,
	    'window_calc_KDP': 7, 'era5_file': era5_file,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20180209_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [17], 'radar_name':'RMA4'}
    icois_input  = [10,17,19] 
    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

		
    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [0,10]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)

    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20180209.nc', 'w')

    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20180209.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)	
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]


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
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 'azimuth_ray': 150,
	    'window_calc_KDP': 7,'era5_file': era5_file,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181001_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [37], 'radar_name':'RMA4'}
    icois_input  = [26,36,37] 

    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [0,7]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)

    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])


    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181001.nc', 'w')

    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181001.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)	
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]



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
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 'azimuth_ray': 268,
	    'window_calc_KDP': 7,  'era5_file': era5_file,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190209_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [15], 'radar_name':'RMA4'}
    icois_input  = [11,15,16] 

    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [2,4]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)

    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])


    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20190209.nc', 'w')


    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20190209.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)	
	
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]




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
	    'window_calc_KDP': 7,  'era5_file': era5_file, 'azimuth_ray': 210,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181218_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [7, 6], 'radar_name':'RMA4'}
    icois_input  = [8, 7, 6]

    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],7)

    coi_250 =  [39]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],7)

    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])

    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181218.nc', 'w')


    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181218.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]





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
	    'window_calc_KDP': 7, 'era5_file': era5_file, 'azimuth_ray': 180,
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181215_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [14,11], 'radar_name':'RMA4'}
    icois_input  = [14, 11] 

    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [1]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)

    for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])

    #[ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181215.nc', 'w')


    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181215.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)


    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]



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
    opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -25.5, 
	    'ZDRoffset': 1.5,   
	    'rfile': 'RMA4/'+rfile, 'gfile': gfile, 'azimuth_ray': 157,
	    'window_calc_KDP': 7,  'era5_file': era5_file,'caso':'20181031',
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181031_RMA4/', 
	     'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir, 
	    'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail, 
	   'icoi_PHAIL': [1,58,20], 'radar_name':'RMA4', 'alternate_azi':[230, 199, 157, 65]}
    icois_input  = [1,26,20,58] 
    
    labels_PHAIL = ['coi=1 (Phail=73.8%)','coi=2 (Phail=26.2%)','coi=3 (Phail=93.1%)','coi=4 (Phail=99.3%)' ] 
    xlims_xlims_input  = [250,180,200,240] #  [240, 200, 180, 250] 
    xlims_mins_input  = [150,100,150,150]  #  [150,150,100,150]  	
    run_general_paper(opts, lat_pfs, lon_pfs, icois_input,[230, 199, 157, 65], labels_PHAIL,  xlims_mins_input, xlims_xlims_input)

    GET_TBVH_250ICOIS(opts, gmi_dir+opts['gfile'],1)

    coi_250 =  [0,4,11,17]	
    [GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+opts['gfile'],1)

    colores_in = ['k','darkgreen','darkred','darkblue','cyan']
    tbvbin = 10;	
    coi_250_LABELS = ['coi=1','coi=2','coi=3', 'coi=4']
    TBV_bin  = np.arange(50,300,tbvbin)
    fig = plt.figure(figsize=(5,5))
    for i in range(len(coi_250)):
		#plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i],tbvbin)
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=coi_250_LABELS[i])
    plt.xlabel('89-GHz TBV (K)')
    plt.ylabel('89-GHz Polarization Difference (K)')
    plt.title('Case 31/10/2018')
    plt.xlim([50,300])
    plt.ylim([0,15])
    plt.legend()
    plt.grid(True)





	for i in range(len(GMI_latlat)):
		print('-------- icoi NR. '+str(coi_250[i])+str(' -----'))
		print('MAX(TBVH 19:', round(np.max(GMI_tbs1_19[i]-GMI_tbs1_19H[i]),1))
		print('MAX(TBVH 37:', round(np.max(GMI_tbs1_37[i]-GMI_tbs1_37H[i]),1))
		print('MAX(TBVH 89:', round(np.max(GMI_tbs1_85[i]-GMI_tbs1_85H[i]),1))	
		
    TBV_bin  = np.arange(50,300,5)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_85[i], GMI_tbs1_85[i]-GMI_tbs1_85H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_85[i]-GMI_tbs1_85H[i], GMI_tbs1_85[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('85 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])
    fig = plt.figure(figsize=(10,10))
    for i in range(len(coi_250)):
		plt.plot(GMI_tbs1_37[i], GMI_tbs1_37[i]-GMI_tbs1_37H[i],'x',color=colores_in[i])
    		running_median = get_median(GMI_tbs1_37[i]-GMI_tbs1_37H[i], GMI_tbs1_37[i])
    		plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, color=colores_in[i], linestyle='-', label=str(coi_250[i]))
    plt.xlabel('TBV')
    plt.ylabel('TBVH')
    plt.title('37 GHz')
    plt.xlim([50,300])
    plt.ylim([0,15])



    [ check_resolxy, check_resolz, HIDs_coi_GRID, HIDs_coi, gridz] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)

    #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, 
    #	PIXELS_NOPHAIL, GATES_PHAIL, GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, 
    #	ZDRarray_PHAIL, ZDRarray_NOPHAIL] = run_general_case(opts, lat_pfs, lon_pfs, icois_input)
    
    #contourstats = xr.Dataset( {
    #                "HIDs_coi_GRID": (('grid_nz','icois','HIDs'), HIDs_coi_GRID),
    #                "HIDs_coi": (('nsweeps','icois','HIDs'), HIDs_coi),
    #	            "gridz": (('altitude'), gridz) })
    #contourstats.to_netcdf('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181031.nc', 'w')

	
    DATA = xr.open_dataset('/home/victoria.galligani/Work/Studies/Hail_MW/case_outputfiles_stats/contourstats_20181031.nc')
    CalcandPlot_HID_fraction(DATA['HIDs_coi_GRID'], icois_input, DATA['gridz'], opts)
	
	
    return #[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
    #	    GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL]



#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------- 















#----------------------------------------------------------------------------------------------
# main_20180208 OK 
#----------------------------------------------------------------------------------------------  	
import xarray as xr

[PCTarray_PHAIL_out, PCTarray_NOPHAIL_out, AREA_PHAIL, AREA_NOPHAIL,  PIXELS_PHAIL, PIXELS_NOPHAIL, GATES_PHAIL, 
	 GATES_NOPHAIL, ZHarray_PHAIL, ZHarray_NOPHAIL, ZDRarray_PHAIL, ZDRarray_NOPHAIL] = main_20180208()

PCTarray_PHAIL_out_mean = []
PCTarray_NOPHAIL_out_mean = []
for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
	PCTarray_PHAIL_out_mean.append(np.nanmean(PCTarray_PHAIL_out[:,ifreq]))
	PCTarray_NOPHAIL_out_mean.append(np.nanmean(PCTarray_NOPHAIL_out[:,ifreq]))

if len(PCTarray_PHAIL_out) == 1:
	PCTarray_PHAIL_out_ = []
	for ifreq in range(PCTarray_NOPHAIL_out.shape[1]):
		PCTarray_PHAIL_out_.append( PCTarray_PHAIL_out[0][ifreq])
		
# And create a netcdf file
RMA1_20180208 = xr.Dataset( {
                    "PCTarray_PHAIL_out": (('PCTs'), PCTarray_PHAIL_out_),
                    "PCTarray_NOPHAIL_out": (('noicois','PCTs'), PCTarray_NOPHAIL_out),
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
                    "meanAREA_PHAIL":            (('Nr'),   [np.nanmean(AREA_PHAIL)]),
                    "meanAREA_NOPHAIL":          (('Nr'),   [np.nanmean(AREA_NOPHAIL)]),
                    "meanPIXELS_PHAIL":          (('Nr'),   [np.nanmean(PIXELS_PHAIL)]), 
                    "meanPIXELS_NOPHAIL":        (('Nr'),   [np.nanmean(PIXELS_NOPHAIL)]), 
                    "meanGATES_PHAIL":           (('Nr'),   [np.nanmean(GATES_PHAIL)]), 
                    "meanGATES_NOPHAIL":         (('Nr'),   [np.nanmean(GATES_NOPHAIL)]),
	            "AREA_PHAIL":            (('icois'),    AREA_PHAIL),
                    "AREA_NOPHAIL":          (('noicois'),    AREA_NOPHAIL),
                    "PIXELS_PHAIL":          (('icois'),    PIXELS_PHAIL), 
                    "PIXELS_NOPHAIL":        (('noicois'),    PIXELS_NOPHAIL), 
                    "GATES_PHAIL":           (('icois'),    GATES_PHAIL), 
                    "GATES_NOPHAIL":         (('noicois'),    GATES_NOPHAIL),
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
                    "PCTarray_NOPHAIL_out": (('noicois','PCTs'), PCTarray_NOPHAIL_out),
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
                    "PCTarray_NOPHAIL_out": (('noicois','PCTs'), PCTarray_NOPHAIL_out),
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
                    "PCTarray_NOPHAIL_out": (('noicois','PCTs'), PCTarray_NOPHAIL_out),
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
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,2], c=RMA4_20180209[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,2], c=RMA4_20180209[novarTitle].data[1],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,2], c=RMA4_20181001[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,2], c=RMA4_20181001[novarTitle].data[1],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,1],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,2], c=RMA4_20190209[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,1],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,2], c=RMA4_20190209[novarTitle].data[1],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181031['PCTarray_NOPHAIL_out'].data[1],  RMA4_20181031['PCTarray_NOPHAIL_out'].data[2], c=RMA4_20181031[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181218['PCTarray_NOPHAIL_out'].data[1],  RMA4_20181218['PCTarray_NOPHAIL_out'].data[2], c=RMA4_20181218[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())

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
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[0,3], c=RMA4_20180209[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20180209['PCTarray_NOPHAIL_out'].data[1,3], c=RMA4_20180209[novarTitle].data[1],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[0,3], c=RMA4_20181001[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20181001['PCTarray_NOPHAIL_out'].data[1,3], c=RMA4_20181001[novarTitle].data[1],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,2],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[0,3], c=RMA4_20190209[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,2],  RMA4_20190209['PCTarray_NOPHAIL_out'].data[1,3], c=RMA4_20190209[novarTitle].data[1],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181031['PCTarray_NOPHAIL_out'].data[2],  RMA4_20181031['PCTarray_NOPHAIL_out'].data[3], c=RMA4_20181031[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA4_20181218['PCTarray_NOPHAIL_out'].data[2],  RMA4_20181218['PCTarray_NOPHAIL_out'].data[3], c=RMA4_20181218[novarTitle].data[0],
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())


	plt.scatter(np.nan, np.nan, marker='o',color='w', edgecolor='k', label='Phail')
	plt.scatter(np.nan, np.nan, marker='s',color='w', edgecolor='k', label='noPhail')
	plt.legend()
	
	plt.grid(True)
	plt.xlabel('MINPCT(37)')
	plt.ylabel('MINPCTT(85)')
	plt.xlim([80,230])
	plt.ylim([50,170])
	plt.suptitle('RMA3+RMA4 '+ var4title,y=0.9)

	return



#----------------------------------------------------------------------------------------------
# scatter plots 37/19 AND 37/85 w/ further info
#----------------------------------------------------------------------------------------------

def make_scatterplots_sector2_with3Dvalue(var4title, varTitle, novarTitle, vminn, vmaxx ): 
	
	fig = plt.figure(figsize=(10,10)) 
	gs1 = gridspec.GridSpec(2, 2)
	ax1 = plt.subplot(gs1[0,0])

	pcm = plt.scatter(RMA5_20200815['PCTarray_PHAIL_out'].data[1],  RMA5_20200815['PCTarray_PHAIL_out'].data[2], 
			  c=RMA5_20200815[varTitle].data[0], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())	
	plt.colorbar(pcm)

	plt.legend(fontsize=10)
	plt.grid(True)
	plt.xlabel('MINPCT(19)')
	plt.ylabel('MINPCTT(37)')
	plt.xlim([170,300])
	plt.ylim([80,240])

	ax1 = plt.subplot(gs1[0,1])
	
	pcm = plt.scatter(RMA5_20200815['PCTarray_PHAIL_out'].data[2],  RMA5_20200815['PCTarray_PHAIL_out'].data[3], 
			  c=RMA5_20200815[varTitle].data[0], marker='o', s=30, vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.colorbar(pcm)


	plt.scatter(np.nan, np.nan, marker='o',color='w', edgecolor='k', label='Phail')
	plt.legend()
	
	plt.grid(True)
	plt.xlabel('MINPCT(37)')
	plt.ylabel('MINPCTT(85)')
	plt.xlim([80,230])
	plt.ylim([50,170])
	plt.suptitle('RMA5 '+ var4title,y=0.9)

	return


#---------------------------------------------------------------------------------------------
def make_scatterplots_sector1_with3Dvalue(var4title, varTitle, novarTitle, vminn, vmaxx ): 
	
	fig = plt.figure(figsize=(10,10)) 
	gs1 = gridspec.GridSpec(2, 2)
	ax1 = plt.subplot(gs1[0,0])
	plt.scatter(RMA1_20180208['PCTarray_PHAIL_out'].data[1],  RMA1_20180208['PCTarray_PHAIL_out'].data[2], 
		    c=RMA1_20180208[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(DOW7_20181214['PCTarray_PHAIL_out'].data[1],  DOW7_20181214['PCTarray_PHAIL_out'].data[2], 
		    c=DOW7_20181214[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(CSPR2_20181111['PCTarray_PHAIL_out'].data[1],  CSPR2_20181111['PCTarray_PHAIL_out'].data[2], 
		    c=CSPR2_20181111[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())	
	pcm = plt.scatter(RMA1_20190308['PCTarray_PHAIL_out'].data[1],  RMA1_20190308['PCTarray_PHAIL_out'].data[2], 
		    c=RMA1_20190308[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())		
	
	plt.scatter(RMA1_20180208['PCTarray_NOPHAIL_out'].data[0,1],  RMA1_20180208['PCTarray_NOPHAIL_out'].data[0,2],
		    c=RMA1_20180208[novarTitle].data[0], 
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA1_20180208['PCTarray_NOPHAIL_out'].data[1,1],  RMA1_20180208['PCTarray_NOPHAIL_out'].data[1,2],
		    c=RMA1_20180208[novarTitle].data[1], 
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())	

	plt.colorbar(pcm)

	plt.grid(True)
	plt.xlabel('MINPCT(19)')
	plt.ylabel('MINPCTT(37)')
	plt.xlim([170,300])
	plt.ylim([80,240])

	ax1 = plt.subplot(gs1[0,1])
	plt.scatter(RMA1_20180208['PCTarray_PHAIL_out'].data[2],  RMA1_20180208['PCTarray_PHAIL_out'].data[3], 
		    c=RMA1_20180208[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(DOW7_20181214['PCTarray_PHAIL_out'].data[2],  DOW7_20181214['PCTarray_PHAIL_out'].data[3], 
		    c=DOW7_20181214[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(CSPR2_20181111['PCTarray_PHAIL_out'].data[2],  CSPR2_20181111['PCTarray_PHAIL_out'].data[3], 
		    c=CSPR2_20181111[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())	
	pcm = plt.scatter(RMA1_20190308['PCTarray_PHAIL_out'].data[2],  RMA1_20190308['PCTarray_PHAIL_out'].data[3], 
		    c=RMA1_20190308[varTitle].data[0], s=30, marker='o', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())		
	
	plt.scatter(RMA1_20180208['PCTarray_NOPHAIL_out'].data[0,2],  RMA1_20180208['PCTarray_NOPHAIL_out'].data[0,3],
		    c=RMA1_20180208[novarTitle].data[0], 
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())
	plt.scatter(RMA1_20180208['PCTarray_NOPHAIL_out'].data[1,2],  RMA1_20180208['PCTarray_NOPHAIL_out'].data[1,3],
		    c=RMA1_20180208[novarTitle].data[1], 
		    s=40, marker='s',edgecolor='k', vmin=vminn, vmax=vmaxx, norm=matplotlib.colors.LogNorm())	


	plt.scatter(np.nan, np.nan, marker='o',color='w', edgecolor='k', label='Phail')
	plt.scatter(np.nan, np.nan, marker='s',color='w', edgecolor='k', label='noPhail')
	plt.legend()
	
	plt.grid(True)
	plt.xlabel('MINPCT(37)')
	plt.ylabel('MINPCTT(85)')
	plt.xlim([80,230])
	plt.ylim([50,170])
	plt.suptitle('COR.(RMA1+CSPR2+DOW7) '+ var4title,y=0.9)

	return










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


#===================================================================================================================
make_scatterplots_sector3_with3Dvalue('AREA_PHAIL', 'AREA_PHAIL','AREA_NOPHAIL', 10, 1e4 )
make_scatterplots_sector3_with3Dvalue('PIXELS_PHAIL', 'PIXELS_PHAIL','PIXELS_NOPHAIL', 10, 1e4 )
make_scatterplots_sector3_with3Dvalue('GATES_PHAIL', 'GATES_PHAIL','GATES_NOPHAIL', 10, 1e4 )

make_scatterplots_sector2_with3Dvalue('AREA_PHAIL', 'AREA_PHAIL','AREA_NOPHAIL', 10, 1e4 )
make_scatterplots_sector2_with3Dvalue('PIXELS_PHAIL', 'PIXELS_PHAIL','PIXELS_NOPHAIL', 10, 1e4 )
make_scatterplots_sector2_with3Dvalue('GATES_PHAIL', 'GATES_PHAIL','GATES_NOPHAIL', 10, 1e4 )

make_scatterplots_sector1_with3Dvalue('AREA_PHAIL', 'AREA_PHAIL','AREA_NOPHAIL', 10, 1e4 )
make_scatterplots_sector1_with3Dvalue('PIXELS_PHAIL', 'PIXELS_PHAIL','PIXELS_NOPHAIL', 10, 1e4 )
make_scatterplots_sector1_with3Dvalue('GATES_PHAIL', 'GATES_PHAIL','GATES_NOPHAIL', 10, 1e4 )










dsds




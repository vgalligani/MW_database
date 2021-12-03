def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print ("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim)
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print ("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def create_logarithmic_scaled_vectors(vec_min,vec_max, dim):
    
    import numpy as np

    vec = np.logspace(np.log10(vec_min), np.log10(vec_max), num=dim, endpoint=True, base=10.0);

    return [vec]

def interpLevel(grid,value,data,interp):
    """
    Interpolate 3d data to a common z coordinate.

    Can be used to calculate the wind/pv/whatsoever values for a common
    potential temperature / pressure level.

    grid : numpy.ndarray
       The grid. For example the potential temperature values for the whole 3d
       grid.

    value : float
       The common value in the grid, to which the data shall be interpolated.
       For example, 350.0

    data : numpy.ndarray
       The data which shall be interpolated. For example, the PV values for
       the whole 3d grid.

    kind : str
       This indicates which kind of interpolation will be done. It is directly
       passed on to scipy.interpolate.interp1d().

    returns : numpy.ndarray
       A 2d array containing the *data* values at *value*.

    """
    import numpy as np
    from scipy import interpolate
    import numpy.ma as ma

    ret = np.zeros_like(data[0, :, :])
    for yIdx in range(grid.shape[1]):
        for xIdx in range(grid.shape[2]):
            # check if we need to flip the column
            if grid[0,yIdx,xIdx] > grid[-1,yIdx,xIdx]:
                ind = -1
            else:
                ind = 1
            f = interpolate.interp1d(grid[::ind,yIdx,xIdx], \
                data[::ind,yIdx,xIdx], \
                kind=interp)
            ret[yIdx,xIdx] = f(value)

    #ret1 = ma.masked_where(ret <=10, ret)
    mask_ret = ma.masked_invalid(ret)

    return mask_ret

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


def return_constants():

    const = dict([('q_rain_min', 1e-8),      # input 1
                  ('q_rain_max', 0.013),               # input 2
                  ('qn_rain_min', 1E-8),               # input 3
                  ('qn_rain_max', 2.6e6),              # input 4
                  ('q_snow_min',  1e-8),      # input 5
                  ('q_snow_max',  0.02),     # input 6
                  ('qn_snow_min', 1E-8),      # input 3
                  ('qn_snow_max', 2.6e6),     # input 4   
                  ('q_grau_min',  1e-8),      # input 5
                  ('q_grau_max',  0.02),     # input 6
                  ('qn_grau_min', 1E-8),      # input 
                  ('qn_grau_max', 2.6e6),     # input 4   
                  ('q_ice_min',  1E-8),      # input 5
                  ('q_ice_max',  0.1),     # input 6 0.02
                  ('qn_ice_min', 1E-8),      # input 
                  ('qn_ice_max', 2.6e6),     # input 4   
                  ('q_clou_min',  1e-10),
                  ('q_clou_max',  0.013),
                  ('dim_ice',  200),
                  ('dim', 200)])          
    return const


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

def rmaxRing(lat_radar,lon_radar,Rmax):
    """
    Calculate lat-lon of the maximum range ring

    lat_radar : float32
       Radar latitude. Positive is north.

    lon_radar : float32
       Radar lngitude. Positive is east.

    rmax : float32
       Radar maximum range in kilometers.

    returns : numpy.array
       A 2d array containing the maximum range latitudes (lat_Rmax) and longitudes (lon_Rmax)

    """
    import numpy as np

    R=12742./2.
    m=2.*np.pi*R/360.
    alfa=np.arange(-np.pi,np.pi,0.0001)

    nazim  = 360.0
    nbins  = 480.0
    binres = 0.5

    # A continuacion calculo las lats y lons alrededor del radar
    azimuth = np.transpose(np.tile(np.arange(0,nazim,1), (nbins,1)))
    rangos  = np.tile(np.arange(0,nbins,1)*binres, (nazim,1))
    lats    = lat_radar + (rangos/m)*np.cos((azimuth)*np.pi/180.0)
    lons    = lon_radar + (rangos/m)*np.sin((azimuth)*np.pi/180.0)/np.cos(lats*np.pi/180)

    lat_Rmax = lat_radar + (Rmax/m)*np.sin(alfa)
    lon_Rmax = lon_radar + ((Rmax/m)*np.cos(alfa)/np.cos(lat_Rmax*np.pi/180))

    return lat_Rmax, lon_Rmax
    
    
def esfericas_a_latlon(ele,az,rng):
    import numpy as np
    import numpy.matlib 
    
    theta_e = ele* 2* np.pi / 360.0       # elevation angle in radians.
    theta_a = az* 2* np.pi / 360.0        # azimuth angle in radians.
    #Re = 6371.0 * 1000.0 * 4.0 / 3.0     # effective radius of earth in meters.
    #r = rng * 1000.0                    # distances to gates in meters.

    dx = rng/1000 * np.cos(theta_e)* np.sin(theta_a)
    dy = rng/1000 * np.cos(theta_e)* np.cos(theta_a)

    dlat= (dy * 360)/(2*np.pi*6371)
    lat_efec= -31.858334 + dlat
    dlon= (dx * 360)/(2*np.pi*6371*np.cos((2*np.pi/360)*lat_efec))
    lon_efec= -60.539722 + dlon
    
    return lat_efec, lon_efec


#===========================================================================================================
# OTRAS FUNCIONES CONTENIDAS EN ESTE MODULO
#===========================================================================================================   

#From Pyart order to ARE (Azmimuth , range , elevation )
# El codifo order_variable de franco lo cambie un poco con respecto al ray_angle_res
# y con var_names, ver comentarios "VG" 

def order_variable ( radar , var_name , undef )  : 
    import numpy as np
    import numpy.matlib 

    #From azimuth , range -> azimuth , range , elevation 

    #if radar.ray_angle_res != None   :                               # VG
    #  #print( radar.ray_angle_res , radar.ray_angle_res == None )    # VG
    #  ray_angle_res = np.unique( radar.ray_angle_res['data'] )       # VG
    #else                             :                               # VG
    print('Warning: ray_angle_res no esta definido, estimo la resolucion en radio como la diferencia entre los primeros angulos')
    ray_angle_res = np.min( np.abs( radar.azimuth['data'][1:] - radar.azimuth['data'][0:-1] ) )
    print('La resolucion en rango estimada es: ',ray_angle_res)


    if( np.size( ray_angle_res ) >= 2 )  :
      print('Warning: La resolucion en azimuth no es uniforme en los diferentes angulos de elevacion ')
      print('Warning: El codigo no esta preparado para considerar este caso y puede producir efectos indeseados ')
    ray_angle_res_copy = np.copy(ray_angle_res)
    ray_angle_res_copy=np.nanmean( ray_angle_res_copy )

    levels=np.sort( np.unique(radar.elevation['data']) )
    nb=radar.azimuth['data'].shape[0]

    order_azimuth=np.arange(0.0,360.0,ray_angle_res_copy) #Asuming a regular azimuth grid

    na=np.size(order_azimuth)
    ne=np.size(levels)
    nr=np.size(radar.range['data'].data) 


    var = np.ones( (nb,nr) )

    if ( var_name == 'altitude' ) :
        var[:]=radar.gate_altitude['data']  
    elif( var_name == 'longitude' ) :
        var[:]=radar.gate_longitude['data']
    elif( var_name == 'latitude'  ) :
        var[:]=radar.gate_latitude['data']
    elif( var_name == 'x' )         :
        var[:]=radar.gate_x['data']
    elif( var_name == 'y' )         : 
        var[:]=radar.gate_y['data']
    else  :
        var[:]=radar.fields[var_name]['data']    # VG (para Parana parece que era ['data'].data)

    #Allocate arrays
    order_var    =np.zeros((na,nr,ne))
    order_time   =np.zeros((na,ne)) 
    azimuth_exact=np.zeros((na,ne))
    order_n      =np.zeros((na,nr,ne),dtype='int')
   
    current_lev = radar.elevation['data'][0]
    ilev = np.where( levels == current_lev )[0]

    for iray in range( 0 , nb )  :   #Loop over all the rays
 
      #Check if we are in the same elevation.
      if  radar.elevation['data'][iray] != current_lev  :
          current_lev = radar.elevation['data'][iray]
          ilev=np.where( levels == current_lev  )[0]
        
      # Compute the corresponding azimuth index.
      az_index = np.round( radar.azimuth['data'][iray] / ray_angle_res_copy ).astype(int)
      # Consider the case when azimuth is larger than na*ray_angle_res-(ray_angle_res/2)
      if az_index >= na   :  
        az_index = 0

      tmp_var = var[iray,:]
      undef_mask = tmp_var == undef 
      tmp_var[ undef_mask ] = 0.0
    
      order_var [ az_index , : , ilev ] = order_var [ az_index , : , ilev ] + tmp_var
      order_n   [ az_index , : , ilev ] = order_n   [ az_index , : , ilev ] + np.logical_not(undef_mask).astype(int)

      order_time[ az_index , ilev ] = order_time[ az_index , ilev ] + radar.time['data'][iray]
      azimuth_exact[ az_index , ilev ] = azimuth_exact[ az_index , ilev ] + radar.azimuth['data'][ iray ]

    order_var[ order_n > 0 ] = order_var[ order_n > 0 ] / order_n[ order_n > 0 ]
    order_var[ order_n == 0] = undef
        

    return order_var , order_azimuth , levels , order_time , azimuth_exact

def order_variable_inv (  radar , var , undef )  :

    import numpy as np
    import numpy.matlib 
   
    #From azimuth , range , elevation -> azimuth , range

    na=var.shape[0]
    nr=var.shape[1]
    ne=var.shape[2]

    nb=radar.azimuth['data'].shape[0]

    levels=np.sort( np.unique(radar.elevation['data']) )

    if radar.ray_angle_res != None   :
      #print( radar.ray_angle_res , radar.ray_angle_res == None )
      ray_angle_res = np.unique( radar.ray_angle_res['data'] )
    else                             :
      print('Warning: ray_angle_res no esta definido, estimo la resolucion en radio como la diferencia entre los primeros angulos')
      ray_angle_res = np.min( np.abs( radar.azimuth['data'][1:] - radar.azimuth['data'][0:-1] ) )
      print('La resolucion en rango estimada es: ',ray_angle_res)

    if( np.size( ray_angle_res ) >= 2 )  :
      print('Warning: La resolucion en azimuth no es uniforme en los diferentes angulos de elevacion ')
      print('Warning: El codigo no esta preparado para considerar este caso y puede producir efectos indesaedos ')
    ray_angle_res=np.nanmean( ray_angle_res )

    current_lev = radar.elevation['data'][0]
    ilev = np.where( levels == current_lev  )[0]

    output_var = np.zeros((nb,nr) )
    output_var[:] = undef

    for iray in range( 0 , nb )  :   #Loop over all the rays

      #Check if we are in the same elevation.
      if  radar.elevation['data'][iray] != current_lev  :
          current_lev = radar.elevation['data'][iray]
          ilev=np.where( levels == current_lev  )[0]

      #Compute the corresponding azimuth index.
      az_index = np.round( radar.azimuth['data'][iray] / ray_angle_res ).astype(int)
      #Consider the case when azimuth is larger than na*ray_angle_res-(ray_angle_res/2)
      if az_index >= na   :
          az_index = 0

      output_var[ iray , : ] = var[ az_index , : , ilev ]

    return output_var


def local_mean( array , kernel_x , kernel_y , undef ) :
    #Asumimos que hay condiciones ciclicas en el axis 0 pero no en el 1.
    #array es el array de datos de entrada
    #kernel_x es cual es el desplazamiento maximo (hacia cada lado) en la direccion de x
    #kernel_y es cual es el desplazamiento maximo (hacia cada lado) en la direccion de y
    #undef son los valores invalidos en el array de entrada.
    import numpy as np
    import numpy.matlib 
    [nx,ny]=np.shape(array)
    arraym = np.zeros( np.shape(array) )
    countm = np.zeros( np.shape(array) )
    for ix in range(-kernel_x,kernel_x+1) :
        for iy in range(-kernel_y,kernel_y +1) :
          tmp_array = np.zeros( np.shape(array) )
          if iy > 0 :
              tmp_array[:,0+iy:] = array[:,0:-iy]
          if iy == 0 :
              tmp_array = np.copy(array)
          if iy < 0 :
              tmp_array[:,0:iy] = array[:,-iy:]
          tmp_array=np.roll( tmp_array , ix , axis=0 )
          mask = tmp_array != undef
          arraym[ mask ] = arraym[mask] + tmp_array[mask]
          countm[ mask ] = countm[mask] + 1
    mask = countm > 0
    arraym[mask] = arraym[mask] / countm[mask]
    arraym[~mask] = undef 

    return arraym


def calcula_vil( dbz_in , x_in , y_in , z_in , undef )  :
  import numpy as np
  import numpy.matlib 
  from scipy.interpolate import interp1d
  
  dbz = np.copy(dbz_in)
  x   = np.copy(x_in)
  y   = np.copy(y_in)
  z   = np.copy(z_in)

  [na,nr,ne] = dbz.shape

  dbz_int = np.zeros( dbz.shape )
  z_int   = np.zeros( dbz.shape )
  vil     = np.zeros( dbz.shape )
  vil_int     = np.zeros( (na , nr) )

  ranger = ( x**2 + y**2 )**0.5
  ranger0 = ranger[:,:,0]

  #Calculo el VIL en la reticula x0 , y0
  for ie in range(ne)   :
    dbz2d = np.copy( dbz[:,:,ie] )
    dbz2d_mean = local_mean( dbz2d , 1 , 1 , undef )
    #Intento salvar algunos agujeros que pueda haber en el campo de reflectividad.
    mask = np.logical_or( dbz2d == undef , dbz2d < 0.0 )
    dbz2d[ mask ] = dbz2d_mean[mask]
    #Los undef que quedaron pasan a ser 0 para el calcuo del VIL 
    dbz2d[dbz2d == undef ] = 0.0  
    dbz2d = 10 ** (  dbz2d / 10.0 )
      
    for ia in range(na)   :
      
      interpolator = interp1d(ranger[ia,:,ie] , dbz2d[ia,:] , kind='linear' , bounds_error = False , fill_value = 0.0 )
      dbz_int[ia,:,ie] = interpolator(ranger0[ia,:])
      interpolator = interp1d(ranger[ia,:,ie] , z[ia,:,ie] , kind='linear' , bounds_error = False , fill_value = np.nan)
      z_int[ia,:,ie] = interpolator(ranger0[ia,:])
      #Completo algunos niveles repitiendo el ultimo valor para hacer mas robusto el calculo del VIL
    if ie > 0 :
      dz = z_int[:,:,ie] - z_int[:,:,ie-1] ; dz[dz==0] = np.nan
      vil_inc = 3.44e-6 * ( ( 0.5*(dbz_int[:,:,ie] + dbz_int[:,:,ie-1]) ) ** (4.0/7.0) ) * ( z_int[:,:,ie] - z_int[:,:,ie-1] )
      vil_inc[np.isnan(vil_inc)] = 0.0 
      vil_int = vil_int + vil_inc
  #Hasta aca tenemos vil_int que es el vil en la reticula x0 y0. Para las cuentas en general nos puede venir bien
  #tener el vil interpolado a la reticula x,y (es decir un vil definido para todoas las elevaciones del radar)
  vil_int[ np.isnan(vil_int) ] = 0.0

  for ie in range(ne)  :
    for ia in range(na) :
        interpolator = interp1d(ranger0[ia,:],vil_int[ia,:],kind='linear', bounds_error = False , fill_value = 0.0 )
        vil[ia,:,ie] = interpolator( ranger[ia,:,ie] )
      

  return vil   

def calcula_shear( vr_in , dbz_in , z_in , azimuth , rango , levels , undef , delta_max = 10000.0 , delta_min = 1000.0 , rango_min = 15000.0 , dbz_thr = 20.0 , z_thr = 8000) :
    
    import numpy as np
    import numpy.matlib 
    
    vr=np.copy(vr_in)
    dbz=np.copy(dbz_in)
    z=np.copy(z_in)
    #delta_max El maximo delta en metros en la direccion del azimuth
    #delta_min El minimo delta en metros en la direccion del azimuth
    #rango_min El minimo rango a partir del cual vamos a hacer los calculos. 
    #dbz_thr   Solo se consideran valores con dbz por encima de este valor en el calculo de las cortantes
    #z_thr     Solo se consideran pixeles por debajo de esta altura en el calculo de las cortantes.

    [rango_mat , azimuth_mat , level_mat ] = np.meshgrid(rango, azimuth , levels)
    
    perimetro= 2*np.pi * rango_mat *np.cos((2.0*np.pi/360.0)* level_mat )
    perimetro[ rango_mat < rango_min ] = np.nan   #NO CONSIDERAMOS LOS ANILLOS QUE ESTAN A UNA DISTANCIA DEL RADAR POR DEBAJO DE LA DISTANCIA MINIMA.  
    res_azimuth = perimetro / azimuth.size #Calculo el espaciamiento variable entre dos azimuths consecutivos (en metros)           
    delta_pixel_max=np.around( 0.5 * delta_max / res_azimuth ) 
    delta_pixel_min=np.around( 0.5 * delta_min / res_azimuth )
    #Trato de garantizar de tener al menos un delta de 1 punto de reticula (equivale a una diferencia finita centrada)
    delta_pixel_min[ delta_pixel_min == 0 ] = 1.0
    delta_pixel_max[ delta_pixel_max == 0 ] = 1.0 
   
    #Asumimos que los datos de velocidad radial y reflectividad estan bien y que los que no tienen un valor igual a undef
    #reemplazamos todos los undef por nan.
    vr[vr == undef ] = np.nan
    dbz[ dbz == undef ] = np.nan
    mask = np.logical_or( np.logical_or( dbz < dbz_thr , z > z_thr ) , rango_mat < rango_min  )
    #No vamos a calcular cortantes que involucren valores de Vr cuya reflectividad este por debajo de 20 dbz.
    #Esta mascara no la voy a necesitar mas porque ya restringimos los valores de Vr que vamos a usar.
    vr[mask] = np.nan

    #Definimos la variable shear que es donde almacenaremos la cortante azimutal maxima, minima y la media
    shear_max = np.zeros( np.shape( vr ) )
    shear_min = np.zeros( np.shape( vr ) )
    shear_mean = np.zeros( np.shape( vr ) )
    shear_count = np.zeros( np.shape( vr ) )
    for l in range( int(np.nanmin(delta_pixel_min)) , int(np.nanmax(delta_pixel_max) + 1 ) ):
        #l es siempre mayor o igual que 1 (esto es a lo sumo calculamos el shear considerando los dos puntos vecinos)
        #Calculo diferencias finitas centradas es decir siempre tomamos diferencias alrededor de un punto determinado.
        #A medida que crece l, crece el delta_x total, pero estamos seimpre centrados en el mismo punto. 
        tmp_shear = ( np.roll( vr , -l , axis=0 ) - np.roll( vr , l , axis=0 ) ) / ( 2.0 * l * res_azimuth )
        #Sobre que pixeles me sirve esta diferencia que calculamos?
        mask = np.logical_and( delta_pixel_min <= l , delta_pixel_max >= l )
        #Mascara 2 me dice en cuales de los puntos tiene sentido usar el valor de diferencia a una distancia de l pixeles

        #A continuacion vamos guardando para cada pixel el shear medio, el maximo y el minimo obtenido a partir de los diferentes 
        #deltas. 
        mask_1 = np.logical_and( mask , tmp_shear > shear_max )
        shear_max[mask_1] = tmp_shear[ mask_1 ]
        mask_1 = np.logical_and( mask , tmp_shear < shear_min )
        shear_min[mask_1] = tmp_shear[ mask_1 ]
        mask_1 = np.logical_and( mask , ~np.isnan( tmp_shear ) )
        shear_mean[mask_1] = shear_mean[mask_1] + tmp_shear[mask_1]
        shear_count[mask_1] = shear_count[mask_1] + 1.0 
        
        #Hasta aca shear tienela maxima cortante azimutal calculada con los l validos para cada pixel.
    mask = shear_count > 0
    shear_mean[mask] = shear_mean[mask] / shear_count[mask]
    
    return shear_mean , shear_max , shear_min 

def var_int( var_in , x_in , y_in , int_lev = 0 , fill_value = 0.0 ) :
    from scipy.interpolate import interp1d
    import numpy as np
    import numpy.matlib 
    var = np.copy(var_in)
    x   = np.copy(x_in)
    y   = np.copy(y_in)

    [na,nr,ne] = var.shape

    var_int = np.zeros( var.shape )
    ranger = ( x**2 + y**2 )**0.5
    ranger0 = ranger[:,:,int_lev]

    for ie in range(ne)   :
        for ia in range(na)   :
          interpolator = interp1d(ranger[ia,:,ie] , var[ia,:,ie] , kind='linear' , bounds_error = False , fill_value = 0.0 )
          var_int[ia,:,ie] = interpolator(ranger0[ia,:])

    return var_int

def calcula_shear_index( shear , vil , x , y , z , z_min , z_max  ) :
    
    import numpy as np
    import numpy.matlib 
    #CALCULAMOS UNA CORTANTE PESADA POR VIL Y CON ENFASIS EN NIVELES MEDIOS.
    shear_vil = shear * vil   
    shear_vil_int = var_int( shear_vil , x , y )

    shear_vil_int[ np.logical_or( z<z_min, z>z_max)] = 0.0

    shear_index = np.sum( shear_vil_int , 2 )
    
    return shear_index


 

############esta funcion acomoda las elevaciones, loos azimuts y los rangos en forma de vector, simplemente para poder calcular la lat lon de todo el dominio del radar######
def posiciones_dominio_radar(rango_mat,azimuth_mat,level_mat):
    
    import numpy as np
        
    levels=[0.5,1.3,2.3,3.5,5,6.9,9.1,11.8,15.1,19.2]
    dim_vector=480*360*len(levels)
    vector_rangos=np.zeros((dim_vector,1))
    vector_azimuts=np.zeros((dim_vector,1))
    vector_elevaciones=np.zeros((dim_vector,1))
    alpha=dim_vector/len(levels)
    for i in range(len(levels)):
        for j in range(360):
            for k in range(480):
                vector_rangos[int(alpha*i + ((j+1)*k + (480-k)*j)),0] = rango_mat[j,k,i]
                vector_azimuts[int(alpha*i + ((j+1)*k + (480-k)*j)),0] = azimuth_mat[j,k,i]
                vector_elevaciones[int(alpha*i + ((j+1)*k + (480-k)*j)),0] = level_mat[j,k,i]
                
    posiciones_todo_el_dominio=np.zeros((dim_vector,3))
    
    for i in range(len(vector_rangos)):
        posiciones_todo_el_dominio[i,0]=vector_elevaciones[i]
        posiciones_todo_el_dominio[i,1]=vector_azimuts[i]
        posiciones_todo_el_dominio[i,2]=vector_rangos[i]
        
    return posiciones_todo_el_dominio

def puntos_de_interes(var,umbral):
    import numpy as np
    if umbral>=0:
        
        acim=np.where(var>=umbral)[0]
        radios=np.where(var>=umbral)[1]
        elev=np.where(var>=umbral)[2]
    
        return acim, radios, elev
    
    else:
        acim=np.where(var<=umbral)[0]
        radios=np.where(var<=umbral)[1]
        elev=np.where(var<=umbral)[2]
    
        return acim, radios, elev

def xy_latlon(radar,undef):
    
    import numpy as np

    x3d = order_variable ( radar , 'x' , radar.gate_x['data'].fill_value  ) 
    x3d = x3d[0]
    y3d = order_variable ( radar , 'y' , radar.gate_y['data'].fill_value  )  
    y3d = y3d[0]
    z3d = order_variable ( radar , 'altitude' , undef ) 
    z3d = z3d[0] 
        
    lat_y3d=(y3d/111000)+float(radar.latitude['data'][0])
    lon_x3d=(x3d/(111000*np.cos(lat_y3d*np.pi/180))) + float(radar.longitude['data'][0])

    return lat_y3d, lon_x3d

def funcion_score(variable,umbral,lat_lon_variable, lado):
    import numpy as np
    
    #score=np.zeros(len(variable))
    x1=np.zeros(len(variable))
    y1=np.zeros(len(variable))
    x2=np.zeros(len(variable))
    y2=np.zeros(len(variable))
    xo=np.zeros(len(variable))
    yo=np.zeros(len(variable))
    
    boxes=np.zeros((len(variable),6))
    i=0
    while i<len(variable):
        # #if shear[i]<0:
        # if abs(umbral) <= abs(variable[i]) < abs(umbral) + 0.1:
        #     score[i] = 7
        # elif abs(umbral) + 0.1 <= abs(variable[i]) < abs(umbral) + 0.2:
        #     score[i] = 8  
        # elif abs(umbral) + 0.2 <= abs(variable[i]) < abs(umbral) + 0.3:
        #     score[i] = 9 
        # elif abs(umbral) + 0.3 <= abs(variable[i]):
        #     score[i] = 10
        #construyo las cajas de lado tamaño xx°lat-lon centradas en el punto i
        
        #x1 corresponde al lado superior
        x1[i]= lat_lon_variable[0][i] + lado
        #y1 corresponde al lado izquierdo
        y1[i]= lat_lon_variable[1][i] - lado
        #x2 corresponde al lado inferior
        x2[i]= lat_lon_variable[0][i] - lado
        #y2 corresponde al lado derecho
        y2[i]= lat_lon_variable[1][i] + lado
        #centroide 
        xo[i]=lat_lon_variable[0][i]
        yo[i]=lat_lon_variable[1][i]
        
        boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3],boxes[i,4],boxes[i,5]=x1[i],y1[i],x2[i],y2[i],xo[i],yo[i] 
        i=i+1
    
    
    return boxes#score, boxes

def NMS(boxes, IoU, scores):
    import numpy as np
  	 #if there are no boxes, return an empty list

    if len(boxes) == 0:
        return []
      	# initialize the list of picked indexes
    
    pick = []
    
      	# grab the coordinates of the bounding boxes
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # compute the area of the bounding boxes and sort the bounding
    	# boxes by the bottom-right y-coordinate of the bounding box
    
    area = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
    #Voy a ponderar la eleccion del cuadro segun el score asignado a ese punto
    idxs = np.argsort(scores)
    # indexfile=[]
    # keep looping while some indexes still remain in the indexes
    	# list
    
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
    		# value to the list of picked indexes, then initialize
    		# the suppression list (i.e. indexes that will be deleted)
    		# using the last index
        last = len(idxs) - 1
        i = idxs[last]
        # indexfile.append(i)
        pick.append(i)
        suppress = [last]
       		# loop over all indexes in the indexes list
    
        for pos in range(0, last):
    
      			# grab the current index
    
            j = idxs[pos]
            
            # find the largest (x, y) coordinates for the start of
    			# the bounding box and the smallest (x, y) coordinates
    			# for the end of the bounding box
            
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            
      			# compute the width and height of the bounding box
            w= max(0, abs(xx2 - xx1) + 1)
            h = max(0, abs(yy2 - yy1) + 1)
    
            # compute the ratio of overlap between the computed
    			# bounding box and the bounding box in the area list
    
            overlap = float(w * h) / area[j]
            
            # if there is sufficient overlap, suppress the
    			# current bounding box
            
            if overlap > IoU:
                suppress.append(pos)
                # delete all indexes from the index list that are in the 
                #supression list
        idxs = np.delete(idxs, suppress)
    return pick


def grafica_ppi(radar, levels, x3d, y3d, z3d, v3d, vm3d, dbz3d, vil3d,shearmean3d, undef, rutasal, fecha, hora):
    
    import numpy as np
    import matplotlib.colors as colors
    from matplotlib import pyplot as plt 
    
#dominio de interes
    dx=240000
    dy=240000
    xc=0
    yc=0
    
    nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    #GRAFICO USANDO MATPLOTLIB DIRECTAMENTE SIN PYART.
    for ilev , my_lev in enumerate(levels):
    #for ilev in range(6,7) :
        ele=str(levels[ilev])
        x2d = x3d[:,:,ilev]
        y2d = y3d[:,:,ilev]
        z2d = z3d[:,:,ilev]
    
        v2d = np.copy(v3d[:,:,ilev])
        vm2d = np.copy(vm3d[:,:,ilev])
        dbz2d = np.copy(dbz3d[:,:,ilev])
        vil2d = np.copy(vil3d[:,:,ilev])
        shearmean2d = np.copy(shearmean3d[:,:,ilev])

        dbz2d[dbz2d==undef]=np.nan
        v2d[v2d==undef]=np.nan
        vm2d[vm2d==undef]=np.nan
        #Defino una mascara que me permite calcular las propiedades del campo de viento sobre la region de interes.
        mask = np.logical_and(x2d > xc-dx/2 , x2d < xc+dx/2 ) 
        mask = np.logical_and( mask , y2d > yc-dy/2 )
        mask = np.logical_and( mask , y2d < yc+dy/2)
        mean_vr = np.nanmean( v2d[mask] )
    
    
        #Creo la imagen PPI
        fig=plt.figure(figsize=(45,7))
        plt.subplot(131)
        plt.pcolor(x2d,y2d,dbz2d,vmin=0,vmax=70,cmap='gist_ncar')
        plt.axis([xc-dx/2 , xc+dx/2 , yc-dy/2 , yc+dy/2])
        plt.grid()
    
        plt.subplot(132)
        plt.pcolor(x2d,y2d,v2d-mean_vr,vmin=-nyq/5,vmax=nyq/5,cmap='bwr')
        plt.contour(x2d,y2d,vil2d,levels=[5,10,15])
        plt.axis([xc-dx/2 , xc+dx/2 , yc-dy/2 , yc+dy/2])
        plt.grid()
    
        plt.subplot(133)
        plt.pcolor(x2d,y2d,shearmean2d,vmin=-0.003,vmax=0.003,cmap='bwr')
        plt.contour(x2d,y2d,vil2d,levels=[5,10,15])
        plt.axis([xc-dx/2 , xc+dx/2 , yc-dy/2 , yc+dy/2])
        plt.grid()
        plt.show()
        #plt.savefig(rutasal + '/' + fecha + '_' + hora + '_' + ele + '_' +"firmas.png" , dpi=300)
        
    return print('guardando figuras')


def interAreas(lat_lon_var1,lat_lon_var2):
    import numpy as np

    if len(lat_lon_var1[0])>len(lat_lon_var2[0]):
        lat_intersec=np.zeros(len(lat_lon_var1[0]))
        lon_intersec=np.zeros(len(lat_lon_var1[1]))
        for i in range(len(lat_lon_var1[0])):
            for j in range(len(lat_lon_var2[0])):
                if lat_lon_var1[0][i]==lat_lon_var2[0][j]:
                    lat_intersec[i]=lat_lon_var1[0][i]
                if lat_lon_var1[1][i]==lat_lon_var2[1][j]:
                    lon_intersec[i]=lat_lon_var1[1][i]
                    
    else:
        lat_intersec=np.zeros(len(lat_lon_var2[0]))
        lon_intersec=np.zeros(len(lat_lon_var2[1]))
        for i in range(len(lat_lon_var2[0])):
            for j in range(len(lat_lon_var1[0])):
                if lat_lon_var2[0][i]==lat_lon_var1[0][j]:
                    lat_intersec[i]=lat_lon_var2[i]
                if lat_lon_var2[1][i]==lat_lon_var1[1][j]:
                    lon_intersec[i]=lat_lon_var2[1][i]
                
    return lat_intersec, lon_intersec


def grafica_ppi_latlon(radar, levels, lon_3d, lat_3d, z3d, v3d, vm3d, dbz3d, vil3d,shearmean3d, shear_index,undef, rutasal, fecha, hora, marcadores):
    
    import numpy as np
    from matplotlib import pyplot as plt 
    
  
    nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    #GRAFICO USANDO MATPLOTLIB DIRECTAMENTE SIN PYART.
    #for ilev , my_lev in enumerate(levels):
    for ilev in range(0,2) :
        ele=str(levels[ilev])
        lon_2d = lon_3d[:,:,ilev]
        lat_2d = lat_3d[:,:,ilev]
        z2d = z3d[:,:,ilev]
    
        v2d = np.copy(v3d[:,:,ilev])
        vm2d = np.copy(vm3d[:,:,ilev])
        dbz2d = np.copy(dbz3d[:,:,ilev])
        vil2d = np.copy(vil3d[:,:,ilev])
        shearmean2d = np.copy(shearmean3d[:,:,ilev])

        dbz2d[dbz2d==undef]=np.nan
        v2d[v2d==undef]=np.nan
        vm2d[vm2d==undef]=np.nan

        #Creo la imagen PPI
        fig=plt.figure(figsize=(8,18))
        ax1=plt.subplot(311)
        plt.pcolor(lon_2d,lat_2d,dbz2d,vmin=0,vmax=70,cmap='gist_ncar');plt.colorbar()
        ax1.axis([np.min(lon_2d) , np.max(lon_2d) , np.min(lat_2d) , np.max(lat_2d)])
        ax1.grid()
        ax1.set_title('Horizontal Equivalent Reflectivity (dBZ)'+'  ' +str(fecha) + '  ' + str(hora[0:4]) + '(UTC)' + '  ' + 'sweep=' + str(ele), fontsize=12)
        ax1.tick_params(axis='both', labelsize=15)
        for i in range(len(marcadores)):
            plt.vlines(marcadores[i,1],marcadores[i,2],marcadores[i,0], ls='-', color='b',linewidth=2.5) 
            plt.hlines(marcadores[i,0],marcadores[i,1],marcadores[i,3], ls='-', color='b',linewidth=2.5)
            plt.vlines(marcadores[i,3],marcadores[i,2],marcadores[i,0], ls='-', color='b',linewidth=2.5) 
            plt.hlines(marcadores[i,2],marcadores[i,1],marcadores[i,3], ls='-', color='b',linewidth=2.5)
        
        
        ax2=plt.subplot(312)
        plt.pcolor(lon_2d,lat_2d,v2d,vmin=-nyq/5,vmax=nyq/5,cmap='bwr');plt.colorbar()
        ax2.contour(lon_2d,lat_2d,vil2d,levels=[5,10,15])
        ax2.axis([np.min(lon_2d) , np.max(lon_2d) , np.min(lat_2d) , np.max(lat_2d)])
        ax2.grid()
        ax2.set_title('Corrected Radial Velocity (m/s)' +'  ' +str(fecha) + '  ' + str(hora[0:4]) + '(UTC)' + '  ' + 'sweep=' + str(ele),fontsize=12)
        ax2.tick_params(axis='both', labelsize=15)
        for i in range(len(marcadores)):
            plt.vlines(marcadores[i,1],marcadores[i,2],marcadores[i,0], ls='-', color='b',linewidth=2.5) 
            plt.hlines(marcadores[i,0],marcadores[i,1],marcadores[i,3], ls='-', color='b',linewidth=2.5)
            plt.vlines(marcadores[i,3],marcadores[i,2],marcadores[i,0], ls='-', color='b',linewidth=2.5) 
            plt.hlines(marcadores[i,2],marcadores[i,1],marcadores[i,3], ls='-', color='b',linewidth=2.5)
        
    
        ax3=plt.subplot(313)
        plt.pcolor(lon_2d,lat_2d,shearmean2d,vmin=-0.003,vmax=0.003,cmap='bwr');plt.colorbar()
        ax3.contour(lon_2d,lat_2d,vil2d,levels=[5,10,15])
        ax3.axis([np.min(lon_2d) , np.max(lon_2d) , np.min(lat_2d) , np.max(lat_2d)])
        ax3.grid()
        ax3.set_title('Azimuthal Shear local mean (1/s)' +'  ' +str(fecha) + '  ' + str(hora[0:4]) + '(UTC)' + '  ' + 'sweep=' + str(ele), fontsize=12)
        ax3.tick_params(axis='both', labelsize=15)
        for i in range(len(marcadores)):
            plt.vlines(marcadores[i,1],marcadores[i,2],marcadores[i,0], ls='-', color='b',linewidth=2.5) 
            plt.hlines(marcadores[i,0],marcadores[i,1],marcadores[i,3], ls='-', color='b',linewidth=2.5)
            plt.vlines(marcadores[i,3],marcadores[i,2],marcadores[i,0], ls='-', color='b',linewidth=2.5) 
            plt.hlines(marcadores[i,2],marcadores[i,1],marcadores[i,3], ls='-', color='b',linewidth=2.5)
        
    
        plt.savefig(rutasal + '/' + fecha + '_' + hora + '_' + ele + '_' +"firmas_geo.png" , dpi=300)
    
    vil2d = np.copy(vil3d[:,:,0])
    dbz2d = np.copy(dbz3d[:,:,0])
    shear_indexs = local_mean( shear_index , 3 , 3 , undef ) 
    vil2ds = local_mean( vil2d , 3 , 3 , undef )
     
    fig=plt.figure(figsize=(9,7))
    plt.pcolor(lon_2d,lat_2d,shear_indexs,cmap='bwr',vmin=-0.4,vmax=0.4);plt.colorbar()
    plt.contour(lon_2d,lat_2d,vil2ds,levels=[10,30,50],colors='y')
    plt.contour(lon_2d,lat_2d,dbz2d,levels=[40,50,60],colors='k')

    for i in range(len(marcadores)):
        plt.vlines(marcadores[i,1],marcadores[i,2],marcadores[i,0], ls='-', color='b',linewidth=2.5) 
        plt.hlines(marcadores[i,0],marcadores[i,1],marcadores[i,3], ls='-', color='b',linewidth=2.5)
        plt.vlines(marcadores[i,3],marcadores[i,2],marcadores[i,0], ls='-', color='b',linewidth=2.5) 
        plt.hlines(marcadores[i,2],marcadores[i,1],marcadores[i,3], ls='-', color='b',linewidth=2.5)
    
    plt.grid()
    plt.title('VIL and IndexShear' +'  ' +str(fecha) + '  ' + str(hora[0:4]) + '(UTC)', fontsize=15)

    plt.savefig(rutasal + '/' + fecha + '_' + hora + '_' + ele + '_' + "IndexShear.png" , dpi=300)

    
    return print('guardando figuras')  
    
    
    
    
    

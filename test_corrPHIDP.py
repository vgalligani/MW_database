from matplotlib import cm;
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pyart
from matplotlib.colors import ListedColormap
import math
import wradlib as wrl    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def set_plot_settings(var): 
    """
    -------------------------------------------------------------
    OUT    [units, cmap, vmin, vmax, mac, int, under] todas de importancia
                    para colormpas de las variables del radar.      
    IN     var      Zhh, Zdr, Kdp, Ah, phidp, rhohv, Doppler Vel. 
    -------------------------------------------------------------
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
def get_sys_phase_simple(radar):

    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]

    phases_nlev = []
    for nlev in range(radar.nsweeps-1):
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats  = radar.gate_latitude['data'][start_index:end_index]
        lons  = radar.gate_longitude['data'][start_index:end_index]
        TH    = radar.fields['TH']['data'][start_index:end_index]
        TV    = radar.fields['TV']['data'][start_index:end_index]
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
          #if (rho_h[j]<0.7):
            phiphi[i,j]  = np.nan 
            rho[i,j]     = np.nan   
	
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
def firstNonNan(listfloats):
  for item in listfloats:
    if math.isnan(item) == False:
      return item

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def correct_PHIDP_KDP(radar, options, nlev, azimuth_ray, diff_value):

    sys_phase = get_sys_phase_simple(radar)

    dphi, uphi, corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], radar.fields['TH']['data'], sys_phase, 280)
    #------------	
    radar.add_field_like('RHOHV','corrPHIDP', corr_phidp, replace_existing=True)

    # Y CALCULAR KDP! 
    calculated_KDP = wrl.dp.kdp_from_phidp(corr_phidp, winlen=options['window_calc_KDP'], dr=(radar.range['data'][1]-radar.range['data'][0])/1e3, 
					   method='lanczos_conv', skipna=True)	
    radar.add_field_like('RHOHV','corrKDP', calculated_KDP, replace_existing=True)
	
    #- Get nlev PPI
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]
	
    #- EJEMPLO de azimuth
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[azimuth_ray]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()

    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    rhoHV = radar.fields['RHOHV']['data'][start_index:end_index]
    PHIDP = radar.fields['PHIDP']['data'][start_index:end_index]
    KDP   = radar.fields['KDP']['data'][start_index:end_index]
    ZH    = radar.fields['TH']['data'][start_index:end_index]

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
    pcm1 = axes[0,2].pcolormesh(lons, lats, KDP, cmap=cmap, 
			  vmin=vmin, vmax=vmax)
    axes[0,2].set_title('KDP radar nlev '+str(nlev)+' PPI')
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
    THH =  radar.fields['TH']['data'][start_index:end_index]
    pcm1 = axes[1,0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, 
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
    #fig.savefig(options['fig_dir']+'PPIs_KDPcorr'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)   
    #plt.close()

    #----------------------------------------------------------------------
    #-figure
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True,figsize=[14,10])
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['RHOHV']['data'][start_index:end_index][filas,:])*100, '-k', label='RHOHV')
    axes[0].plot(radar.range['data']/1e3, np.ravel(radar.fields['TH']['data'][start_index:end_index][filas,:]), '-r', label='ZH')
    axes[0].legend()
    axes[1].plot(radar.range['data']/1e3, np.ravel(radar.fields['PHIDP']['data'][start_index:end_index][filas,:]), 'or', label='obs. phidp')
    axes[1].plot(radar.range['data']/1e3, np.ravel(dphi[start_index:end_index][filas,:]), '*b', label='despeckle phidp'); 
    axes[1].plot(radar.range['data']/1e3, np.ravel(uphi[start_index:end_index][filas,:]), color='darkgreen', label='unfolded phidp');
    axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]+sys_phase), color='magenta', label='phidp corrected');
    axes[1].plot(radar.range['data']/1e3, np.ravel(corr_phidp[start_index:end_index][filas,:]), color='purple', label='phidp corrected-sysphase');
    axes[1].legend()
    axes[2].plot(radar.range['data']/1e3, np.ravel(calculated_KDP[start_index:end_index][filas,:]), color='k', label='Calc. KDP');
    axes[2].plot(radar.range['data']/1e3, np.ravel(radar.fields['KDP']['data'][start_index:end_index][filas,:]), color='gray', label='Obs. KDP');
    axes[2].legend()
    #axes[0].set_xlim([50, 120])
    #axes[1].set_xlim([50, 120])
    #axes[2].set_xlim([50, 120])
    axes[2].set_ylim([-1, 5])
    axes[2].grid(True) 
    axes[2].plot([0, 300], [0, 0], color='darkgreen', linestyle='-') 
    #fig.savefig(options['fig_dir']+'PHIcorrazi'+'nlev'+str(nlev)+'.png', dpi=300,transparent=False)    
    #plt.close()

    return radar

#------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def run_general_case(options, azimuths_oi, xlims_xlims_input, xlims_mins_input):

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
	
    radar = correct_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280)
    
    return
  
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def run_Zhnan(options, azimuths_oi, xlims_xlims_input, xlims_mins_input):

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
    	
    #- Get nlev PPI
    nlev = 0
    start_index = radar.sweep_start_ray_index['data'][nlev]
    end_index   = radar.sweep_end_ray_index['data'][nlev]

    #- EJEMPLO de azimuth
    azimuths = radar.azimuth['data'][start_index:end_index]
    target_azimuth = azimuths[options['azimuth_ray']]
    filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
    
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    rhoHV = radar.fields['RHOHV']['data'][start_index:end_index]
    PHIDP = radar.fields['PHIDP']['data'][start_index:end_index]
    KDP   = radar.fields['KDP']['data'][start_index:end_index]
    ZH    = radar.fields['TH']['data'][start_index:end_index]

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
    axes[0,1].set_title('OBS Phidp radar nlev '+str(nlev)+' PPI')
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

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    THH =  radar.fields['TH']['data'][start_index:end_index]
    pcm1 = axes[1,0].pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, 
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

    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes[1,1].pcolormesh(lons, lats, np.isnan(radar.fields['TH']['data'][start_index:end_index]))
    axes[1,1].set_title('ZH nlev '+str(nlev)+' PPI')
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


	
    return
  
	

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main(): 
	
	rfile_rma3 = 'RMA3/'+'cfrad.20190305_124638.0000_to_20190305_125231.0000_RMA3_0200_01.nc'
	opts = {'xlim_min': -63, 'xlim_max': -58, 'ylim_min': -27, 'ylim_max': -23, 'ylim_max_zoom': -24.5, 'ZDRoffset': 3, 
	    'rfile': rfile_rma3, 
	    'window_calc_KDP': 7, 'azimuth_ray': 210, 
	    'x_supermin':-63, 'x_supermax':-58, 'y_supermin':-27, 'y_supermax':-23, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190305_RMA3/', 'radar_name':'RMA3'}
	run_general_case(opts, [176,210,30], [150, 200, 150] , [0, 0, 0] )
        run_Zhnan(opts, [176,210,30], [150, 200, 150] , [0, 0, 0] )
				
	rfile_rma1 = 'RMA1/'+'cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
	opts = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5, 
	    'ZDRoffset': 4, 'ylim_max_zoom':-30.5, 'rfile': rfile_rma1, 
	    'window_calc_KDP': 7, 'azimuth_ray': 210, 'x_supermin':-65, 'x_supermax':-64,
	    'y_supermin':-33, 'y_supermax':-31.5, 'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20180208_RMA1/', 
	   'radar_name':'RMA1'}			
	run_general_case(opts, [356,220,192], [60, 100, 150], [10, 40, 60] )
        run_Zhnan(opts, [356,220,192], [60, 100, 150], [10, 40, 60] )
				
				

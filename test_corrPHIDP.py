from matplotlib import cm;
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pyart
from matplotlib.colors import ListedColormap

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
def correct_phidp(phi, rho, zh, sys_phase, diferencia):

    phiphi = phi.copy()
    ni = phi.shape[0]
    nj = phi.shape[1]
    for i in range(ni):
        rho_h = rho[i,:]
        zh_h = zh[i,:]
        for j in range(nj):
          if (rho_h[j]<0.7) or (zh_h[j]<30):
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
def correct_PHIDP_KDP(radar, options, nlev, azimuth_ray, diff_value, tfield_ref, alt_ref):

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

    radar = correct_PHIDP_KDP(radar, options, nlev=0, azimuth_ray=options['azimuth_ray'], diff_value=280, tfield_ref=tfield_ref, alt_ref=alt_ref)
    
    return
  

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main(): 
  
  
opts = {'xlim_min': -63, 'xlim_max': -58, 'ylim_min': -27, 'ylim_max': -23, 'ylim_max_zoom': -24.5, 'ZDRoffset': 3, 
	    'rfile': rfile_rma3, 
	    'window_calc_KDP': 7, 'azimuth_ray': 210, 
	    'x_supermin':-63, 'x_supermax':-58, 'y_supermin':-27, 'y_supermax':-23, 
	    'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20190305_RMA3/', 'radar_name':'RMA3'}

run_general(opts, [176,210,30], [150, 200, 150] , [0, 0, 0] )

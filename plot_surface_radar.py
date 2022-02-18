#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:26:28 2022
@author: victoria.galligani
"""
# Code that plots initially Zh for the coincident RMA5, RMA1 and PARANA observations for
# case studies of interest. ver .doc "casos granizo" para más info. 

from os import listdir
import pyart

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
        cmap = fun.colormaps('ref')
        vmin = 0
        vmax = 60
        max = 60.01
        intt = 5
        under = 'white'
        over = 'white'
    elif var == 'Zhh_2':
        units = 'Zh [dBZ]'
        cmap = fun.colormaps('ref2')
        vmin = 0
        vmax = 60
        max = 60.01
        intt = 5
        under = 'white'
        over = 'black'
    elif var == 'Zdr':
        units = '[dBZ]'
        cmap = fun.colormaps('zdr')
        vmin = -2
        vmax = 5.5
        max = 5.51
        intt = 0.5
        under = 'white'
        over = 'white'
    elif var == 'Kdp':
        units = '[deg/km]'
        vmin = -0.5
        vmax = 0.5
        max = 0.51
        intt = 0.1
        N = (vmax-vmin)/intt
        cmap = fun.discrete_cmap(10, 'jet')
        under = 'white'
        over = 'black'
    elif var == 'Ah':
        units = '[dB/km]'
        vmin = 0
        vmax = 0.5
        max = 0.51
        intt = 0.05
        N = (vmax-vmin)/intt
        cmap = fun.discrete_cmap(N, 'brg_r')
        under = 'black'
        over = 'white'
    elif var == 'phidp':
        units = 'deg'
        vmin = 60
        vmax = 180
        max = 180.1
        intt = 10
        N = (vmax-vmin)/intt
        print(N)
        cmap = fun.discrete_cmap(int(N), 'jet')
        under = 'white'
        over = 'white'
    elif var == 'rhohv':
        units = ''
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
        vmin = -15
        vmax = 15
        max = 15.01
        intt = 5
        under = 'white'
        over = 'white'    
        
    return units, cmap, vmin, vmax, max, intt, under, over

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_ppi_Zh(): 

    
    
    
    return
 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  
if __name__ == '__main__':

  files_RMA5 = ['cfrad.20181001_231430.0000_to_20181001_232018.0000_RMA5_0200_01.nc',
              'cfrad.20181001_231430.0000_to_20181001_232018.0000_RMA5_0200_01.nc',
              'cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc']

  files_RMA1 = ['cfrad.20171027_034647.0000_to_20171027_034841.0000_RMA1_0123_02.nc',
              'cfrad.20171209_005909.0000_to_20171209_010438.0000_RMA1_0123_01.nc',
              'cfrad.20180208_205455.0000_to_20180208_205739.0000_RMA1_0201_02.nc',
              'cfrad.20180209_063643.0000_to_20180209_063908.0000_RMA1_0201_03.nc',
              'cfrad.20181111_124509.0000_to_20181111_125150.0000_RMA1_0301_01.nc', 
              'cfrad.20181214_030436.0000_to_20181214_031117.0000_RMA1_0301_01.nc',
              'cfrad.20190102_204413.0000_to_20190102_204403.0000_RMA1_0301_02.nc',
              'cfrad.20190224_061413.0000_to_20190224_061537.0000_RMA1_0301_02.nc',
              'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc']

  files_PAR = ['2018032407543300dBZ.vol',
             '2018100907443400dBZ.vol',
             '2018121403043200dBZ.vol',
             '2019021109400500dBZ.vol',
             '2019022315143100dBZ.vol',
             '2020121903100500dBZ.vol']

  # start w/ RMA1
  for ifiles in range(len(files_RMA1)):
    folder = str(files_RMA1[ifiles][6:14])
    if folder[0:4] == '2021':
      nfile  = '/relampago/datos/salio/RADAR/RMA1/'+ folder + '/' + files_RMA1[ifiles]
    else:
      yearfolder = folder[0:4]
      nfile  = '/relampago/datos/salio/RADAR/RMA1/'+ yearfolder + '/' + folder + '/' + files_RMA1[ifiles]
    print('reading: ' + nfile)
    radar = pyart.io.read(nfile) 
    # Plot Zh, ZDR, PhiDP, RHO_hv: attenuation corrected and phidp unfolded etc. 
    
  


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:52:17 2023

@author: vgalligani
"""
#check resolution of gates: 
import pyart

r_dir    = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'

# RMA1 - 8/2/2018
rfile    = 'RMA1/cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
radar = pyart.io.read(r_dir+rfile)
  
# CSPR2 - 11/11/2018    
rfile     = 'CSPR2_data/corcsapr2cmacppiM1.c1.20181111.130003.nc' #'corcsapr2cfrppiM1.a1.20181111.130003.nc'
radar = pyart.io.read(r_dir+rfile)

# DOW7 - 14/12/2018 
rfile     = 'DOW7/cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc'
radar = pyart.io.read(r_dir+rfile)

# RMA1 - 03/08/2019
rfile = 'RMA1/cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc'
radar = pyart.io.read(r_dir+rfile)

# RMA5 - 15/08/2020
rfile = 'RMA5/cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc' 
radar = pyart.io.read(r_dir+rfile)

# RMA4 - 09/02/2018
rfile = 'RMA4/cfrad.20180209_200449.0000_to_20180209_201043.0000_RMA4_0200_01.nc' ' 
radar = pyart.io.read(r_dir+rfile)

# RMA4 - 31/10/2018
rfile    = 'RMA4/cfrad.20181031_010936.0000_to_20181031_011525.0000_RMA4_0200_01.nc' 
radar = pyart.io.read(r_dir+rfile)

# RMA3 - 05/03/2019
rfile     = 'RMA3/cfrad.20190305_124638.0000_to_20190305_125231.0000_RMA3_0200_01.nc'
radar = pyart.io.read(r_dir+rfile)

# RMA4 - 09/02/2019
rfile    = 'RMA4/cfrad.20190209_192724.0000_to_20190209_193317.0000_RMA4_0200_01.nc' 
radar = pyart.io.read(r_dir+rfile)


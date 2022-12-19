import pyart
import numpy as np

ptype = 'totally_random'
version = 3.
T_grid = np.array([240, 250, 260, 270, 280, 290, 300])
za_grid = np.arange(0, 181, 10)
aa_grid = np.arange(0, 181, 10)        
                  
scat_data = pyarts.scattering.SingleScatteringData()

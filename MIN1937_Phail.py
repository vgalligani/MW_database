import matplotlib.pyplot as plt
from numpy import genfromtxt;
import numpy as np

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
  
GMI_BC19_casestudies = genfromtxt("/home/victoria.galligani/Work/Studies/Hail_MW/GMI_BC2019_0_hailcases_ARGE.txt", skip_header=1, delimiter='')

Phail    = []
MIN19PCT = [] 
MIN37PCT = []

for i in range(GMI_BC19_casestudies.shape[0]):
  Phail.append(GMI_BC19_casestudies[i,7]*100)
  MIN19PCT.append(GMI_BC19_casestudies[i,9])
  MIN37PCT.append(GMI_BC19_casestudies[i,10])  
  
Phail_    = np.array(Phail)
MIN19PCT_ = np.array(MIN19PCT)
MIN37PCT_ = np.array(MIN37PCT)
  
plt.matplotlib.rc('font', family='serif', size = 12)
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12  

fig = plt.figure(figsize=(10,10)) 
plt.plot(MIN19PCT_[np.where(Phail_<50)], MIN37PCT_[np.where(Phail_<50)], 'o', markersize=4, color='skyblue')
im = plt.scatter(MIN19PCT_[np.where(Phail_>=50)], MIN37PCT_[np.where(Phail_>=50)], c=Phail_[np.where(Phail_>=50)], s=30, 
                 marker='o', vmin=50, vmax=100, cmap=discrete_cmap(5, 'YlOrRd'))
                
plt.plot(np.arange(50,300,10), np.arange(50,300,10), '--k')
plt.grid(True)
plt.xlabel('MIN19PCT')
plt.ylabel('MIN37PCT')
fig.colorbar(im, label='P$_{hail}$ %')   

plt.plot(242.9253, 207.4241, 'kv', markersize=15, markerfacecolor='None') # coi2 08/02/2018
plt.plot(270.2810, 238.8242, 'Pk', markersize=15, markerfacecolor='None') # coi1 08/02/2018

#plt.plot(241.5902, 181.1631, '*k', markersize=20, markerfacecolor='None') # este estaba mal

plt.plot(237.4247, 151.6656, 'dk', markersize=15, markerfacecolor='None') #
plt.plot(198.3740, 111.7183, 'Dk', markersize=15, markerfacecolor='None') #
plt.plot(244.4909, 174.1353, '^k', markersize=15, markerfacecolor='None') #

plt.plot(249.4, 190.1, '',   markersize=15, markerfacecolor='None') # 11/11/2018 contour north
plt.plot(273.5, 250.8, '',   markersize=15, markerfacecolor='None') # 11/11/2018 contour south

plt.plot(260.0201, 201.8675, '',  markersize=15, markerfacecolor='None') # 14/12/2018

plt.plot(271.6930, 241.9306, , '',  markersize=15, markerfacecolor='None') # 09/03/2019

plt.plot(273.2686, 241.5902, , '',  markersize=15, markerfacecolor='None') # 15/08/2020






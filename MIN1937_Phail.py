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

plt.plot(242.9253, 207.4241, 'k*', markersize=15, markerfacecolor='None')
plt.plot(270.2810, 238.8242, 'Pk', markersize=15, markerfacecolor='None')
plt.plot(241.5902, 181.1631, '*k', markersize=15, markerfacecolor='NOne')













plt.plot(242.9253, 207.4241, 'r*', markersize=20)
plt.plot(270.2810, 238.8242, 'rP', markersize=20)
plt.plot(241.5902, 181.1631, 'b*', markersize=20)
plt.plot(, color='darkgreen', marker='D', markersize=20)

plt.plot(, color='darkgreen', marker='d', markersize=20)
plt.plot(, color='darkgreen', marker='d', markersize=20)
plt.plot(, color='darkgreen', marker='d', markersize=20)
plt.plot(, color='darkgreen', marker='d', markersize=20)
plt.plot(, color='darkgreen', marker='d', markersize=20)



plt.grid(True)
plt.xlabel('MIN19PCT')
plt.ylabel('MIN37PCT')
fig.colorbar(im, label='P$_{hail}$ %')   
  

import matplotlib.pyplot as plt
from numpy import genfromtxt;
import numpy as np

GMI_BC19_casestudies = genfromtxt("/home/victoria.galligani/Work/Studies/Hail_MW/GMI_BC2019_0_hailcases_ARGE.txt", skip_header=1, delimiter='')

Phail    = []
MIN19PCT = [] 
MIN37PCT = []

for i in range(GMI_BC19_casestudies.shape[0]):
  Phail.append(GMI_BC19_casestudies[i,7]*100)
  MIN19PCT.append(GMI_BC19_casestudies[i,9])
  MIN37PCT.append(GMI_BC19_casestudies[i,10])  
  
  
plt.matplotlib.rc('font', family='serif', size = 12)
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12  

fig = plt.figure(figsize=(10,10)) 
im = plt.scatter(MIN19PCT, MIN37PCT, c=Phail, s=30, marker='o', vmin=0, vmax=100)
plt.plot(np.arange(50,300,10), np.arange(50,300,10), '--k')
plt.plot(242.9253, 207.4241, 'rx', markersize=20)
plt.plot(270.2810, 238.8242, 'ro', markersize=20)

plt.grid(True)
plt.xlabel('MIN19PCT')
plt.ylabel('MIN37PCT')
fig.colorbar(im, label='P$_{hail}$ %')   
  

import matplotlib.pyplot as plt
import numpy as np

def calc_sizeparam(freq, dmax):
  
  wavelength = 3E8/freq
  
  return (2*np.pi*dmax)/(2*wavelength)


fig = plt.figure(figsize=(8,9))   
plt.plot()

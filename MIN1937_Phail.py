import matplotlib.pyplot as plt
from numpy import genfromtxt;
import numpy as np

GMI_BC19_casestudies = genfromtxt("/home/victoria.galligani/Work/Studies/Hail_MW/GMI_BC2019_0_hailcases_ARGE.txt", skip_header=1, delimiter='')

Phail    = []
MIN19PCT = [] 
MIN37PC  = []

for i in range(GMI_BC19_casestudies.shape[0]):
  Phail.append()
  MIN19PCT.append()
  MIN37PC.append()  
  
  

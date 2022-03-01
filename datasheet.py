import numpy as np

# define general variables

dT_SN = 10 # 10 s SN time window
year = 3600*24*365


distance_0 = 10 # 10 kpc



# number of signal events for IceCube-Gen2 assuming a 27M SN progenitor (heavy)
sig_gen2_heavy = np.array([2.3E6, 1.84E5, 7E4, 3.35E4, 1.89E4, 1E4, 4.8E3, 2.03E3])#, 6.7E2, 1.77E2, 3.4E1, 5E0, 9E-1])

# number of signal events for IceCube-Gen2 assuming a 9.6M SN progenitor (light)
sig_gen2_light = np.array([1.33E6, 1.55E5, 4.1E4, 1.93E4, 1.05E4, 5.5E3, 2.679E3, 1.12E3])#, 3.7E2, 9.3E1, 1.6E1, 2E0, 0])

# total noise rate for IceCube-Gen2 assuming low noise mDOM glass
f_noise_gen2 = np.array([1.1E8, 3.1E4, 2.3E3, 8.5E1, 5.5E0, 2.94E-1, 1.6E-2, 4.9E-4])#, 3E-8])
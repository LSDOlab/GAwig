import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['text.usetex'] = True

# sns.set()

# h_inf_VAST = {
#     'alpha': [-10, -5, -1, 0, 1, 2, 3, 4, 5, 10, 15, 20],
#     'CL': [-0.42110446, -0.09460199, 0.16835239, 0.23406397, 0.29969863, 0.36522386, 0.43060729, 0.49581669, 0.56082006, 0.88160348, 1.19268576, 1.49053072],
#     'CDi': [0.0134704, -0.09460199, 0.00222084, 0.00426832, 0.00697743, 0.01034555, 0.014369, 0.01904301, 0.02436178, 0.06036207, 0.11102288, 0.17452073]
# }

h_inf_VAST = {
    'alpha': [-5, -1, 0, 1, 2, 3, 4, 5, 10, 15],
    'CL': [-0.09460199, 0.16835239, 0.23406397, 0.29969863, 0.36522386, 0.43060729, 0.49581669, 0.56082006, 0.88160348, 1.19268576],
    'CDi': [-0.09460199, 0.00222084, 0.00426832, 0.00697743, 0.01034555, 0.014369, 0.01904301, 0.02436178, 0.06036207, 0.11102288]
}

file_name='clarky_DegenGeom.polar'
clarky_polar_data = np.genfromtxt(file_name, skip_header=1)
clarky_alpha = clarky_polar_data[:,2]
clarky_CL = clarky_polar_data[:,4]
clarky_CDi = clarky_polar_data[:,6]

file_name='clarky_DegenGeom_AR=10.polar'
clarky_polar_data_AR_10 = np.genfromtxt(file_name, skip_header=1)
clarky_alpha_AR_10 = clarky_polar_data_AR_10[:,2]
clarky_CL_AR_10 = clarky_polar_data_AR_10[:,4]
clarky_CDi_AR_10 = clarky_polar_data_AR_10[:,6]

h_04_VAST = {
    'alpha': np.arange(-1, 6),
    'CL': [0.30990245, 0.49149974, 0.64657454, 0.79622889, 0.93467747, 1.06337391, 1.18488992],
    'CDi': [0.00195775, 0.00461891, 0.00831912, 0.01308144, 0.01895701, 0.02612612, 0.03520845]
}

h_08_VAST = {
    'alpha': np.arange(-1, 6),
    'CL': [0.25435121, 0.38152892, 0.50354866, 0.62033668, 0.73179546, 0.83781209, 0.93826911],
    'CDi': [0.00174519, 0.00382394, 0.00665834, 0.01022053, 0.01447744, 0.01939198, 0.02492516]
}

h_16_VAST = {
    'alpha': np.arange(-1, 6),
    'CL': [0.21219638, 0.30758088, 0.40092398, 0.49211593, 0.58104322, 0.66758893, 0.75163323],
    'CDi': [0.00170799, 0.00352471, 0.00596295, 0.00900023, 0.01261023, 0.01676272, 0.02142359]
}


plt.figure(1)
im = plt.imread('CL_plot.png')
implot = plt.imshow(im, aspect='auto', extent=[-9, 20, -0.08, 1.52])
plt.plot(h_inf_VAST['alpha'], h_inf_VAST['CL'], 'r-', linewidth=4, markersize=8, label='VAST OGE, AR = 4')
plt.plot(clarky_alpha, clarky_CL, 'g-', linewidth=4, label='OpenVSP OGE, AR = 4')
plt.plot(clarky_alpha_AR_10, clarky_CL_AR_10, 'c-', linewidth=4, label='OpenVSP OGE, AR = 10')
plt.xlabel(r'$\alpha $ $(^\circ)$', fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel(r'$C_L$', fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.legend(fontsize=15)


plt.figure(2)
im = plt.imread('CL_alpha_IGE_data_plot.png')
implot = plt.imshow(im, aspect='auto', extent=[-5.55, 5.4, -0.077, 1.21])
plt.plot(h_04_VAST['alpha'], h_04_VAST['CL'], 'r-', linewidth=4, label=r'$\frac{h_o}{\sqrt{S}} = 0.04$')
plt.plot(h_08_VAST['alpha'], h_08_VAST['CL'], 'g-', linewidth=4, label=r'$\frac{h_o}{\sqrt{S}} = 0.08$')
plt.plot(h_16_VAST['alpha'], h_16_VAST['CL'], 'c-', linewidth=4, label=r'$\frac{h_o}{\sqrt{S}} = 0.16$')
plt.xlabel(r'$\alpha $ $ (^\circ)$', fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel(r'$C_L$', fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.legend(loc='best', fontsize=20)


plt.show()
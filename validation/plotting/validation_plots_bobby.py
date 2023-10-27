import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['text.usetex'] = True

# sns.set()

h_04_VAST = {
    'alpha': np.arange(-1, 6),
    'CL': [0.38079322, 0.5864674, 0.7671628, 0.92459137, 1.0551881, 1.18689801, 1.28803569],
    'CDi': [0.00149855, 0.00388587, 0.00724706, 0.01151688, 0.01640831, 0.02101358, 0.02347888]
}

h_08_VAST = {
    'alpha': np.arange(-1, 6),
    'CL': [0.26844509, 0.4031922, 0.53313468, 0.65873544, 0.78118927, 0.90480628, 1.03130548],
    'CDi': [0.00170882, 0.00372931, 0.00650331, 0.01001494, 0.01426857, 0.01935542, 0.02511646]
}

h_16_VAST = {
    'alpha': np.arange(-1, 6),
    'CL': [0.22015347, 0.31856802, 0.41480924, 0.50877223, 0.60034937, 0.68943083, 0.77590527],
    'CDi': [0.0017355, 0.00355566, 0.00600321, 0.00905229, 0.01267335, 0.01683318, 0.02149488]
}

h_04_data = {
    'alpha': np.arange(-1, 6),
    'CL': [0.35, 0.54, 0.68, 0.81, 0.94, 1.06, 1.17],
    'CDi': []
}

h_08_data = {
    'alpha': np.arange(-1, 6),
    'CL': [0.35, 0.5, 0.61, 0.71, 0.82, 0.92, 1.01],
    'CDi': []
}

h_16_data = {
    'alpha': np.arange(-1, 6),
    'CL': [0.35, 0.46, 0.55, 0.65, 0.74, 0.82, 0.9],
    'CDi': []
}

if False:
    plt.figure(1)
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

fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)

ax1.plot(h_04_VAST['alpha'], h_04_VAST['CL'], 'r-', linewidth=3, markersize=8, label=r'VAST')
ax1.plot(h_04_data['alpha'], h_04_data['CL'], 'k*', linewidth=3, markersize=8, label=r'Data')
ax1.set_title(r'$\frac{h_o}{\sqrt{S}} = 0.04$', fontsize=20)
ax1.set_xticks(h_04_VAST['alpha'])
ax1.grid()

ax2.plot(h_08_VAST['alpha'], h_08_VAST['CL'], 'r-', linewidth=3, markersize=8, label=r'VAST')
ax2.plot(h_08_data['alpha'], h_08_data['CL'], 'k*', linewidth=3, markersize=8, label=r'Data')
ax2.set_title(r'$\frac{h_o}{\sqrt{S}} = 0.08$', fontsize=20)
ax2.set_xticks(h_08_VAST['alpha'])
ax2.grid()
ax2.legend(fontsize=20)

ax3.plot(h_16_VAST['alpha'], h_16_VAST['CL'], 'r-', linewidth=3, markersize=8, label=r'VAST')
ax3.plot(h_16_data['alpha'], h_16_data['CL'], 'k*', linewidth=3, markersize=8, label=r'Data')
ax3.set_title(r'$\frac{h_o}{\sqrt{S}} = 0.16$', fontsize=20)
ax3.set_xticks(h_16_VAST['alpha'])
ax3.set_yticks(np.arange(0, 1.41, 0.2))
ax3.grid()

fig.supxlabel(r'$\alpha $ $ (^\circ)$', fontsize=20)
fig.supylabel(r'$C_L$', fontsize=20)



# fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)

# ax1.plot(h_04_VAST['alpha'], h_04_VAST['CDi'], 'r-*', linewidth=3, markersize=8, label=r'VAST')
# # ax1.plot(h_04_data['alpha'], h_04_data['CD'], 'k-', linewidth=3, markersize=8, label=r'Data')
# ax1.set_title(r'$\frac{h_o}{\sqrt{S}} = 0.04$', fontsize=20)
# ax1.set_xticks(h_04_VAST['alpha'])
# ax1.grid()

# ax2.plot(h_08_VAST['alpha'], h_08_VAST['CDi'], 'r-*', linewidth=3, markersize=8, label=r'VAST')
# # ax2.plot(h_08_data['alpha'], h_08_data['CD'], 'k-', linewidth=3, markersize=8, label=r'Data')
# ax2.set_title(r'$\frac{h_o}{\sqrt{S}} = 0.08$', fontsize=20)
# ax2.set_xticks(h_08_VAST['alpha'])
# ax2.grid()
# ax2.legend(fontsize=20)

# ax3.plot(h_16_VAST['alpha'], h_16_VAST['CDi'], 'r-*', linewidth=3, markersize=8, label=r'VAST')
# # ax3.plot(h_16_data['alpha'], h_16_data['CD'], 'k-', linewidth=3, markersize=8, label=r'Data')
# ax3.set_title(r'$\frac{h_o}{\sqrt{S}} = 0.16$', fontsize=20)
# ax3.set_xticks(h_16_VAST['alpha'])
# # ax3.set_yticks(np.arange(0, 1.41, 0.2))
# ax3.grid()

# fig.supxlabel(r'$\alpha $ $ (^\circ)$', fontsize=20)
# fig.supylabel(r'$C_D$', fontsize=20)


plt.show()
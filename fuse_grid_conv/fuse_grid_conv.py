import numpy as np
import matplotlib.pyplot as plt
import pickle

nx = 14
ny = [20, 30]
nx_f = 21
nz_f = 5

# Loading data for case without fuselage
L_spanwise = []
D_i_spanwise = []
CL = []
CDi = []

for i in range(len(ny)):
    file_name = f'wing_nc_{int(nx)}_ns_{int(ny[i])}.pkl'
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()

    if i == 0:
        span_panels = data[0]
        span = span_panels[-1] - span_panels[0]
    L_spanwise.append(data[1])
    D_i_spanwise.append(data[2])
    CL.append(data[3])
    CDi.append(data[4])

# Loading data for case with fuselage
L_spanwise_f = []
D_i_spanwise_f = []
CL_f = []
CDi_f = []

for i in range(len(ny)):
    file_name = f'wing_nc_{int(nx)}_ns_{int(ny[i])}_fuse_nc_{nx_f}_ns_{nz_f}.pkl'
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()

    if i == 0:
        span_panels = data[0]
        span = span_panels[-1] - span_panels[0]
    L_spanwise_f.append(data[1])
    D_i_spanwise_f.append(data[2])
    CL_f.append(data[3])
    CDi_f.append(data[4])

# PLOTTING CL AND CDi
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.semilogx(ny, CL, '-*', label='No fuselage')
ax1.semilogx(ny, CL_f, '-*', label='With fuselage')
ax1.grid()
ax1.set_ylabel('CL', fontsize=12)
ax1.legend(loc='best')

ax2.semilogx(ny, CDi, '-*')
ax2.semilogx(ny, CDi_f, '-*')
ax2.grid()
ax2.set_xlabel('Number of span-wise panels', fontsize=12)
ax2.set_ylabel('CDi', fontsize=12)
ax2.set_xticks(ny)
ax2.set_xticklabels(ny, fontsize=12)

plt.show()
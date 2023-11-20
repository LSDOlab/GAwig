import numpy as np
import matplotlib.pyplot as plt
import pickle

nx = [1, 2, 5, 10, 15, 30, 45]
# nx = [1, 2, 5, 10, 15, 30]
ny = 40

L_spanwise = []
D_i_spanwise = []
CL = []
CDi = []

for i in range(len(nx)):
    print(i)
    file_name = f'wing_nc_{nx[i]}_ns_{ny}.pkl'
    print(file_name)
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

# PLOTTING CL AND CDi
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.semilogx(nx, CL, '-ok')
ax1.grid()
ax1.set_ylabel('CL', fontsize=12)

ax2.semilogx(nx, CDi, '-ok')
ax2.grid()
ax2.set_xlabel('Number of chord-wise panels', fontsize=12)
ax2.set_ylabel('CDi', fontsize=12)
ax2.set_xticks(nx)
ax2.set_xticklabels(nx, fontsize=12)

# PLOTTING LIFT AND INDUCED DRAG DISTRIBUTIONS
color = plt.cm.rainbow(np.linspace(0, 1, len(nx)))
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for i in range(len(nx)):
    ax1.plot(span_panels/span, L_spanwise[i], '-*', c=color[i], label=f'nx = {nx[i]}')
    ax2.plot(span_panels/span, D_i_spanwise[i], '-*', c=color[i])
    

ax1.grid()
ax1.set_ylabel('Sectional Lift (N)', fontsize=15)
ax1.legend(loc='best', fontsize=12)
ax1.tick_params(axis='y', labelsize=15)

ax2.grid()
ax2.set_xlabel('Normalized Spanwise Location', fontsize=15)
ax2.set_ylabel('Sectional Induced Drag (N)', fontsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

plt.show()
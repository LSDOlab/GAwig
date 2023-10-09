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

alpha_0_VAST = {
    'h_c': [0.025, 0.05, 0.1, 0.2, 0.3],
    'CL': [0.74054408, 0.65861803, 0.53520989, 0.42976839, 0.38656111],
    'CDi': [0.03189291, 0.02530936, 0.01960455, 0.01603359, 0.01496228]
}

alpha_2_VAST = {
    'h_c': [0.025, 0.05, 0.1, 0.2, 0.3],
    'CL': [0.74205211, 0.6833168, 0.606558, 0.52775925, 0.48917974],
    'CDi': [0.03739803, 0.03405486, 0.03016127, 0.02680117, 0.02550486]
}

alpha_4_VAST = {
    'h_c': [0.025, 0.05, 0.1, 0.2, 0.3],
    'CL': [0.75856851, 0.72480211, 0.67530953, 0.61653288, 0.58395969],
    'CDi': [0.0477304, 0.04558154, 0.04267959, 0.03970326, 0.03837312]
}

alpha_6_VAST = {
    'h_c': [0.025, 0.05, 0.1, 0.2, 0.3],
    'CL': [0.80320228, 0.78100294, 0.74633065, 0.70133052, 0.67424856],
    'CDi': [0.06118437, 0.05961771, 0.05734004, 0.05476861, 0.05351622]
}

alpha_8_VAST = {
    'h_c': [0.025, 0.05, 0.1, 0.2, 0.3],
    'CL': [0.86011375, 0.84443035, 0.81895498, 0.78389783, 0.76154637],
    'CDi': [0.07713017, 0.0759379, 0.07413237, 0.07197901, 0.07088079]
}

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(alpha_0_VAST['h_c'], alpha_0_VAST['CL'], 'k-s', linewidth=1, markersize=8, markerfacecolor='k', label=r'$C_L, \alpha$ = 0$^\circ$')
ax1.plot(alpha_2_VAST['h_c'], alpha_2_VAST['CL'], 'k-^', linewidth=1, markersize=8, markerfacecolor='k', label=r'$C_L, \alpha$ = 2$^\circ$')
ax1.plot(alpha_4_VAST['h_c'], alpha_4_VAST['CL'], 'k-v', linewidth=1, markersize=8, markerfacecolor='k', label=r'$C_L, \alpha$ = 4$^\circ$')
ax1.plot(alpha_6_VAST['h_c'], alpha_6_VAST['CL'], 'k-D', linewidth=1, markersize=8, markerfacecolor='k', label=r'$C_L, \alpha$ = 6$^\circ$')
ax1.plot(alpha_8_VAST['h_c'], alpha_8_VAST['CL'], 'k-o', linewidth=1, markersize=8, markerfacecolor='k', label=r'$C_L, \alpha$ = 8$^\circ$')
ax1.set_xlabel(r'$h/c$', fontsize=20)
ax1.set_xticks(np.arange(0,0.31,0.1), fontsize=20)
ax1.set_ylabel(r'$C_L$', fontsize=20)
ax1.set_yticks(np.arange(0,1.21,0.2), fontsize=20)
ax1.grid()
ax1.legend(fontsize=15, loc='upper left')

ax2.plot(alpha_0_VAST['h_c'], alpha_0_VAST['CDi'], 'k-s', linewidth=1, markersize=8, markerfacecolor='w', label=r'$C_{Di}, \alpha$ = 0$^\circ$')
ax2.plot(alpha_2_VAST['h_c'], alpha_2_VAST['CDi'], 'k-^', linewidth=1, markersize=8, markerfacecolor='w', label=r'$C_{Di}, \alpha$ = 2$^\circ$')
ax2.plot(alpha_4_VAST['h_c'], alpha_4_VAST['CDi'], 'k-v', linewidth=1, markersize=8, markerfacecolor='w', label=r'$C_{Di}, \alpha$ = 4$^\circ$')
ax2.plot(alpha_6_VAST['h_c'], alpha_6_VAST['CDi'], 'k-D', linewidth=1, markersize=8, markerfacecolor='w', label=r'$C_{Di}, \alpha$ = 6$^\circ$')
ax2.plot(alpha_8_VAST['h_c'], alpha_8_VAST['CDi'], 'k-o', linewidth=1, markersize=8, markerfacecolor='w', label=r'$C_{Di}, \alpha$ = 8$^\circ$')
ax2.set_xlabel(r'$h/c$', fontsize=20)
ax2.set_xticks(np.arange(0,0.31,0.1), fontsize=20)
ax2.set_ylabel(r'$C_{Di}$', fontsize=20)
ax2.set_yticks(np.arange(0,0.241,0.04), fontsize=20)
ax2.grid()
ax2.legend(fontsize=15)


plt.show()
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (8.3,3.2)

fontsize = 16

ld_oge = 47.4103145


ld_a0 = np.array([56.81463931,62.57153456,67.74550309,83.27284116,100.0148803,112.5213406,117.544589,87.68583256,])

hs = np.array([0.49212598,0.32808399,0.24606299,0.1230315,0.06151575,0.03075787,0.01537894,0.00768947,])

hsref = np.array([0.3,0.15,0.08,0.04,0.02,0.01,])
ldref = np.array([1.13,1.25,1.5,1.85,2.375,2.85,])

lda0_ldoge = ld_a0/ld_oge



fig, ax = plt.subplots(constrained_layout=True)


ax.semilogx(hs,lda0_ldoge,linewidth=2)
ax.semilogx(hsref,ldref,linewidth=2)




#plt.xlabel(r'$h_o/\sqrt{S}$', fontsize=fontsize)
#plt.ylabel(r'$\frac{(L/D)_{IGE}}{(L/D)_{OGE}}$', fontsize=fontsize)


ax.set_xticks([0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3], labels=[0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3], fontsize=fontsize - 3)
ax.set_yticks([1,2,3,4], labels=[1,2,3,4], fontsize=fontsize - 3)

ax.set_xlim([0.01,0.3])
ax.set_ylim([1,4])

#plt.legend([r'$\alpha=0^\circ$',r'$\alpha=2^\circ$',r'$\alpha=4^\circ$',r'$\alpha=6^\circ$'], fontsize=fontsize-4)
ax.grid(color='lavender')


#plt.savefig('ld_ige_oge.png', dpi=600, transparent=True, bbox_inches="tight")
plt.show()
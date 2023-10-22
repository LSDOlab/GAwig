import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (8,4.5)

fontsize = 16

ld_oge = np.array([0.29443702,0.45589505,0.61224124,0.76186946,0.9034281,1.03585445,])
ld_a0 = np.array([0.31699217,0.34722763,0.4088607,0.51800695,0.67715792,0.86882222,1.05024802,])
ld_a2 = np.array([0.49129421,0.54008191,0.63299309,0.77835611,0.95885763,1.13918404,1.28414822,])
ld_a4 = np.array([0.65762141,0.71876537,0.82766239,0.98063953,1.14629028,1.28982615,1.39205543,])
ld_a6 = np.array([0.81468118,0.88313313,0.99754808,1.14423502,1.28689962,1.39868003,1.47196309,])

hs = np.array([0.49212598,0.24606299,0.1230315,0.06151575,0.03075787,0.01537894,0.00768947,])

hsref = np.array([0.3,0.15,0.08,0.04,0.02,0.01,])
ldref = np.array([1.13,1.25,1.5,1.85,2.375,2.85,])


lda0_ldoge = ld_a0/ld_oge[0]
lda2_ldoge = ld_a2/ld_oge[1]
lda4_ldoge = ld_a4/ld_oge[2]
lda6_ldoge = ld_a6/ld_oge[3]


fig, ax = plt.subplots(constrained_layout=True)

ax.semilogx(hs,lda0_ldoge,linewidth=2,label='_nolegend_')
ax.scatter(hs,lda0_ldoge,linewidth=2,marker='v',zorder=10)

ax.semilogx(hs,lda2_ldoge,linewidth=2,label='_nolegend_')
ax.scatter(hs,lda2_ldoge,linewidth=2,marker='s',zorder=10)

ax.semilogx(hs,lda4_ldoge,linewidth=2,label='_nolegend_')
ax.scatter(hs,lda4_ldoge,linewidth=2,marker='o',zorder=10)

ax.semilogx(hs,lda6_ldoge,linewidth=2,label='_nolegend_')
ax.scatter(hs,lda6_ldoge,linewidth=2,marker='D',zorder=10)

# ax.scatter(hsref,ldref,linewidth=2,zorder=10)



ax.axvline(x=0.1,linewidth=2,linestyle='dashed',color='black')


ax.set_xlabel(r'$\frac{h_{TE}}{\sqrt{S}}$', fontsize=fontsize)
ax.set_ylabel(r'$\frac{\left(\frac{L}{D}\right)_{IGE}}{\left(\frac{L}{D}\right)_{OGE}}$', fontsize=fontsize)


ax.set_xticks([0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3], labels=[0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3], fontsize=fontsize - 3)
ax.set_yticks([1,2,3,4], labels=[1,2,3,4], fontsize=fontsize - 3)

ax.set_xlim([0.01,0.3])
ax.set_ylim([1,4])

ax.legend([r'$\alpha=0^{\circ}$',r'$\alpha=2^{\circ}$',r'$\alpha=4^{\circ}$',r'$\alpha=6^{\circ}$','Min expected '+r'$\frac{h_{TE}}{\sqrt{S}}\approx \frac{8}{\sqrt{6000}}$'], fontsize=fontsize-4)
ax.grid(color='lavender')


def fun(x):
    return x*np.sqrt(6000)/30.5

def inv(x):
    return x*30.5/np.sqrt(6000)

secax = ax.secondary_xaxis('top',functions=(fun, inv))
secax.set_xlabel(r'$\frac{h_{TE}}{b}$', fontsize=fontsize)
secax.set_xticks([0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5], labels=[0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5], fontsize=fontsize - 3)

ax.set_title('L/D IGE/OGE Ratios for the Clark-Y AR4 Wing at Varying AOA')

plt.savefig('ld_alpha_sweep.png', dpi=600, transparent=True, bbox_inches="tight")
plt.show()
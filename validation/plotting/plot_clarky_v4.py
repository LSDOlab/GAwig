import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (6,4)

fontsize = 16

ld= np.array([23.3780019,24.57894506,26.53037988,29.76021825,33.70322594,37.02164747,38.19725874,])

hs = np.array([13.91942483,0.695971241,0.347985621,0.17399281,0.086996405,0.043498203,0.021749101,])

hsref = np.array([0.3,0.15,0.08,0.04,0.02,0.01,])
ldref = np.array([26,29,33.5,39.7,44,47])



fig, ax = plt.subplots(constrained_layout=True)

ax.semilogx(hs,ld,linewidth=2,label='_nolegend_')
ax.scatter(hs,ld,linewidth=2,marker='v',zorder=10)

#plt.semilogx(hs,lda2_ldoge,linewidth=2)
#plt.semilogx(hs,lda4_ldoge,linewidth=2)
#plt.semilogx(hs,lda6_ldoge,linewidth=2)

ax.scatter(hsref,ldref,linewidth=2,zorder=10)



ax.axvline(x=0.1,linewidth=2,linestyle='dashed',color='black')


ax.set_xlabel(r'$\frac{h_{TE}}{\sqrt{S}}$', fontsize=fontsize)
ax.set_ylabel(r'$\left(\frac{L}{D}\right)_{MAX}$', fontsize=fontsize)


ax.set_xticks([0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5], labels=[0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5], fontsize=fontsize - 3)
ax.set_yticks([10,20,30,40,50], labels=[10,20,30,40,50], fontsize=fontsize - 3)

ax.set_xlim([0.04,0.6])
ax.set_ylim([15,50])

ax.legend(['UCSD VAST','Brown et al. (Lockheed)','Min expected '+r'$\frac{h_{TE}}{\sqrt{S}}\approx \frac{8}{\sqrt{6000}}$'], fontsize=fontsize-4)
ax.grid(color='lavender')


def fun(x):
    return x*np.sqrt(1.032256)/0.508

def inv(x):
    return x*0.508/np.sqrt(1.032256)

secax = ax.secondary_xaxis('top',functions=(fun, inv))
secax.set_xlabel(r'$\frac{h_{TE}}{b}$', fontsize=fontsize)
secax.set_xticks([0.1,0.2,0.5,0.7,0.9,1.1], labels=[0.1,0.2,0.5,0.7,0.9,1.1], fontsize=fontsize - 3)

ax.set_title('L/D Max for the Clark-Y AR2 Wing')

plt.savefig('ld_data.png', dpi=600, transparent=True, bbox_inches="tight")
plt.show()
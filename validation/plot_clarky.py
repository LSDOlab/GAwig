import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (8.3,3.2)

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



plt.semilogx(hs,lda0_ldoge,linewidth=2)
#plt.semilogx(hs,lda2_ldoge,linewidth=2)
#plt.semilogx(hs,lda4_ldoge,linewidth=2)
#plt.semilogx(hs,lda6_ldoge,linewidth=2)

plt.semilogx(hsref,ldref,linewidth=2)


plt.xlabel(r'$h_o/\sqrt{S}$', fontsize=fontsize)
plt.ylabel(r'$\frac{(L/D)_{IGE}}{(L/D)_{OGE}}$', fontsize=fontsize)


plt.xticks([0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3], labels=[0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3], fontsize=fontsize - 3)
plt.yticks([1,2,3,4], fontsize=fontsize - 3)

plt.xlim([0.01,0.3])
plt.ylim([1,4])

plt.legend(['UCSD VAST','Brown et al. (Lockheed)'], fontsize=fontsize-4)
plt.grid(color='lavender')


plt.savefig('ige_comp.png', dpi=600, transparent=True, bbox_inches="tight")
plt.show()
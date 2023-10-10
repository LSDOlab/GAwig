import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (8,8)

fontsize = 16

ld = np.array([14.61501107,14.80653919,14.99602674,15.15389591,15.26600943,14.87651298,14.24330059,])

hc = np.array([0.3,0.25,0.2,0.15,0.1,0.05,0.025,])



plt.plot(hc,ld,linewidth=2)

im = plt.imread('ldref.jpg')
implot = plt.imshow(im, aspect='auto', extent=[0, 0.35, -2, 65])


#plt.xlabel(r'$h_o/\sqrt{S}$', fontsize=fontsize)
#plt.ylabel(r'$\frac{(L/D)_{IGE}}{(L/D)_{OGE}}$', fontsize=fontsize)


#plt.xticks([0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3], labels=[0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3], fontsize=fontsize - 3)
#plt.yticks([1,2,3,4], fontsize=fontsize - 3)

plt.xlim([0,0.3])
plt.ylim([10,40])

#plt.legend([r'$\alpha=0^\circ$',r'$\alpha=2^\circ$',r'$\alpha=4^\circ$',r'$\alpha=6^\circ$'], fontsize=fontsize-4)
plt.grid(color='lavender')


plt.savefig('ld_naca.png', dpi=600, transparent=True, bbox_inches="tight")
plt.show()
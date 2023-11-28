import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


# plot the lift distribution across the half span:
def plot_lift_spanwise(nt, n_avg, var, num_span, num_chordwise_vlm, num_props, rpos):

    fig = plt.figure(figsize=(8,3))
    xpos = np.linspace(0,num_span,num_span)

    data = np.zeros((num_span))
    for i in range(nt - n_avg - 1, nt - 1):
        temp = np.zeros(num_span)
        for j in range(num_chordwise_vlm - 1):
            temp[:] += var[i,j*num_span:(j+1)*num_span,0].flatten()
        data[:] += temp

    plt.plot(xpos/max(xpos), data/n_avg, label='_nolegend_')
    plt.scatter(xpos/max(xpos), data/n_avg, label='_nolegend_')
    for i in range(int(num_props/2)): plt.axvline(x=rpos[i], color='black', linestyle='dashed', linewidth=2)
    plt.xlim([0,1])

    plt.xlabel('Spanwise location')
    plt.ylabel('Lift (N/m)')
    plt.legend(['Rotor locations'], frameon=False)
    # plt.savefig('lift_distribution_pos_y.png', transparent=True, bbox_inches="tight", dpi=400)
    plt.show()
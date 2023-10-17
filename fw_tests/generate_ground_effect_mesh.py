import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_ground_effect_mesh(wing_mesh_temp, theta, h, test_plot=False):
    TE_x = wing_mesh_temp[-1,0,0] # extracting TE location
    wing_mesh_temp[:,:,0] -= TE_x # shifting s.t. TE is at origin

    r = R.from_euler('y', -theta, degrees=True).as_matrix() # num_nodes,3,3 ===== rotating about origin
    wing_mesh = np.einsum('ijk,kl->ijl', wing_mesh_temp, r)
    wing_mesh[:,:,2] += h # shifting wing up by h

    wing_image_mesh = wing_mesh.copy()
    wing_image_mesh[:,:,2] *= -1.

    # RESETTING TEMPORARY MESH
    wing_mesh_temp[:,:,0] += TE_x

    # SHIFTING BACK THE TRAILING EDGE POINT
    wing_mesh[:,:,0] += TE_x
    wing_image_mesh[:,:,0] += TE_x

    if test_plot:

        wing_line = wing_mesh[:,0,:]
        wing_image_line = wing_image_mesh[:,0,:]
        symmetry_plane = np.zeros((2,2))
        symmetry_plane[0,0] = wing_line[0,0]
        symmetry_plane[0,-1] = wing_line[-1,0]

        import matplotlib.pyplot as plt
        plt.plot(wing_line[:,0], wing_line[:,2], 'k-*', label='wing')
        plt.plot(wing_image_line[:,0], wing_image_line[:,2], 'b-*', label='image')
        plt.plot(symmetry_plane[0,:], symmetry_plane[1,:], 'r--', label='ground plane')
        # plt.gca().invert_xaxis()
        plt.legend()
        plt.axis('equal')
        plt.grid()
        plt.show()
        # exit()
    return wing_mesh, wing_image_mesh
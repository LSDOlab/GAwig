import numpy as np


def deflect_flap(camber_mesh : np.ndarray, deflection_angle, num_chordwise : int=2):
    """
    Function that deflects a VLM camber mesh to simulate flap deflection.
    The last 'num_chordwise' panels are rotated doward by the deflection angle.

    Parameters
    -----------
    camber_mesh: numpy array
        An array of shape (nx, ny, 3)

    deflection angle: int, float
        The flap deflection angle in degrees

    num_chordwise: int
        The number of chordwise panles that will be deflected
    """
    flap_mesh = camber_mesh[-num_chordwise:, :, :]

    shape = camber_mesh.shape
    flap_mesh = np.zeros(shape=shape)
    flap_mesh[0:-num_chordwise, :, :] =  camber_mesh[0:-num_chordwise, :, :]

    theta = np.deg2rad(-deflection_angle)
    rot_mat = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)],
    ])

    nx_list = np.arange(-num_chordwise, 0, 1).tolist()

    for nx in nx_list:
        for ny in range(shape[1]):
            rot_vec = camber_mesh[nx, ny, :]
            index = (num_chordwise + 1) + nx
            origin_vec = camber_mesh[nx-index, ny, :]

            flap_mesh[nx, ny, :] = rot_mat @ (rot_vec-origin_vec) + origin_vec
    
    return flap_mesh
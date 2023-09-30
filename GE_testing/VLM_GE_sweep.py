import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator

from generate_ground_effect_mesh import generate_ground_effect_mesh

import seaborn as sns
sns.set()

'''
This file demonstrates a sweep across different GE heights and angles of attack

WIG and WOG are compared (CL and CDi)

'''

def GE_sweep(h_b_array, alpha_array, AR=4, span=10., plot_results=True):

    # region inputs
    nx, ny = 3, 11
    num_nodes = 1

    mach = 0.3
    sos = 340.3
    v_inf_scalar = mach*sos
    v_inf = np.ones((num_nodes,1)) * v_inf_scalar
    # endregion

    # region initialization
    num_h_b = len(h_b_array)
    num_alpha = len(alpha_array)
    CL = np.zeros((num_h_b, num_alpha))
    CDi = np.zeros_like(CL)

    CL_image = np.zeros_like(CL)
    CDi_image = np.zeros_like(CL)

    CL_noGE = np.zeros_like(CL)
    CDi_noGE = np.zeros_like(CL)

    # endregion

    # region VLM meshes
    mesh_dict = {
        "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": span,
        "chord": span/AR, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0, # 'offset': np.array([0., 0., h])
    }
    # endregion


    for i, alpha in enumerate(alpha_array):
        print('alpha:', alpha)
        for j, h_b in enumerate(h_b_array):
            print('height:', h_b)

            surface_names, surface_shapes = [], []

            h = span*h_b
            # theta = np.deg2rad(np.ones((num_nodes,1))*alpha)
            theta = alpha

            wing_mesh_temp = generate_mesh(mesh_dict) # temporary wing mesh
            # print(wing_mesh_temp)
            # print(wing_mesh_temp.shape)
            wing_mesh, wing_image_mesh = generate_ground_effect_mesh(wing_mesh_temp, theta, h, test_plot=False)
            # region NO GROUND EFFECT SIMULATION
            model = csdl.Model()

            surface_names.append('wing')
            surface_shapes.append((num_nodes, nx, ny, 3))
            wing = model.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), wing_mesh))

            acstates_model = CreateACSatesModel(v_inf=v_inf, theta=np.zeros((num_nodes,1))*180/np.pi, num_nodes=num_nodes)
            model.add(acstates_model, 'ac_states_model')

            eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
            solver_model = VLMSolverModel(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                num_nodes=num_nodes,
                eval_pts_shapes=eval_pts_shapes,
                AcStates='dummy',
                frame='inertial',
                cl0 = [0.0,0.0],
            )
            model.add(solver_model, 'VLMSolverModel')

            sim = Simulator(model)
            sim.run()

            CL_noGE[j,i] = sim['wing_C_L']
            CDi_noGE[j,i] = sim['wing_C_D_i']
            # endregion

            # region GROUND EFFECT SIMULATION
            model = csdl.Model()

            # surface_names.append('wing')
            # surface_shapes.append((num_nodes, nx, ny, 3))
            wing = model.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), wing_mesh))

            surface_names.append('wing_image')
            surface_shapes.append((num_nodes, nx, ny, 3))
            wing_image = model.create_input('wing_image', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), wing_image_mesh))
    
            # region aircraft states
            acstates_model = CreateACSatesModel(v_inf=v_inf, theta=np.zeros((num_nodes,1))*180/np.pi, num_nodes=num_nodes)
            model.add(acstates_model, 'ac_states_model')
            # endregion

            # VAST SOLVER
            eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
            solver_model = VLMSolverModel(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                num_nodes=num_nodes,
                eval_pts_shapes=eval_pts_shapes,
                AcStates='dummy',
                frame='inertial',
                cl0 = [0.0,0.0],
            )

            model.add(solver_model, 'VLMSolverModel')

            sim = Simulator(model)
            sim.run()

            # del(model)
            # del(sim)

            CL[j,i] = sim['wing_C_L']
            CDi[j,i] = sim['wing_C_D_i']
            CL_image[j,i] = sim['wing_image_C_L']
            CDi_image[j,i] = sim['wing_image_C_D_i']
            # endregion

    if plot_results:
        import matplotlib.pyplot as plt

        color = plt.cm.rainbow(np.linspace(0, 1, num_alpha))
        # LIFT COEFFICIENT
        plt.figure()
        for i, alpha in enumerate(alpha_array):
            plt.plot(h_b_array, CL_noGE[:,i], '-', c=color[i], label=f'No GE, aoa = {alpha} deg')
            plt.plot(h_b_array, CL[:,i], '-*', markersize=8, c=color[i], label=f'GE, aoa = {alpha} deg')
        plt.ylabel('CL')
        plt.xlabel('h/b')
        plt.legend(loc='best', fontsize=8)

        # INDUCED DRAG COEFFICIENT
        plt.figure()
        for i, alpha in enumerate(alpha_array):
            plt.plot(h_b_array, CDi_noGE[:,i], '-', c=color[i], label=f'No GE, aoa = {alpha} deg')
            plt.plot(h_b_array, CDi[:,i], '-*', markersize=8, c=color[i], label=f'GE, aoa = {alpha} deg')
        plt.ylabel('CDi')
        plt.xlabel('h/b')
        plt.legend(loc='best', fontsize=8)
        plt.show()

    return CL, CL_image, CDi, CDi_image, CL_noGE, CDi_noGE


h_b_array = np.array([0.15, 0.25, 0.5, 0.75, 1.])
alpha_array = np.array([2., 5., 10., 15.])
# alpha_array = np.array([2.])

CL, CL_image, CDi, CDi_image, CL_noGE, CDi_noGE = GE_sweep(h_b_array, alpha_array)

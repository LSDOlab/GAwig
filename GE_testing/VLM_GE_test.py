
'''Example 1 : simulation of a rectangular wing'''
import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator

from generate_ground_effect_mesh import generate_ground_effect_mesh

# region inputs
nx, ny = 3, 11
num_nodes = 1

h = 1.

mach = 0.02
sos = 340.3
v_inf_scalar = mach*sos

pitch_scalar = 5. # degrees
# endregion

fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

model = csdl.Model()

# region aircraft states
v_inf = np.ones((num_nodes,1)) * v_inf_scalar
theta = np.deg2rad(np.ones((num_nodes,1))*pitch_scalar)

# acstates_model = CreateACSatesModel(v_inf=v_inf, num_nodes=num_nodes)
acstates_model = CreateACSatesModel(v_inf=v_inf, theta=np.zeros((num_nodes,1))*180/np.pi, num_nodes=num_nodes)
model.add(acstates_model, 'ac_states_model')
# endregion

# region VLM meshes
surface_names, surface_shapes = [], []

mesh_dict = {
    "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
    "chord": 4, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0, # 'offset': np.array([0., 0., h])
}

wing_mesh_temp = generate_mesh(mesh_dict) # temporary wing mesh
theta = pitch_scalar

wing_mesh, wing_image_mesh = generate_ground_effect_mesh(wing_mesh_temp, theta, h, test_plot=False)

surface_names.append('wing')
surface_shapes.append((num_nodes, nx, ny, 3))
wing = model.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), wing_mesh))

surface_names.append('wing_image')
surface_shapes.append((num_nodes, nx, ny, 3))
wing_image = model.create_input('wing_image', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), wing_image_mesh))

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

sim = Simulator(model, analytics=True)
sim.run()

print('==== RESULTS ====')
print('\n')
for surface in surface_names:
    print(f'==== Results for surface: {surface} ====')
    print(f'{surface} Lift (N)', sim[f'{surface}_L'])
    print(f'{surface} Drag (N)', sim[f'{surface}_D'])
    print(f'{surface} CL', sim[f'{surface}_C_L'])
    print(f'{surface} CD_i', sim[f'{surface}_C_D_i'])

    print('\n')

print('==== Results for total values ====')
print('Total Lift (N)', sim['total_lift'])
print('Total Drag (N)', sim['total_drag'])
print('L/D', sim['L_over_D'])
print('Total CL', sim['total_CL'])
print('Total CD', sim['total_CD'])
print('Total Moments', sim['M'])
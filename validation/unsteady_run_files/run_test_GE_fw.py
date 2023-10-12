import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
import array_mapper as am
import lsdo_geo as lg
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
import csdl
from mirror import Mirror
from VAST.utils.generate_mesh import *
from generate_ground_effect_mesh import generate_ground_effect_mesh 

from lsdo_uvlm.uvlm_operation import *

import python_csdl_backend

'''
This file runs a ground effect case using prescribed wake.
The mesh is made using VAST's mesh generation tool.
Everything is done outside of CADDEE (so no OpenVSP geometry).

NOTE: The mirroring is done WITHOUT Nick's M3L model.
    - this will be addressed in the future

File structure:
- UVLM mesh generation
    - generate mesh
    - mirror mesh
    - extend mesh (for time steps)
- Mesh mirroring
- UVLM states
- UVLM solver
- setting connections

'''

# ==================== INITIALIZE PARAMETERS ====================
nx = 15
ny = 5

h = 10
alpha = 15.
AR = 8
span = 12
chord = span/AR

num_nodes = 20
nt = num_nodes + 1

# region mesh
mesh_dict = {
    "num_y": ny,
    "num_x": nx,
    "wing_type": "rect",
    "symmetry": False,
    "span": span,
    "root_chord": chord,
    "span_cos_spacing": False,
    "chord_cos_spacing": False,
}

mesh_temp = generate_mesh(mesh_dict)
mesh, image_mesh = generate_ground_effect_mesh(mesh_temp, alpha, h, test_plot=False)
# exit()
u_val = np.ones(num_nodes).reshape(num_nodes,1)*10
theta_val = np.ones(num_nodes).reshape(num_nodes,1)*np.deg2rad(0.)
uvlm_parameters = [('u',True,u_val),
                    ('v',True,np.zeros((num_nodes, 1))),
                    ('w',True,np.zeros((num_nodes, 1))),
                    ('p',True,np.zeros((num_nodes, 1))),
                    ('q',True,np.zeros((num_nodes, 1))),
                    ('r',True,np.zeros((num_nodes, 1))),
                    ('theta',True,theta_val),
                    ('psi',True,np.zeros((num_nodes, 1))),
                #    ('x',True,np.zeros((num_nodes, 1))),
                #    ('y',True,np.zeros((num_nodes, 1))),
                #    ('z',True,np.zeros((num_nodes, 1))),
                #    ('phiw',True,np.zeros((num_nodes, 1))),
                    ('gamma',True,np.zeros((num_nodes, 1))),
                    ('psiw',True,np.zeros((num_nodes, 1)))]

mesh_val = np.zeros((num_nodes, nx, ny, 3))
image_mesh_val = np.zeros_like(mesh_val)
for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh.copy()
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1]

    image_mesh_val[i, :, :, :] = image_mesh.copy()
    image_mesh_val[i, :, :, 0] = image_mesh.copy()[:, :, 0]
    image_mesh_val[i, :, :, 1] = image_mesh.copy()[:, :, 1]

uvlm_parameters.append(('wing', True, mesh_val))
uvlm_parameters.append(('wing_image', True, image_mesh_val))

surface_names= [
    'wing', 
    'wing_image'
]
surface_shapes = [
    (nx, ny, 3), 
    (nx, ny, 3)
]
h_stepsize = delta_t = 1/16

# endregion

# # design scenario
# design_scenario = cd.DesignScenario(name='wig')
# wig_model = m3l.Model()
# wig_condition = cd.CruiseCondition(name='wig')
# wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
# wig_condition.set_module_input(name='altitude', val=0)
# wig_condition.set_module_input(name='mach_number', val=0.21623, dv_flag=True, lower=0.1, upper=0.3)
# wig_condition.set_module_input(name='range', val=1000)
# wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0), dv_flag=False, lower=np.deg2rad(-10), upper=np.deg2rad(10))
# wig_condition.set_module_input(name='flight_path_angle', val=0)
# wig_condition.set_module_input(name='roll_angle', val=0)
# wig_condition.set_module_input(name='yaw_angle', val=0)
# wig_condition.set_module_input(name='wind_angle', val=0)
# wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))
# ac_states = wig_condition.evaluate_ac_states()
# wig_model.register_output(ac_states)

# region UVLM
initial_conditions = []
for i in range(len(surface_names)):
    surface_name = surface_names[i]
    gamma_w_0_name = surface_name + '_gamma_w_0'
    wake_coords_0_name = surface_name + '_wake_coords_0'
    surface_shape = surface_shapes[i]
    nx = surface_shape[0]
    ny = surface_shape[1]
    initial_conditions.append((gamma_w_0_name, np.zeros((num_nodes, ny - 1))))

    initial_conditions.append((wake_coords_0_name, np.zeros((num_nodes, ny, 3))))

profile_outputs = []
for i in range(len(surface_names)):
    profile_outputs.append((f'{surface_names[i]}_gamma_b', ((surface_shapes[i][0]-1)*(surface_shapes[i][1]-1),)))
    profile_outputs.append((f'{surface_names[i]}_eval_pts_coords', ((surface_shapes[i][0]-1),(surface_shapes[i][1]-1),3)))
    profile_outputs.append((f'{surface_names[i]}_s_panel', ((surface_shapes[i][0]-1),(surface_shapes[i][1]-1))))
    profile_outputs.append((f'{surface_names[i]}_eval_total_vel', ((surface_shapes[i][0]-1)*(surface_shapes[i][1]-1),3)))

    profile_outputs.append((f'{surface_names[i]}_wake_coords', (num_nodes, ny, 3)))
    profile_outputs.append((f'{surface_names[i]}_wake_total_vel', (num_nodes, ny, 3)))
    profile_outputs.append((f'{surface_names[i]}_wake_induced_vel', (num_nodes, ny, 3)))
    profile_outputs.append((f'{surface_names[i]}_wake_kinematic_vel', (num_nodes, ny, 3)))

    profile_outputs.append((f'{surface_names[i]}_gamma_w', (num_nodes, ny-1)))

    profile_outputs.append((f'{surface_names[i]}_bd_vtx_coords', (nx, ny, 3)))

profile_outputs.append(('gamma_b', ((nx-1)*(ny-1)*len(surface_names),)))
profile_outputs.append(('gamma_w', (num_nodes, (ny-1)*len(surface_names),)))


# profile outputs are outputs from the ode integrator that are not states.
# instead they are outputs of a function of the solved states and parameters
# profile_outputs.append(('wing_gamma_w', (num_nodes,4)))

# profile_outputs.append(('wing_gamma_b', ((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),)))
# profile_outputs.append(('wing_eval_pts_coords', ((surface_shapes[0][0]-1),(surface_shapes[0][1]-1),3)))
# profile_outputs.append(('wing_s_panel', ((surface_shapes[0][0]-1),(surface_shapes[0][1]-1))))
# profile_outputs.append(('wing_eval_total_vel', ((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),3)))
profile_outputs.append(('rho',(1,)))
profile_outputs.append(('alpha',(1,)))
profile_outputs.append(('beta',(1,)))
profile_outputs.append(('frame_vel',(3,)))
# profile_outputs.append(('evaluation_pt'))
profile_outputs.append(('bd_vec', ((nx-1)*(ny-1)*len(surface_names),3))) # NOTE: UPDATE THE SHAPE ACCORDING TO nx, ny, nt, etc.

profile_outputs.append(('horseshoe_circulation', ((nx-1)*(ny-1)*len(surface_names),))) # NOTE: UPDATE THE SHAPE ACCORDING TO nx, ny, nt, etc.
# profile_outputs.append(('F', (num_nodes, 3)) )
# profile_outputs.append(('wing_L', (num_nodes,1)))
# # profile_outputs.append(('uvlm_wing2_C_L', (num_nodes,1)))
# profile_outputs.append(('uvlm_wing_D', (num_nodes,1)))
# profile_outputs.append(('uvlm_wing2_C_D_i', (num_nodes,1)))

profile_system = ProfileOpModel
profile_params_dict = {
    'surface_names': surface_names,
    'surface_shapes': surface_shapes,
    'delta_t': delta_t,
    'nt': nt
}


model = m3l.DynamicModel()
uvlm = UVLMCore(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    delta_t=delta_t,
    nt=nt
)
uvlm_residual = uvlm.evaluate()
model.register_output(uvlm_residual)
model.set_dynamic_options(
    initial_conditions=initial_conditions,
    num_times=num_nodes,
    h_stepsize=delta_t,
    parameters=uvlm_parameters,
    integrator='ForwardEuler',
    profile_outputs=profile_outputs,
    profile_system=profile_system,
    profile_parameters=profile_params_dict
)
model_csdl = model.assemble()

eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
ode_surface_shapes = [(num_nodes, ) + item for item in surface_shapes]
eval_pts_shapes = [
    tuple(map(lambda i,j: i-j, item, (0,1,1,0)))
    for item in ode_surface_shapes
]

submodel = ThrustDrag(
        surface_names=surface_names,
        surface_shapes=ode_surface_shapes,
        eval_pts_option='auto',
        eval_pts_shapes=eval_pts_shapes,
        eval_pts_names=eval_pts_names,
        sprs=None,
        coeffs_aoa=None,
        coeffs_cd=None,
    )

# endregion

model_csdl.add(submodel, name='ThrustDrag')

sim = python_csdl_backend.Simulator(model_csdl, analytics=True)

import cProfile
profiler = cProfile.Profile()
profiler.enable()
sim.run()
# After code
profiler.disable()
profiler.dump_stats('output')

if True:
    from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
    axs = Axes(
        xrange=(0, 35),
        yrange=(-10, 10),
        zrange=(-3, 4),
    )
    video = Video("uvlm_m3l_GE_test.gif", duration=10, backend='ffmpeg')
    for i in range(nt - 1):
        vp = Plotter(
            bg='beige',
            bg2='lb',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1)
        # Any rendering loop goes here, e.g.:
        for surface_name in surface_names:
            surface_name = 'prob.' + surface_name
            vps = Points(np.reshape(sim[surface_name][i, :, :, :], (-1, 3)),
                        r=8,
                        c='red')
            vp += vps
            vp += __doc__
            vps = Points(np.reshape(sim[surface_name+'_wake_coords_integrated'][i, 0:i, :, :],
                                    (-1, 3)),
                        r=8,
                        c='blue')
            vp += vps
            vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        vp.show(axs, elevation=-90, azimuth=-0,
                axes=False, interactive=False)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.close_window()
    vp.close_window()
    video.close()  # merge all the recorded frames
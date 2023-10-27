from VAST.core.vast_solver_unsteady import VASTSolverUnsteady
from VAST.core.profile_model import gen_profile_output_list, PPSubmodel
import python_csdl_backend
from VAST.utils.generate_mesh import *
import m3l
import csdl
import caddee.api as cd
from expansion_op import ac_expand
from generate_ground_effect_mesh import generate_ground_effect_mesh

########################################
# define mesh here
########################################
nx = 7
ny = 21
AR = 8
span = 8
chord = span/AR
num_nodes = 10
h = 2

nt = num_nodes+1
alpha = 15

h_stepsize = delta_t = 1/4

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

panel_neg_y = mesh.copy()[:,:int((ny+1)/2),:]
panel_neg_y[:,:,1] -= 1
panel_neg_y[:,:,2] += h
panel_pos_y = mesh.copy()[:,int((ny-1)/2):,:]
panel_pos_y[:,:,1] += 1
panel_pos_y[:,:,2] += h

panel_2_neg_y = mesh.copy()[:,:int((ny+1)/2),:]
panel_2_neg_y[:,:,1] -= 2
panel_2_neg_y[:,:,2] += 2*h
panel_2_pos_y = mesh.copy()[:,int((ny-1)/2):,:]
panel_2_pos_y[:,:,1] += 2
panel_2_pos_y[:,:,2] += 2*h


panel_neg_y_image = image_mesh.copy()[:,:int((ny+1)/2),:]
panel_neg_y_image[:,:,1] -= 1
panel_neg_y_image[:,:,2] -= h
panel_pos_y_image = image_mesh.copy()[:,int((ny-1)/2):,:]
panel_pos_y_image[:,:,1] += 1
panel_pos_y_image[:,:,2] -= h

panel_2_neg_y_image = image_mesh.copy()[:,:int((ny+1)/2),:]
panel_2_neg_y_image[:,:,1] -= 2
panel_2_neg_y_image[:,:,2] -= 2*h
panel_2_pos_y_image = image_mesh.copy()[:,int((ny-1)/2):,:]
panel_2_pos_y_image[:,:,1] += 2
panel_2_pos_y_image[:,:,2] -= 2*h

mesh_val = np.zeros((num_nodes, nx, ny, 3))
neg_panel_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
pos_panel_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
neg_panel_2_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
pos_panel_2_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
image_mesh_val = np.zeros((num_nodes, nx, ny, 3))
image_neg_panel_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
image_pos_panel_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
image_neg_panel_2_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
image_pos_panel_2_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))

for j in range(num_nodes):
    mesh_val[j, :, :, 0] = mesh.copy()[:, :, 0]
    mesh_val[j, :, :, 1] = mesh.copy()[:, :, 1]
    mesh_val[j, :, :, 2] = mesh.copy()[:, :, 2]

    neg_panel_val[j, :, :, 0] = panel_neg_y.copy()[:, :, 0]
    neg_panel_val[j, :, :, 1] = panel_neg_y.copy()[:, :, 1]
    neg_panel_val[j, :, :, 2] = panel_neg_y.copy()[:, :, 2]

    pos_panel_val[j, :, :, 0] = panel_pos_y.copy()[:, :, 0]
    pos_panel_val[j, :, :, 1] = panel_pos_y.copy()[:, :, 1]
    pos_panel_val[j, :, :, 2] = panel_pos_y.copy()[:, :, 2]

    neg_panel_2_val[j, :, :, 0] = panel_2_neg_y.copy()[:, :, 0]
    neg_panel_2_val[j, :, :, 1] = panel_2_neg_y.copy()[:, :, 1]
    neg_panel_2_val[j, :, :, 2] = panel_2_neg_y.copy()[:, :, 2]

    pos_panel_2_val[j, :, :, 0] = panel_2_pos_y.copy()[:, :, 0]
    pos_panel_2_val[j, :, :, 1] = panel_2_pos_y.copy()[:, :, 1]
    pos_panel_2_val[j, :, :, 2] = panel_2_pos_y.copy()[:, :, 2]

    image_mesh_val[j, :, :, 0] = image_mesh.copy()[:, :, 0]
    image_mesh_val[j, :, :, 1] = image_mesh.copy()[:, :, 1]
    image_mesh_val[j, :, :, 2] = image_mesh.copy()[:, :, 2]

    image_neg_panel_val[j, :, :, 0] = panel_neg_y_image.copy()[:, :, 0]
    image_neg_panel_val[j, :, :, 1] = panel_neg_y_image.copy()[:, :, 1]
    image_neg_panel_val[j, :, :, 2] = panel_neg_y_image.copy()[:, :, 2]

    image_pos_panel_val[j, :, :, 0] = panel_pos_y_image.copy()[:, :, 0]
    image_pos_panel_val[j, :, :, 1] = panel_pos_y_image.copy()[:, :, 1]
    image_pos_panel_val[j, :, :, 2] = panel_pos_y_image.copy()[:, :, 2]

    image_neg_panel_2_val[j, :, :, 0] = panel_2_neg_y_image.copy()[:, :, 0]
    image_neg_panel_2_val[j, :, :, 1] = panel_2_neg_y_image.copy()[:, :, 1]
    image_neg_panel_2_val[j, :, :, 2] = panel_2_neg_y_image.copy()[:, :, 2]

    image_pos_panel_2_val[j, :, :, 0] = panel_2_pos_y_image.copy()[:, :, 0]
    image_pos_panel_2_val[j, :, :, 1] = panel_2_pos_y_image.copy()[:, :, 1]
    image_pos_panel_2_val[j, :, :, 2] = panel_2_pos_y_image.copy()[:, :, 2]

# u_vel = np.ones(num_nodes).reshape(num_nodes,1)*1
# w_vel = np.zeros((num_nodes, 1))
# theta_val = np.zeros((num_nodes, 1))*np.deg2rad(alpha)

# design scenario
design_scenario = cd.DesignScenario(name='wig')
wig_model = m3l.Model()
wig_condition = cd.CruiseCondition(name='wig')
wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
wig_condition.set_module_input(name='altitude', val=0)
wig_condition.set_module_input(name='mach_number', val=0.05, dv_flag=True, lower=0.1, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0.), dv_flag=False, lower=np.deg2rad(-10), upper=np.deg2rad(10))
wig_condition.set_module_input(name='flight_path_angle', val=0)
wig_condition.set_module_input(name='roll_angle', val=0)
wig_condition.set_module_input(name='yaw_angle', val=0)
wig_condition.set_module_input(name='wind_angle', val=0)
wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))
ac_states = wig_condition.evaluate_ac_states()

ac_expander = ac_expand(num_nodes=num_nodes)
ac_states_expanded = ac_expander.evaluate(ac_states)

uvlm_parameters = [('u',True,ac_states_expanded['u']),
                    ('v',True,ac_states_expanded['v']),
                    ('w',True,ac_states_expanded['w']),
                    ('p',True,ac_states_expanded['p']),
                    ('q',True,ac_states_expanded['q']),
                    ('r',True,ac_states_expanded['r']),
                    ('theta',True,ac_states_expanded['theta']),
                    ('psi',True,ac_states_expanded['psi']),
                    ('gamma',True,ac_states_expanded['gamma']),
                    ('psiw',True,np.zeros((num_nodes, 1)))]

uvlm_parameters.append(('wing', True, mesh_val))
uvlm_parameters.append(('neg_panel', True, neg_panel_val))
uvlm_parameters.append(('pos_panel', True, pos_panel_val))
uvlm_parameters.append(('wing_image', True, image_mesh_val))
uvlm_parameters.append(('neg_panel_image', True, image_neg_panel_val))
uvlm_parameters.append(('pos_panel_image', True, image_pos_panel_val))
uvlm_parameters.append(('neg_panel_2', True, neg_panel_2_val))
uvlm_parameters.append(('pos_panel_2', True, pos_panel_2_val))
uvlm_parameters.append(('neg_panel_2_image', True, image_neg_panel_2_val))
uvlm_parameters.append(('pos_panel_2_image', True, image_pos_panel_2_val))
surface_names = [
    'wing',
    'neg_panel',
    'pos_panel',
    'wing_image',
    'neg_panel_image',
    'pos_panel_image',
    'neg_panel_2',
    'pos_panel_2',
    'neg_panel_2_image',
    'pos_panel_2_image',
]
surface_shapes = [
    (nx,ny,3),
    (nx, int((ny+1)/2), 3),
    (nx, int((ny+1)/2), 3),
    (nx,ny,3),
    (nx, int((ny+1)/2), 3),
    (nx, int((ny+1)/2), 3),
    (nx, int((ny+1)/2), 3),
    (nx, int((ny+1)/2), 3),
    (nx, int((ny+1)/2), 3),
    (nx, int((ny+1)/2), 3),
]

system_size = 0
for i in range(len(surface_shapes)):
    system_size += surface_shapes[i][0] * surface_shapes[i][1]

initial_conditions = []
for i in range(len(surface_names)):
    surface_name = surface_names[i]
    gamma_w_0_name = surface_name + '_gamma_w_0'
    wake_coords_0_name = surface_name + '_wake_coords_0'
    surface_shape = surface_shapes[i]
    nx = surface_shape[0]
    ny = surface_shape[1]
    initial_conditions.append((gamma_w_0_name, np.zeros((num_nodes-1, ny - 1))))

    initial_conditions.append((wake_coords_0_name, np.zeros((num_nodes-1, ny, 3))))

# profile_outputs = []

# profile_outputs.append(('wing_L', (num_nodes,1)))
# profile_outputs.append(('wing_D', (num_nodes,1)))

# profile_params_dict = {
#         'surface_names': ['wing'],
#         'surface_shapes': surface_shapes,
#         'delta_t': delta_t,
#         'nt': nt
#     }

# sub_eval_list = [0, 0, 0, 1, 1, 1, 2, 2, 2]
# sub_induced_list = [0, 1, 2, 0, 1, 2, 0, 1, 2]
sub_eval_list = []
for i in range(len(surface_names)):
    sub_eval_list.extend([i]*10)
    # sub_eval_list.extend([i]*4)
sub_induced_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10
# sub_induced_list = [0, 1, 2, 3] * 4
sub = True

sym_struct_list = [[0,3], [1,2,4,5], [6,7,8,9]]
# sym_struct_list = [[0,3], [1,2,4,5]]
# sym_struct_list = [[0,1, 2, 3], [4,5,6,7]]
# sym_struct_list = [[0,1, 2, 3]]



'''
INTERACTIONS:
- wing on wing (0,0)
- neg_panel on wing ()
- pos_panel on wing
'''
profile_outputs = gen_profile_output_list(surface_names, surface_shapes)
ode_surface_shapes = [(num_nodes, ) + item for item in surface_shapes]
post_processor = PPSubmodel(
    surface_names = surface_names,
    ode_surface_shapes = ode_surface_shapes,
    delta_t = h_stepsize,
    nt = num_nodes + 1,
    # sub=sub,
    # sub_eval_list=sub_eval_list,
    # sub_induced_list=sub_induced_list,
    # symmetry=True,
    # sym_struct_list=sym_struct_list
)
pp_vars = []
# for name in surface_names:
#     pp_vars.append((name+'_L', (nt, 1)))
pp_vars.append('panel_forces_x')
pp_vars.append('panel_forces_y')
pp_vars.append('panel_forces_z')

model = m3l.DynamicModel()
uvlm = VASTSolverUnsteady(
    num_nodes=num_nodes, 
    surface_names=surface_names, 
    surface_shapes=surface_shapes, 
    delta_t=delta_t, 
    nt=num_nodes+1,
    # free_wake=True,
    # frame='inertial',
    sub=sub,
    sub_eval_list=sub_eval_list,
    sub_induced_list=sub_induced_list,
    symmetry=True,
    sym_struct_list=sym_struct_list
)
uvlm_residual = uvlm.evaluate()
model.register_output(uvlm_residual)
model.set_dynamic_options(initial_conditions=initial_conditions,
                            num_times=num_nodes,
                            h_stepsize=delta_t,
                            parameters=uvlm_parameters,
                            int_naming=('op_',''),
                            integrator='ForwardEuler',
                            approach='time-marching checkpointing',
                            copycat_profile=True,
                            profile_outputs=profile_outputs,
                            post_processor=post_processor,
                            pp_vars=pp_vars)
uvlm_op = model.assemble(return_operation=True)
outputs = uvlm_op.evaluate()[0:len(pp_vars)]
# wing_C_L, int1, _, _, _, _, _ = uvlm_op.evaluate()
# int1, int2, _, _, _, _, _, _ = uvlm_op.evaluate()
# data = uvlm_op.evaluate()
# wing_C_L = data[0]


overmodel = m3l.Model()
for var in outputs:
    overmodel.register_output(var)
# overmodel.register_output(panel_forces)
# overmodel.register_output(wing_C_L)
model_csdl = overmodel.assemble()


import time 

start = time.time()
sim = python_csdl_backend.Simulator(model_csdl, analytics=True)
setup_time = time.time()
print('simulator setup time:', setup_time-start)

sim_start_1 = time.time()
sim.run()
run_time_1 = time.time()
print('simulator 1 run time:', run_time_1 - sim_start_1)
exit()
sim_start_2 = time.time()
sim.run()
run_time_2 = time.time()

print('simulator 2 run time:', run_time_2 - sim_start_2)
1

wing_CL = sim['operation.post_processor.ThrustDrag.wing_C_L']
wing_CDi = sim['operation.post_processor.ThrustDrag.wing_C_D_i']

print('===============')
print('wing CL: ')
print(wing_CL)
print('===============')
print('wing CDi: ')
print(wing_CDi)

# print(sim['operation.post_processor.LiftDrag.panel_forces'])

if False:
    from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
    axs = Axes(
        xrange=(0, 35),
        yrange=(-10, 10),
        zrange=(-3, 4),
    )
    video = Video("pw_single_wing.gif", duration=10, backend='ffmpeg')
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
            # surface_name = 'operation.' + surface_name
            vps = Points(np.reshape(sim['operation.' + surface_name][i, :, :, :], (-1, 3)),
                        r=8,
                        c='red')
            vp += vps
            vp += __doc__
            vps = Points(np.reshape(sim['operation.prob.op_'+surface_name+'_wake_coords'][i, 0:i, :, :],
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


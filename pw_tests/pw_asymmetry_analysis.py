from VAST.core.vast_solver_unsteady import VASTSolverUnsteady
from VAST.core.profile_model import gen_profile_output_list, PPSubmodel
import python_csdl_backend
from VAST.utils.generate_mesh import *
import m3l
import csdl
import caddee.api as cd
from expansion_op import ac_expand
from generate_ground_effect_mesh import generate_ground_effect_mesh

'''
function inputs:
- distance between wings
'''

single_wing_lift = np.array([30.6721246 , 35.53366853, 38.20418945, 39.72171451, 40.57742575,
       40.96795797, 40.96795797, 40.57742575, 39.72171451, 38.20418945,
       35.53366853, 30.6721246 ]) # SPANWISE SECTIONAL LIFT AT THE FURTHEST FORWARD CHORD PANEL

single_wing_position = np.array([-1.83333333, -1.5       , -1.16666667, -0.83333333, -0.5       ,
       -0.16666667,  0.16666667,  0.5       ,  0.83333333,  1.16666667,
        1.5       ,  1.83333333])



########################################
# define mesh here
########################################
nx = 25
ny = 25
AR = 8
span = 8
chord = span/AR
num_nodes = 3
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

y_gap_list = [2, 5, 10, 20, 40] # distance between wings; need to divide by 2

lift_array = np.zeros(((len(y_gap_list), 2, int((ny-1)/2)))) # gaps, 2 (for 2 surfaces), num spanwise panels
gamma_b_array = np.zeros_like(lift_array)
circulations_array = np.zeros_like(lift_array)
y_positions = np.zeros_like(lift_array)

chord_station = round(nx/2)
# chord_station = 0
if chord_station > int(nx-1):
    raise KeyError(f'chord station cannot exceed {int(nx-1)}')

for y_ind in range(len(y_gap_list)):
    y_gap = y_gap_list[y_ind]/2
    

    panel_neg_y = mesh.copy()[:,:int((ny+1)/2),:]
    panel_neg_y[:,:,1] -= y_gap
    panel_neg_y[:,:,2] += h
    panel_pos_y = mesh.copy()[:,int((ny-1)/2):,:]
    panel_pos_y[:,:,1] += y_gap
    panel_pos_y[:,:,2] += h

    panel_neg_y_image = image_mesh.copy()[:,:int((ny+1)/2),:]
    panel_neg_y_image[:,:,1] -= y_gap
    panel_neg_y_image[:,:,2] -= h
    panel_pos_y_image = image_mesh.copy()[:,int((ny-1)/2):,:]
    panel_pos_y_image[:,:,1] += y_gap
    panel_pos_y_image[:,:,2] -= h

    neg_panel_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
    pos_panel_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
    image_neg_panel_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))
    image_pos_panel_val = np.zeros((num_nodes, nx, int((ny+1)/2), 3))

    for j in range(num_nodes):
        neg_panel_val[j, :, :, 0] = panel_neg_y.copy()[:, :, 0]
        neg_panel_val[j, :, :, 1] = panel_neg_y.copy()[:, :, 1]
        neg_panel_val[j, :, :, 2] = panel_neg_y.copy()[:, :, 2]

        pos_panel_val[j, :, :, 0] = panel_pos_y.copy()[:, :, 0]
        pos_panel_val[j, :, :, 1] = panel_pos_y.copy()[:, :, 1]
        pos_panel_val[j, :, :, 2] = panel_pos_y.copy()[:, :, 2]

        image_neg_panel_val[j, :, :, 0] = panel_neg_y_image.copy()[:, :, 0]
        image_neg_panel_val[j, :, :, 1] = panel_neg_y_image.copy()[:, :, 1]
        image_neg_panel_val[j, :, :, 2] = panel_neg_y_image.copy()[:, :, 2]

        image_pos_panel_val[j, :, :, 0] = panel_pos_y_image.copy()[:, :, 0]
        image_pos_panel_val[j, :, :, 1] = panel_pos_y_image.copy()[:, :, 1]
        image_pos_panel_val[j, :, :, 2] = panel_pos_y_image.copy()[:, :, 2]

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

    # uvlm_parameters.append(('wing', True, mesh_val))
    uvlm_parameters.append(('neg_panel', True, neg_panel_val))
    uvlm_parameters.append(('pos_panel', True, pos_panel_val))
    # uvlm_parameters.append(('wing_mirror', True, image_mesh_val))
    # uvlm_parameters.append(('neg_panel_mirror', True, image_neg_panel_val))
    # uvlm_parameters.append(('pos_panel_mirror', True, image_pos_panel_val))
    surface_names = [
        # 'wing',
        'neg_panel',
        'pos_panel',
        # 'wing_mirror',
        # 'neg_panel_mirror',
        # 'pos_panel_mirror',
    ]
    surface_shapes = [
        # (nx,ny,3),
        (nx, int((ny+1)/2), 3),
        (nx, int((ny+1)/2), 3),
        # (nx,ny,3),
        # (nx, int((ny+1)/2), 3),
        # (nx, int((ny+1)/2), 3),
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
        nx_temp = surface_shape[0]
        ny_temp = surface_shape[1]
        initial_conditions.append((gamma_w_0_name, np.zeros((num_nodes-1, ny_temp - 1))))

        initial_conditions.append((wake_coords_0_name, np.zeros((num_nodes-1, ny_temp, 3))))



    # sub_eval_list = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # sub_induced_list = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    sub_eval_list = []
    # for i in range(len(surface_names)):
        # sub_eval_list.extend([i]*10)
        # sub_eval_list.extend([i]*4)
    # sub_induced_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10
    sub_eval_list = [0, 2, 1, 3, 0, 2, 1, 3]
    # sub_induced_list = [0, 1, 2, 3] * 4
    sub_induced_list = [0, 0, 1, 1, 2, 2, 3, 3]
    sub = True


    # sym_struct_list = [[0,3], [1,2,4,5]]
    sym_struct_list = [[0, 1, 2, 3]]

    profile_outputs = gen_profile_output_list(surface_names, surface_shapes)
    ode_surface_shapes = [(num_nodes, ) + item for item in surface_shapes]
    post_processor = PPSubmodel(
        surface_names = surface_names,
        ode_surface_shapes = ode_surface_shapes,
        delta_t = h_stepsize,
        nt = num_nodes + 1,
    )
    pp_vars = []
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
        # sub=sub,
        # sub_eval_list=sub_eval_list,
        # sub_induced_list=sub_induced_list,
        # symmetry=True,
        # sym_struct_list=sym_struct_list
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


    overmodel = m3l.Model()
    for var in outputs:
        overmodel.register_output(var)
    model_csdl = overmodel.assemble()

    sim = python_csdl_backend.Simulator(model_csdl, analytics=True)
    sim.run()
    mesh_vals = [neg_panel_val, pos_panel_val]
    for i, surf_name in enumerate(surface_names):
        lift_array[y_ind, i, :] = sim[f'operation.post_processor.ThrustDrag.{surf_name}_L_panel'][-1].reshape(int(nx-1), int((ny-1)/2))[chord_station,:]
        gamma_b_array[y_ind, i, :] = sim[f'operation.post_processor.ThrustDrag.{surf_name}_gamma_b'][-1].reshape(int(nx-1), int((ny-1)/2))[chord_station,:]
        circulations_array[y_ind, i, :] = sim[f'operation.post_processor.ThrustDrag.{surf_name}_circulations'][-1].reshape(int(nx-1), int((ny-1)/2))[chord_station,:]
        y_positions[y_ind, i, :] = (mesh_vals[i][0][1,:-1,1] + mesh_vals[i][0][1,1:,1])/2

        print(sim[f'operation.post_processor.ThrustDrag.{surf_name}_C_L'])



# reshape panel_forces[-1] to (nx-1) * ((ny-1)/2) 
names = ['neg_panel', 'pos_panel']
mesh_vals = [neg_panel_val, pos_panel_val]
mode = 'lift'
# mode = 'horseshoe circulations'
# mode = 'gamma_b'

import matplotlib.pyplot as plt
color = plt.cm.rainbow(np.linspace(0, 1, len(y_gap_list)))
if mode == 'lift':
    data = lift_array
elif mode == 'horseshoe circulations':
    data = circulations_array
elif mode == 'gamma_b':
    data = gamma_b_array

delta_lift = []
plt.figure()

# plt.plot(single_wing_position, single_wing_lift, 'k', linewidth=3, label='Single Wing')
for y_ind in range(len(y_gap_list)):
    plt.plot(y_positions[y_ind, 0, :], data[y_ind, 0, :], c=color[y_ind], label=f'dy = {y_gap_list[y_ind]}')
    plt.plot(y_positions[y_ind, 1, :], data[y_ind, 1, :], c=color[y_ind])
    delta_lift.append(np.max(data[y_ind, 0, :]) - np.max(data[y_ind, 1, :]))
print(delta_lift)
plt.legend(fontsize=15, loc='best')
plt.xlabel('spanwise location', fontsize=15)
if mode == 'lift':
    plt.ylabel('sectional lift', fontsize=15)
elif mode == 'horseshoe circulations':
    plt.ylabel('horeshoe circulation', fontsize=15)
elif mode == 'gamma_b':
    plt.ylabel('bound circulations', fontsize=15)
plt.grid()
plt.show()














# =======================
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
                axes=False, interactive=True)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.close_window()
    vp.close_window()
    video.close()  # merge all the recorded frames


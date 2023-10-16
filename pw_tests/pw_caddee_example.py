from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, ProfileOpModel, ProfileOpModel2, PostProcessor
import python_csdl_backend
from VAST.utils.generate_mesh import *
import m3l
import csdl
import caddee.api as cd
from expansion_op import ac_expand

########################################
# define mesh here
########################################
nx = 2
ny = 5 
chord = 1
span = 12
num_nodes = 10

num_surfaces = 4
surface_offset = [0,0,2]

nt = num_nodes+1
alpha = np.deg2rad(5)
system_size = 0
for i in range(num_surfaces):
    system_size += (nx - 1) * (ny - 1)

# define the direction of the flappsing motion (hardcoding for now)

# u_val = np.concatenate((np.array([0.01, 0.5,1.]),np.ones(num_nodes-3))).reshape(num_nodes,1)
# u_val = np.ones(num_nodes).reshape(num_nodes,1)
u_vel = np.ones(num_nodes).reshape(num_nodes,1)*10
# u_vel = 10
# w_vel = np.ones((num_nodes,1)) *np.sin(alpha)
w_vel = np.zeros((num_nodes, 1))
# theta_val = np.linspace(0,alpha,num=num_nodes)
theta_val = np.ones((num_nodes, 1))*alpha



# design scenario
design_scenario = cd.DesignScenario(name='wig')
wig_model = m3l.Model()
wig_condition = cd.CruiseCondition(name='wig')
wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
wig_condition.set_module_input(name='altitude', val=0)
wig_condition.set_module_input(name='mach_number', val=0.21623, dv_flag=True, lower=0.1, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(5), dv_flag=False, lower=np.deg2rad(-10), upper=np.deg2rad(10))
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



surface_names = []
surface_shapes = []
mesh = generate_mesh(mesh_dict)
for i in range(num_surfaces):
    mesh_val = np.zeros((num_nodes, nx, ny, 3))
    for j in range(num_nodes):
        mesh_val[j, :, :, 0] = mesh.copy()[:, :, 0] + i * surface_offset[0]
        mesh_val[j, :, :, 1] = mesh.copy()[:, :, 1] + i * surface_offset[1]
        mesh_val[j, :, :, 2] = mesh.copy()[:, :, 2] + i * surface_offset[2]
    uvlm_parameters.append(('wing' + str(i), True, mesh_val))
    uvlm_parameters.append(('wing' + str(i) + '_coll_vel', True, np.zeros((num_nodes,nx-1,ny-1,3))))
    surface_names.append('wing' + str(i))
    surface_shapes.append((nx,ny,3))

h_stepsize = delta_t = 1/128

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

submodel = PostProcessor(
    num_nodes = num_nodes-1,
    surface_names = surface_names,
    surface_shapes = surface_shapes,
    delta_t = h_stepsize,
    nt = num_nodes + 1
)
pp_vars = [('panel_forces', (num_nodes-1, system_size, 3)), ('eval_pts_all', (num_nodes-1, system_size, 3))]

model = m3l.DynamicModel()
uvlm = VASTSolverUnsteady(num_nodes=num_nodes, surface_names=surface_names, surface_shapes=surface_shapes, delta_t=delta_t, nt=num_nodes+1)
uvlm_residual = uvlm.evaluate()
model.register_output(uvlm_residual)
model.set_dynamic_options(initial_conditions=initial_conditions,
                            num_times=num_nodes,
                            h_stepsize=delta_t,
                            parameters=uvlm_parameters,
                            int_naming=('op_',''),
                            integrator='ForwardEuler',
                            approach='time-marching checkpointing',
                            profile_outputs=None,
                            profile_system=None,
                            profile_parameters=None,
                            post_processor=submodel,
                            pp_vars=pp_vars)
uvlm_op = model.assemble(return_operation=True)

panel_forces, eval_pts, int1, _, _, _, _, _, _, _ = uvlm_op.evaluate()
# int1, int2, _, _, _, _, _, _ = uvlm_op.evaluate()


overmodel = m3l.Model()
overmodel.register_output(panel_forces)
model_csdl = overmodel.assemble()



sim = python_csdl_backend.Simulator(model_csdl, analytics=True)

sim.run()

# print(sim['operation.post_processor.LiftDrag.panel_forces'])
print(sim['operation.post_processor.ThrustDrag.wing0_C_L'])


if False:
    from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
    axs = Axes(
        xrange=(0, 35),
        yrange=(-10, 10),
        zrange=(-3, 4),
    )
    video = Video("uvlm_m3l_test.gif", duration=10, backend='ffmpeg')
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
            surface_name = surface_name
            vps = Points(np.reshape(sim['operation.' + surface_name][i, :, :, :], (-1, 3)),
                        r=8,
                        c='red')
            vp += vps
            vp += __doc__
            vps = Points(np.reshape(sim['operation.' + 'op_' + surface_name+'_wake_coords'][i, 0:i, :, :],
                                    (-1, 3)),
                        r=8,
                        c='blue')
            vp += vps
            vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        vp.show(axs, elevation=-60, azimuth=-0,
                axes=False, interactive=False)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.close_window()
    vp.close_window()
    video.close()  # merge all the recorded frames


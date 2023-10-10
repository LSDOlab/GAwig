from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, ProfileOpModel
import python_csdl_backend
from VAST.utils.generate_mesh import *
import m3l
import csdl

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

# define the direction of the flappsing motion (hardcoding for now)

# u_val = np.concatenate((np.array([0.01, 0.5,1.]),np.ones(num_nodes-3))).reshape(num_nodes,1)
# u_val = np.ones(num_nodes).reshape(num_nodes,1)
u_vel = np.ones(num_nodes).reshape(num_nodes,1)*10
# w_vel = np.ones((num_nodes,1)) *np.sin(alpha)
w_vel = np.zeros((num_nodes, 1))
# theta_val = np.linspace(0,alpha,num=num_nodes)
theta_val = np.ones((num_nodes, 1))*alpha


uvlm_parameters = [('u',True,u_vel),
                    ('v',True,np.zeros((num_nodes, 1))),
                    ('w',True,w_vel),
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
    surface_names.append('wing' + str(i))
    surface_shapes.append((nx,ny,3))

h_stepsize = delta_t = 1/16

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

# profile_outputs = []

# profile_outputs.append(('wing_L', (num_nodes,1)))
# profile_outputs.append(('wing_D', (num_nodes,1)))

# profile_params_dict = {
#         'surface_names': ['wing'],
#         'surface_shapes': surface_shapes,
#         'delta_t': delta_t,
#         'nt': nt
#     }

model = m3l.DynamicModel()
uvlm = VASTSolverUnsteady(num_nodes=num_nodes, surface_names=surface_names, surface_shapes=surface_shapes, delta_t=delta_t, nt=num_nodes+1)
uvlm_residual = uvlm.evaluate()
model.register_output(uvlm_residual)
model.set_dynamic_options(initial_conditions=initial_conditions,
                            num_times=num_nodes,
                            h_stepsize=delta_t,
                            parameters=uvlm_parameters,
                            integrator='ForwardEuler',
                            approach='time-marching',
                            profile_outputs=None,
                            profile_system=None,
                            profile_parameters=None)
model_csdl = model.assemble()

# submodel = ProfileOpModel(
#     num_nodes = num_nodes,
#     surface_names = surface_names,
#     surface_shapes = surface_shapes,
#     delta_t = h_stepsize,
#     nt = num_nodes + 1
# )

# model_csdl.add(submodel, name='post_processing')



wing0_gamma = model_csdl.declare_variable('wing0_gamma_w_integrated',shape = (num_nodes,num_nodes,4))

fwing0_gamma = csdl.pnorm(wing0_gamma)

model_csdl.register_output('fwing0_gamma', fwing0_gamma)


# Before code
from mpi4py import MPI
comm = MPI.COMM_WORLD

sim = python_csdl_backend.Simulator(model_csdl, analytics=False)

import cProfile
profiler = cProfile.Profile()
profiler.enable()
sim.run()
# After code
profiler.disable()
profiler.dump_stats('output')

comm.Barrier()
print(sim['fwing0_gamma']) 
# 4 core: 19.73037898877024
# 1 core: 19.73037898877024
# 1 core: 19.73037898877024

# sim.compute_totals(of = 'fwing0_gamma', wrt = 'wing0')

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
        vp.show(axs, elevation=-60, azimuth=-0,
                axes=False, interactive=False)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.close_window()
    vp.close_window()
    video.close()  # merge all the recorded frames


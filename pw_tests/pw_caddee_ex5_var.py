from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, ProfileOpModel, ProfileOpModel2, PostProcessor
import python_csdl_backend
from VAST.utils.generate_mesh import *
import m3l
import csdl
import caddee.api as cd
from expansion_op import ac_expand
import time

def run_fixed(span,num_nodes,frame='wing_fixed'):
    ########################################
    # 1. define geometry
    ########################################
    nx = 5; ny = 13
    chord = 1; 
    nt = num_nodes

    mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False,
                    "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
    mesh = generate_mesh(mesh_dict)

    ########################################
    # 2. define kinematics
    ########################################

    alpha = np.deg2rad(5) 
    t_vec = np.linspace(0, 9, num_nodes) 

    u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1)) 
    w_vel = np.ones((num_nodes,1)) *np.sin(alpha)

    states_dict = {
        'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
        'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
        'theta': alpha* np.ones((num_nodes,1)), 'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
    }

    uvlm_parameters = [('u',True,states_dict['u']),
                    ('v',True,states_dict['v']),
                    ('w',True,states_dict['w']),
                    ('p',True,states_dict['p']),
                    ('q',True,states_dict['q']),
                    ('r',True,states_dict['r']),
                    ('theta',True,states_dict['theta']),
                    ('psi',True,states_dict['psi']),
                    ('gamma',True,states_dict['gamma']),
                    ('psiw',True,states_dict['psiw'])]


    surface_properties_dict = {'surface_names':['wing'],
                                'surface_shapes':[(nx, ny, 3)],
                            'frame':frame,}
    
    surface_names = surface_properties_dict['surface_names']
    surface_shapes = surface_properties_dict['surface_shapes']

    mesh_val = np.zeros((num_nodes, nx, ny, 3))

    if frame == 'wing_fixed':
        z_offset = -w_vel.flatten()*t_vec*0
        vz = -np.zeros((num_nodes,nx-1,ny-1,3))*np.tan(np.deg2rad(5)).copy()
    elif frame == 'inertia':
        z_offset = (-np.ones((num_nodes,1)) *np.tan(alpha)).flatten()*t_vec
        vz = -np.ones((num_nodes,nx-1,ny-1,3))*np.tan(np.deg2rad(5)).copy()
        vz[:,:,:,0] = 0
        vz[:,:,:,1] = 0
    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
        mesh_val[i, :, :, 2] += z_offset[i]

    h_stepsize = delta_t = t_vec[1] 

    uvlm_parameters.append(('wing_coll_vel', True, vz))
    uvlm_parameters.append(('wing', True, mesh_val))

    initial_conditions = [
        ('wing_gamma_w_0', np.zeros((num_nodes-1, ny-1))),
        ('wing_wake_coords_0', np.zeros((num_nodes-1, ny, 3)))
    ]

    submodel = PostProcessor(
        num_nodes = num_nodes,
        surface_names = surface_names,
        surface_shapes = surface_shapes,
        delta_t = h_stepsize,
        nt = num_nodes + 1,
        symmetry=True
    )
    # pp_vars = [('panel_forces', (num_nodes, system_size, 3)), ('eval_pts_all', (num_nodes, system_size, 3))]
    pp_vars = [('wing_C_L', (num_nodes-1, 1))]

    model = m3l.DynamicModel()
    uvlm = VASTSolverUnsteady(num_nodes=num_nodes, surface_names=surface_names, surface_shapes=surface_shapes, delta_t=delta_t, nt=num_nodes+1, symmetry=True)
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
    cl, int1, int2 = uvlm_op.evaluate()

    overmodel = m3l.Model()
    overmodel.register_output(cl)
    model_csdl = overmodel.assemble() 


    sim = python_csdl_backend.Simulator(model_csdl)
        
    t_start = time.time()
    sim.run()
    print('simulation time is', time.time() - t_start)
    wing_C_L = sim['operation.post_processor.ThrustDrag.wing_C_L']
    del sim
    return wing_C_L, t_vec

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

be = 'python_csdl_backend'
make_video = 0
plot_cl = 1
span = [4, 8, 12, 20]

num_nodes = [20] * len(span)

wing_C_L_list = []
t_vec_list = []
for (i,j) in zip(span,num_nodes):
    wing_C_L, t_vec = run_fixed(i,j)
    plt.plot(t_vec, wing_C_L,'.-')
    # wing_C_L_, t_vec_ = run_fixed(i,num_nodes=j,frame='inertia')
    # plt.plot(t_vec_, wing_C_L_,'.-')
    plt.ylim([0,0.6])
    plt.xlim([0,t_vec.max()+1])
    # np.savetxt('wing_C_L_'+str(i)+'_'+str(j)+'.txt',wing_C_L)
    # np.savetxt('t_vec_'+str(i)+'_'+str(j)+'.txt',t_vec)
    wing_C_L_list.append(wing_C_L)
    # t_vec_list.append(t_vec)

print(wing_C_L_list)

plt.legend(['AR = '+str(i) for i in span])
plt.xlabel('$U_{\inf}t/c$')
plt.ylabel('C_L')
plt.savefig('C_L_m3l.png',dpi=300,transparent=True)
plt.show()




from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, ProfileOpModel, ProfileOpModel2, PostProcessor
from python_csdl_backend import Simulator
from VAST.utils.generate_mesh import *
import m3l
import csdl
import caddee.api as cd
from expansion_op import ac_expand
from generate_ground_effect_mesh import generate_ground_effect_mesh

def GE_sweep(nx, ny, num_nodes, AR, span, alpha_list, h_list):
    chord = span/AR
    nt = num_nodes + 1

    num_alpha = len(alpha_list)
    num_h = len(h_list)

    CL = np.zeros((num_alpha, num_h))
    CL_image = np.zeros_like(CL)
    CDi = np.zeros_like(CL)
    CDi_image = np.zeros_like(CL)

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

    for alpha in alpha_list:
        for h in h_list:
            mesh, image_mesh = generate_ground_effect_mesh(mesh_temp, alpha, h, test_plot=False)

            mesh_val = np.zeros((num_nodes, nx, ny, 3))
            image_mesh_val = np.zeros((num_nodes, nx, ny, 3))
            for j in range(num_nodes):
                mesh_val[j, :, :, 0] = mesh.copy()[:, :, 0]
                mesh_val[j, :, :, 1] = mesh.copy()[:, :, 1]
                mesh_val[j, :, :, 2] = mesh.copy()[:, :, 2]

                image_mesh_val[j, :, :, 0] = image_mesh.copy()[:, :, 0]
                image_mesh_val[j, :, :, 1] = image_mesh.copy()[:, :, 1]
                image_mesh_val[j, :, :, 2] = image_mesh.copy()[:, :, 2]

            sim_GE = create_GE_sim(mesh=mesh_val, image_mesh=image_mesh_val)

            sim_GE.run()
            

    return CL, CL_image, CDi, CDi_image

def create_GE_sim(mesh, image_mesh):
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
    
    uvlm_parameters.append(('wing', True, mesh),)
    uvlm_parameters.append(('wing_image', True, image_mesh),)
    surface_names = [
        'wing',
        'wing_image'
    ]
    nx, ny = mesh.shape[1], mesh.shape[2]
    surface_shapes = [
        (nx,ny,3),
        (nx,ny,3),
    ]
    h_stepsize = delta_t = 1/16

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

    submodel = PostProcessor(
        num_nodes = num_nodes,
        surface_names = surface_names,
        surface_shapes = surface_shapes,
        delta_t = h_stepsize,
        nt = num_nodes + 1,
        symmetry=True,
        frame='inertial'
    )
    pp_vars = [
        ('wing_C_L', (num_nodes-1, 1)), 
        ('wing_image_C_L', (num_nodes-1, 1))
    ]

    model = m3l.DynamicModel()
    uvlm = VASTSolverUnsteady(
        num_nodes=num_nodes, 
        surface_names=surface_names, 
        surface_shapes=surface_shapes, 
        delta_t=delta_t, 
        nt=num_nodes+1,
        frame='inertial'
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
                                profile_outputs=None,
                                profile_system=None,
                                profile_parameters=None,
                                post_processor=submodel,
                                pp_vars=pp_vars)
    uvlm_op = model.assemble(return_operation=True)

    wing_C_L, wing_image_C_L, _, _, _, _  = uvlm_op.evaluate()

    overmodel = m3l.Model()
    overmodel.register_output(wing_C_L)
    overmodel.register_output(wing_image_C_L)
    model_csdl = overmodel.assemble()
    

    sim = Simulator(model_csdl, analytics=True, name='GE')
    return sim

AR = 8
span = 8
nx = 5
ny = 13
num_nodes = 20


h_list = [1]
# h_list.insert(0,span) # ensures first entry is OGE
alpha_list = [15] 

CL, CL_image, CDi, CDi_image = GE_sweep(nx, ny, num_nodes, AR, span, alpha_list, h_list)
from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, ProfileOpModel, ProfileOpModel2, PostProcessor
from VAST.core.profile_model import ProfileOPModel3
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_system import ODESystemModel
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
num_nodes = 30

num_surfaces = 8
surface_offset = [0,0,2]
lazy = 1

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
wig_condition.set_module_input(name='mach_number', val=0.21623, dv_flag=False, lower=0.1, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(5), dv_flag=True, lower=np.deg2rad(-10), upper=np.deg2rad(10))
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


sub = True

def generate_sub_lists(interaction_groups):
    sub_eval_list = []
    sub_induced_list = []
    for group in interaction_groups:
        for i in group:
            for j in group:
                sub_eval_list.append(i)
                sub_induced_list.append(j)
    return sub_eval_list, sub_induced_list

groups = []
for i in range(int(num_surfaces/4)):
    groups.append(list(range(i*4,i*4+4)))
# groups = [list(range(0,num_surfaces))]
sub_eval_list, sub_induced_list = generate_sub_lists(groups)


profile_parameters = {
    # num_nodes : num_nodes-1,
    'surface_names' : surface_names,
    'surface_shapes' : surface_shapes,
    'delta_t' : h_stepsize,
    'nt' : num_nodes,
    'sub' : sub,
    'sub_eval_list' : sub_eval_list,
    'sub_induced_list' : sub_induced_list
}
# pp_vars = [('panel_forces', (num_nodes, system_size, 3)), ('eval_pts_all', (num_nodes, system_size, 3))]
pp_vars = []
for name in surface_names:
    pp_vars.append((name+'_L', (1,1)))


model = m3l.DynamicModel()
uvlm = VASTSolverUnsteady(num_nodes=num_nodes, surface_names=surface_names, surface_shapes=surface_shapes, delta_t=delta_t, nt=num_nodes+1, sub=sub, sub_eval_list=sub_eval_list, sub_induced_list=sub_induced_list)
uvlm_residual = uvlm.evaluate()
model.register_output(uvlm_residual)
model.set_dynamic_options(initial_conditions=initial_conditions,
                            num_times=num_nodes,
                            h_stepsize=delta_t,
                            parameters=uvlm_parameters,
                            int_naming=('op_',''),
                            integrator='ForwardEuler',
                            approach='time-marching',
                            profile_outputs=pp_vars,
                            profile_system=None,
                            profile_parameters=None,
                            copycat_profile=True,
                            post_processor=None,
                            pp_vars=None)
uvlm_op = model.assemble(return_operation=True)

lift_vars = uvlm_op.evaluate()[0:num_surfaces]

overmodel = m3l.Model()
for var in lift_vars:
    overmodel.register_output(var)
model_csdl = overmodel.assemble()

last_lifts = model_csdl.create_output('last_lifts', shape=(num_surfaces,1))
i = 0
for name in surface_names:
    lift = model_csdl.create_input(name+'_lift', shape=(num_nodes, 1))
    model_csdl.connect('operation.prob.'+name+'_L', name+'_lift')
    last_lifts[i,0] = lift[-1,0]
    i += 1
lift_last = csdl.pnorm(last_lifts)
model_csdl.register_output('lift_norm', lift_last)

model_csdl.add_objective('lift_norm', scaler=1e-3)



sim = python_csdl_backend.Simulator(model_csdl, analytics=True, lazy=lazy)

sim.run()
sim.compute_total_derivatives()
# sim.check_totals()
# exit()



import time
start = time.time()
sim.run()
sim.compute_total_derivatives()
end = time.time()

print('Total run time:')
print(end-start)

# sim.check_totals()
# exit()


# from modopt.scipy_library import SLSQP
# from modopt.csdl_library import CSDLProblem

# prob = CSDLProblem(problem_name='pav', simulator=sim)
# optimizer = SLSQP(prob, maxiter=50, ftol=1E-5)

# optimizer.solve()
# optimizer.print_results()

# print(sim['operation.input_model.wig_ac_states_operation.wig_pitch_angle'])
# print(sim['last_lifts'])


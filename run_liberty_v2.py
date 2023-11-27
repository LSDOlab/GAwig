import caddee.api as cd
import m3l
import time
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface
import array_mapper as am
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
import csdl
from mirror import Mirror
from rotor import Rotor3
from expansion_op import ac_expand
from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, PostProcessor
from deflect_flap import deflect_flap
from VAST.core.profile_model import gen_profile_output_list, PPSubmodel
from last_n_average import LastNAverage
from plot import plot_wireframe, plot_wireframe_line, plot_lift_spanwise
from engine import Engine
from torque_model import TorqueModel
# from breguet_range_eqn import BreguetRange
# from modopt.snopt_library import SNOPT
# from mpi4py import MPI



# region hyperparameters
num_props = 4 # must be even
num_blades = 2
rpm = 1090. # fixed rpm
nt = 18
dt = 0.005 # sec
h = 2.5 # the height (m) from the image plane to the rotation_point
pitch = 0.039425 # np.deg2rad(3) # rad
rotor_blade_angle = -0.211512# -0.30411512 # np.deg2rad(-4) # rad (negative is more thrust)
rotation_point = np.array([24,0,0]) # np.array([37,0,0]) with fuselages
do_wing = True
do_flaps = True
do_fuselage = False
mirror = True
sub = True
free_wake = True
symmetry = True # only works with mirror = True
log_space = False # log spacing spanwise for wing mesh

# airplane params:
max_pwr = 4500. # hp
m = 150000. # kg
wing_area = 550 # m^2

# VLM params:
core_size = 0.5 # set the viscous core size
# wing:
num_spanwise_vlm = 31
num_chordwise_vlm = 10
# prop:
num_spanwise_prop= 5
num_chordwise_prop = 2
# fuse:
num_long_vlm = 6
num_vert_vlm = 3

# flap params:
flap_deflection = 20 # deg
flap_frac = 0.25 # percent of the chord deflected by the flap

# average the uvlm forces:
num_periods = 1.0
n_avg = int(num_periods/(rpm*dt/60))
print('n average: ', n_avg)

# set constraints on rotor delta: (lower, upper) for rotors 0, 1, 2, and 3
dx_const = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
dy_const = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
dz_const = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

# set the baseline rotor deltas: rotors 0, 1, 2, 3
dxlist = [0,0,0,0]
dylist = [0,0,0,0]
dzlist = [0,0,0,0]
# endregion





# region caddee setup
file_name = 'LibertyLifter3.stp'
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=file_name)
spatial_rep.refit_geometry(file_name=file_name)
# spatial_rep.plot(plot_types=['mesh'])
# endregion



# region components: returns component with given name made from surfaces containing search_names
def build_component(name, search_names):
    primitive_names = list(spatial_rep.get_primitives(search_names=search_names).keys())
    component = LiftingSurface(name=name, spatial_representation=spatial_rep, primitive_names=primitive_names)
    sys_rep.add_component(component)
    return component

wing, htail, fuse = build_component('wing', ['WingGeom']), build_component('htail', ['HTail']), build_component('fuse', ['FuselageGeom'])


# props
props, prop_indices = [], list(range(0,int(num_props/2))) + list(range(int(8-num_props/2),8))
for i in range(num_props):
    prop = build_component('prop_'+str(i), ['Prop'+str(prop_indices[i]+1),'Hub'+str(prop_indices[i]+1)])
    props.append(prop)
#endregion



# region meshes

# create the wing mesh:
if log_space:
    start, end = 0.001, 1.0
    num_spanwise_temp = num_spanwise_vlm + 1 # used for log spacing

    le_half_points = ((((np.logspace(start, end, int(num_spanwise_temp/2), endpoint=True)))-10**start)/(10**end-10**start)-1)*103
    le_points = np.concatenate([le_half_points, np.flip(le_half_points*-1)[1:]])
    le_points = np.vstack((30*np.ones(num_spanwise_vlm), le_points, 6*np.ones(num_spanwise_vlm))).T
    leading_edge = wing.project(le_points, direction=np.array([0., 0., -1.]), plot=False)

    te_half_points = ((((np.logspace(start, end, int(num_spanwise_temp/2), endpoint=True)))-10**start)/(10**end-10**start)-1)*105
    te_points = np.concatenate([te_half_points, np.flip(te_half_points*-1)[1:]])
    te_points = np.vstack((80*np.ones(num_spanwise_vlm), te_points, 6*np.ones(num_spanwise_vlm))).T
    trailing_edge = wing.project(te_points, direction=np.array([0., 0., -1.]), plot=False)

else:
    leading_edge = wing.project(np.linspace(np.array([30, -103, 6]), np.array([30, 103, 6]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
    trailing_edge = wing.project(np.linspace(np.array([80, -105, 6]), np.array([80, 105, 6]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
    chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)


chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
# spatial_rep.plot_meshes([chord_surface])
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 5.]), direction=np.array([0., 0., -5.]), grid_search_n=30, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 5.]), direction=np.array([0., 0., 5.]), grid_search_n=30, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
# spatial_rep.plot_meshes([wing_camber_surface])
wing_camber_surface_np = wing_camber_surface.value.reshape((num_chordwise_vlm, num_spanwise_vlm, 3))
wing_vlm_mesh_name = 'wing_vlm_mesh'




# add flap deflections:
if do_flaps:
    flap_mesh = deflect_flap(wing_camber_surface_np, flap_deflection, int(num_chordwise_vlm*flap_frac))
    wing_camber_surface_np = flap_mesh
    # spatial_rep.plot_meshes([flap_mesh])



# split the wing mesh to exploit symmetry:
if symmetry:
    wing_camber_surface_np_neg_y = np.flip(wing_camber_surface_np[:,:int((num_spanwise_vlm+1)/2),:].copy(),1)
    wing_camber_surface_np_pos_y = wing_camber_surface_np[:,int((num_spanwise_vlm-1)/2):,:].copy()




# right fuselage mesh:
rtop = fuse.project(np.linspace(np.array([0, 27, -0.25]), np.array([120, 27, 9]), num_long_vlm+2)[1:-1], direction=np.array([0., 0., -1.]), plot=False)
rbot = fuse.project(np.linspace(np.array([0, 27, -10]), np.array([120, 27, -2]), num_long_vlm+2)[1:-1], direction=np.array([0., 0., -1.]), plot=False)
right_fuse_surface = am.linspace(rtop, rbot, num_vert_vlm)
right_fuse_surface_reordered = np.swapaxes(right_fuse_surface.value, 0, 1)
# spatial_rep.plot_meshes([right_fuse_surface])
right_fuse_mesh_name = 'right_fuselage_mesh'
sys_rep.add_output(right_fuse_mesh_name, right_fuse_surface)

# left fuselage mesh:
ltop = fuse.project(np.linspace(np.array([0, -27, -0.25]), np.array([120, -27, 9]), num_long_vlm+2)[1:-1], direction=np.array([0., 0., -1.]), plot=False)
lbot = fuse.project(np.linspace(np.array([0, -27, -10]), np.array([120, -27, -2]), num_long_vlm+2)[1:-1], direction=np.array([0., 0., -1.]), plot=False)
left_fuse_surface = am.linspace(ltop, lbot, num_vert_vlm)
left_fuse_surface_reordered = np.swapaxes(left_fuse_surface.value, 0, 1)
# spatial_rep.plot_meshes([left_fuse_surface])
left_fuse_mesh_name = 'left_fuselage_mesh'
sys_rep.add_output(left_fuse_mesh_name, left_fuse_surface)





# prop meshes
offsets = [0,20,20,38,18,38,20,20] # gaps between rotors, left to right
p1 = [39.754, -88.35, 4.769]
p2 = [39.848-0.3, -93.75, 4.342-0.5]
p3 = [40.246, -88.35, 5.231]
p4 = [40.152+0.3, -93.75, 5.658+0.5]
p5 = [40., -87., 5.]
p6 = [37., -87., 5.]

# create mesh for prop 0, blade 0
if num_props > 0:
    leading_edge = props[0].project(np.linspace(np.array(p1), np.array(p2), num_spanwise_prop), direction=np.array([0., 0, -1.]), grid_search_n=50, plot=False)
    trailing_edge = props[0].project(np.linspace(np.array(p3), np.array(p4), num_spanwise_prop), direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False)
    chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_prop)
    prop0b0_mesh = chord_surface.value
    propb1_mesh_names, prop_vector_names, prop_point_names = [], [], []
    prop_meshes_np = [prop0b0_mesh]


# list of hub front/back for different rotors
p5_list, p6_list = [p5], [p6]
for i in range(1,8):
    p5_list.append([p5[0], p5_list[i-1][1]+offsets[i], p5[2]])
    p6_list.append([p6[0], p6_list[i-1][1]+offsets[i], p6[2]])



prop_points = []
for i in range(num_props):
    propb1_mesh_name = 'p'+str(i)+'b1_mesh'

    # prop hub:
    hub_back, hub_front = props[i].project(np.array(p5_list[prop_indices[i]])), props[i].project(np.array(p6_list[prop_indices[i]]))
    prop_vec = hub_front - hub_back
    prop_points.append(hub_back.value)
    prop_vector_name, prop_point_name = 'p' + str(i) + '_vector', 'p' + str(i) + '_point'
    sys_rep.add_output(prop_vector_name, prop_vec)
    sys_rep.add_output(prop_point_name, hub_back)
    propb1_mesh_names.append(propb1_mesh_name)
    prop_vector_names.append(prop_vector_name)
    prop_point_names.append(prop_point_name)

for i in range(1,num_props):
    if i < num_props/2:
        shape = prop0b0_mesh.shape[0:2]
        offset = np.zeros(prop0b0_mesh.shape)
        offset[:,:,1] = offsets[i]*np.ones(shape)
        prop_meshes_np.append(prop_meshes_np[i-1] + offset)
    else:
        ref_mesh = prop_meshes_np[num_props-1 - i].copy()
        ref_mesh[:,:,1] = -1*ref_mesh[:,:,1]
        prop_meshes_np.append(ref_mesh)
# endregion




# region design scenario
overmodel = m3l.Model()

design_scenario = cd.DesignScenario(name='wig')
wig_condition = cd.CruiseCondition(name='wig')
wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
wig_condition.set_module_input(name='altitude', val=0)
wig_condition.set_module_input(name='mach_number', val=0.14169198, dv_flag=True)
wig_condition.set_module_input(name='range', val=1000)
# ptich angle is always zero, mirroring functions apply pitch by offsetting meshes:
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0))
wig_condition.set_module_input(name='flight_path_angle', val=0)
wig_condition.set_module_input(name='roll_angle', val=0)
wig_condition.set_module_input(name='yaw_angle', val=0)
wig_condition.set_module_input(name='wind_angle', val=0)
wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))

# expand states across time steps for ode
ac_states = wig_condition.evaluate_ac_states()
ac_expander = ac_expand(num_nodes=nt)
ac_states_expanded = ac_expander.evaluate(ac_states)
# endregion

# region mirroring

h_m3l = m3l.Variable('h', shape=(1,), value=np.array([h]))
non_rotor_surfaces = []
# wing mirroring
if symmetry:
    wing_mirror_model_neg_y = Mirror(component=wing,mesh_name=wing_vlm_mesh_name + '_neg_y',nt=nt,ns=int((num_spanwise_vlm+1)/2),nc=num_chordwise_vlm,point=rotation_point, mesh=wing_camber_surface_np_neg_y*0.3048)
    wing_mesh_neg_y_out, wing_mirror_neg_y_mesh = wing_mirror_model_neg_y.evaluate(h_m3l)

    wing_mirror_model_pos_y = Mirror(component=wing,mesh_name=wing_vlm_mesh_name + '_pos_y',nt=nt,ns=int((num_spanwise_vlm+1)/2),nc=num_chordwise_vlm,point=rotation_point, mesh=wing_camber_surface_np_pos_y*0.3048)
    wing_mesh_pos_y_out, wing_mirror_pos_y_mesh = wing_mirror_model_pos_y.evaluate(h_m3l)

    if do_wing:
        non_rotor_surfaces.extend([wing_mesh_neg_y_out, wing_mesh_pos_y_out])
        if mirror:
            non_rotor_surfaces.extend([wing_mirror_neg_y_mesh, wing_mirror_pos_y_mesh])
else:
    wing_mirror_model = Mirror(component=wing,mesh_name=wing_vlm_mesh_name,nt=nt,ns=num_spanwise_vlm,nc=num_chordwise_vlm,point=rotation_point, mesh=wing_camber_surface_np*0.3048)
    wing_mesh_out, wing_mirror_mesh = wing_mirror_model.evaluate(h_m3l)
    if do_wing:
        non_rotor_surfaces.append(wing_mesh_out)
        if mirror:
            non_rotor_surfaces.append(wing_mirror_mesh)

# right fuselage mirroring
right_fuse_mirror_model = Mirror(component=fuse,mesh_name=right_fuse_mesh_name,nt=nt,ns=num_vert_vlm,nc=num_long_vlm,point=rotation_point, mesh=right_fuse_surface_reordered*0.3048)
right_fuse_mesh_out, right_fuse_mirror_mesh = right_fuse_mirror_model.evaluate(h_m3l)

# left fuselage mirroring
left_fuse_mirror_model = Mirror(component=fuse,mesh_name=left_fuse_mesh_name,nt=nt,ns=num_vert_vlm,nc=num_long_vlm,point=rotation_point, mesh=left_fuse_surface_reordered*0.3048)
left_fuse_mesh_out, left_fuse_mirror_mesh = left_fuse_mirror_model.evaluate(h_m3l)

if do_fuselage:
    non_rotor_surfaces.append(left_fuse_mesh_out)
    non_rotor_surfaces.append(right_fuse_mesh_out)
    if mirror:
        non_rotor_surfaces.append(left_fuse_mirror_mesh)
        non_rotor_surfaces.append(right_fuse_mirror_mesh)


prop_meshes = []
prop_meshes_vel = []
for i in range(num_props):
    direction = -1
    if i >= num_props/2: direction = 1
    
    prop_model = Rotor3(mesh_name = propb1_mesh_names[i], 
                        num_blades = num_blades, 
                        ns = num_spanwise_prop, 
                        nc = num_chordwise_prop, 
                        nt = nt, 
                        dt = dt, 
                        dir = direction, 
                        r_point = rotation_point,
                        mesh = prop_meshes_np[i],
                        rpm = rpm,
                        point = prop_points[i])
    prop_mesh_out, mirror_prop_meshes, prop_mesh_out_vel, prop_mesh_mirror_vel = prop_model.evaluate(h_m3l)

    if mirror:
        prop_meshes.append(prop_mesh_out + mirror_prop_meshes)
        prop_meshes_vel.append(prop_mesh_out_vel + prop_mesh_mirror_vel)
    else:
        prop_meshes.append(prop_mesh_out)
        prop_meshes_vel.append(prop_mesh_out_vel)



if mirror: num_blades = num_blades*2
# endregion




# region uvlm_parameters, interactions
uvlm_parameters = [('u',True,ac_states_expanded['u']),
                    ('v',True,ac_states_expanded['v']),
                    ('w',True,ac_states_expanded['w']),
                    ('p',True,ac_states_expanded['p']),
                    ('q',True,ac_states_expanded['q']),
                    ('r',True,ac_states_expanded['r']),
                    ('theta',True,ac_states_expanded['theta']),
                    ('psi',True,ac_states_expanded['psi']),
                    ('gamma',True,ac_states_expanded['gamma']),
                    ('psiw',True,np.zeros((nt, 1)))]

# TODO: connect rho, maybe other values

def generate_sub_lists(interaction_groups):
    # generates sub lists for groups of full interaction with no cross-interaction
    sub_eval_list, sub_induced_list = [], []
    for group in interaction_groups:
        for i in group:
            for j in group:
                sub_eval_list.append(i)
                sub_induced_list.append(j)
    return sub_eval_list, sub_induced_list

# ode stuff for props
surface_names, surface_shapes, initial_conditions, interaction_groups, symmetry_list, i = [], [], [], [], [], 0
for prop_mesh in prop_meshes:
    interaction_groups.append(list(range(num_blades*i,num_blades*(i+1))))
    for var in prop_mesh:
        shape = var.shape
        nx = shape[1]
        ny = shape[2]
        name = var.name
        surface_names.append(name)
        surface_shapes.append(shape[1:4])
        uvlm_parameters.append((name, True, var))
        # uvlm_parameters.append((name+'_coll_vel', True, np.zeros((nt, nx-1, ny-1, 3))))
        initial_conditions.append((name+'_gamma_w_0', np.zeros((nt-1, ny-1))))
        initial_conditions.append((name+'_wake_coords_0', np.zeros((nt-1, ny, 3))))
    i += 1

for prop_mesh_vel in prop_meshes_vel:
    for vel in prop_mesh_vel:
        shape = vel.shape
        nx = shape[1]
        ny = shape[2]
        name = vel.name
        uvlm_parameters.append((f'{name}', True, vel))

# interactions for props
sub_eval_list, sub_induced_list = generate_sub_lists(interaction_groups)

# ode stuff and interactions for non-rotors:
for surface in non_rotor_surfaces:
    # ode parameters and ICs
    surface_names.append(surface.name)
    num_chordwise = surface.shape[1]
    num_spanwise = surface.shape[2]
    surface_shapes.append(surface.shape[1:4])
    uvlm_parameters.append((surface.name, True, surface))
    # uvlm_parameters.append((surface.name+'_coll_vel', True, np.zeros((nt, num_chordwise-1, num_spanwise-1, 3))))
    initial_conditions.append((surface.name+'_gamma_w_0', np.zeros((nt-1, num_spanwise-1))))
    initial_conditions.append((surface.name+'_wake_coords_0', np.zeros((nt-1, num_spanwise, 3))))

    # interactions - fully interactive
    index = len(surface_names)-1
    for i in range(index):
        sub_eval_list.append(i)
        sub_induced_list.append(index)
        sub_eval_list.append(index)
        sub_induced_list.append(i)
    sub_eval_list.append(index)
    sub_induced_list.append(index)
# endregion

# region symmetry
if symmetry:
    # symmetry for props
    for i in range(int(num_props/2)):
        for j in range(int(num_blades/2)):
            symmetry_list.append([int(i*num_blades+j),                                     # left blade
                                int((num_props-1-i)*num_blades + j),                       # right blade
                                int(i*num_blades+j + num_blades/2),                        # left mirror blade
                                int((num_props-1-i)*num_blades + j + num_blades/2)])       # right mirror blade

    # symmetry for wing
    if do_wing:
        if mirror:
            wing_list = [wing_mesh_neg_y_out, wing_mesh_pos_y_out, wing_mirror_neg_y_mesh, wing_mirror_pos_y_mesh]
        else:
            wing_list = [wing_mesh_neg_y_out, wing_mesh_pos_y_out]
        wing_indices = []
        for val in wing_list:
            wing_index = surface_names.index(val.name)
            wing_indices.append(wing_index)
        symmetry_list.append(wing_indices)

    # symmetry for fuselages
    if right_fuse_mesh_out.name in surface_names:
        fuselage0_index = surface_names.index(right_fuse_mesh_out.name)
        fuselage1_index = surface_names.index(left_fuse_mesh_out.name)
        fuselage0_mirror_index = surface_names.index(right_fuse_mirror_mesh.name)
        fuselage1_mirror_index = surface_names.index(left_fuse_mirror_mesh.name)
        symmetry_list.append([fuselage0_index, fuselage1_index, fuselage0_mirror_index, fuselage1_mirror_index])
# endregion




# region post-processor and profile-output
pp_vars, num_panels = [], 0
if do_wing:
    num_panels += (num_spanwise_vlm-1)*(num_chordwise_vlm-1)
if do_fuselage:
    num_panels += (num_long_vlm-1)*(num_vert_vlm-1)*2
if mirror:
    num_panels += num_props*num_blades/2*(num_spanwise_prop-1)*(num_chordwise_prop-1)
    num_panels = int(2*num_panels)
else:
    num_panels += num_props*num_blades*(num_spanwise_prop-1)*(num_chordwise_prop-1)
    num_panels = int(num_panels)

pp_vars.append(('panel_forces_x',(nt,num_panels,1)))
pp_vars.append(('panel_forces_y',(nt,num_panels,1)))
pp_vars.append(('panel_forces_z',(nt,num_panels,1)))

if do_wing:
    pp_vars.append(('wing_vlm_mesh_neg_y_out_L', (nt, 1)))
    pp_vars.append(('wing_vlm_mesh_neg_y_out_D', (nt, 1)))
    pp_vars.append(('wing_vlm_mesh_pos_y_out_L', (nt, 1)))
    pp_vars.append(('wing_vlm_mesh_pos_y_out_D', (nt, 1)))

for i in range(num_props):
    for j in range(int(num_blades/2)):
        pp_vars.append((f'p{i}b1_mesh_rotor{j}_out_panel_forces_x', (nt, num_spanwise_prop-1)))


profile_outputs = gen_profile_output_list(surface_names, surface_shapes)
ode_surface_shapes = [(nt, ) + item for item in surface_shapes]
post_processor = PPSubmodel(surface_names = surface_names, ode_surface_shapes=ode_surface_shapes, delta_t=dt, nt=nt+1, symmetry=False)
# endregion

# region uvlm
model = m3l.DynamicModel()
uvlm = VASTSolverUnsteady(num_nodes = nt, 
                          surface_names = surface_names, 
                          surface_shapes = surface_shapes, 
                          delta_t = dt, 
                          nt = nt+1,
                          sub = sub,
                          sub_eval_list = sub_eval_list,
                          sub_induced_list = sub_induced_list,
                          symmetry = symmetry,
                          sym_struct_list = symmetry_list,
                          free_wake=free_wake,
                          core_size=core_size,)
uvlm_residual = uvlm.evaluate()
model.register_output(uvlm_residual)
model.set_dynamic_options(initial_conditions=initial_conditions,
                          num_times=nt,
                          h_stepsize=dt,
                          parameters=uvlm_parameters,
                          int_naming=('op_',''),
                          integrator='ForwardEuler',
                          approach='time-marching',
                          copycat_profile=True,
                          profile_outputs=profile_outputs,
                          post_processor=post_processor,
                          pp_vars=pp_vars)
uvlm_op = model.assemble(return_operation=True)
outputs = uvlm_op.evaluate()[0:len(pp_vars)]
# endregion




# region post-processing
average_op = LastNAverage(n=n_avg,end_offset=1)
# time averaged quantities:
ave_outputs = average_op.evaluate(outputs)
fx, fy, fz = ave_outputs[0], ave_outputs[1], ave_outputs[2]

offset = 3
if do_wing:
    if symmetry:
        wing_lift_neg_y = ave_outputs[3]
        wing_drag_neg_y = ave_outputs[4]
        wing_lift_pos_y = ave_outputs[5]
        wing_drag_pos_y = ave_outputs[6]
        offset = 7
    else:
        wing_lift, wing_drag = ave_outputs[3], ave_outputs[4]
        offset = 5


prop_fx_list = []
for i in range(num_props):
    blade_forces = []
    for j in range(int(num_blades/2)):
        blade_forces.append(ave_outputs[i*int(num_blades/2)+offset+j])
    prop_fx_list.append(tuple(blade_forces))

for var in ave_outputs:
    overmodel.register_output(var)


for i in range(len(prop_fx_list)):
        torque_model = TorqueModel(rotor_name=f'rotor_{i}')
        torque = torque_model.evaluate(prop_fx_list[i])
        engine_model = Engine(engine_name=f'engine_{i}')
        fc, pwr = engine_model.evaluate(torque)    
        overmodel.register_output(fc)
        overmodel.register_output(pwr)
# endregion







# region model assembly
# add the wig m3l model to the cruise condition:
wig_condition.add_m3l_model('wig_model', overmodel)
# add the design condition to the design scenario:
design_scenario.add_design_condition(wig_condition)
system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()
# caddee_csdl_model = overmodel.assemble_csdl()
model_csdl = caddee_csdl_model
# endregion



# region manual connections
for i in range(len(prop_meshes)):
    i = str(i)

    rpm_csdl = model_csdl.create_input(f'rpm_rotor_{i}', val=rpm)
    model_csdl.connect('p' + i + '_vector', 'system_model.wig.wig.wig.operation.input_model.p' + i + 'b1_mesh_rotor.vector')
    model_csdl.connect(f'rpm_rotor_{i}', f'system_model.wig.wig.wig.torque_operation_rotor_{i}.rpm')
    model_csdl.connect(f'rpm_rotor_{i}', f'system_model.wig.wig.wig.engine_{i}_engine.rpm')
    model_csdl.connect('system_model.wig.wig.wig.operation.input_model.wig_ac_states_operation.u', f'system_model.wig.wig.wig.torque_operation_rotor_{i}.velocity')

# endregion









# the pitch angle design variable:
set_pitch = model_csdl.create_input('set_pitch', val=pitch)
# add the pitch angle design variable:
model_csdl.add_design_variable('set_pitch', lower=np.deg2rad(0), scaler=1E1)
# print the pitch angle during optimization:
model_csdl.print_var(set_pitch)
# connect set_pitch to the wing mirror:
if do_wing:
    model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.wing_vlm_mesh_pos_ymirror.theta')
    model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.wing_vlm_mesh_neg_ymirror.theta')
# connect set_pitch to the fuselage mirrors:
if do_fuselage:
    model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.right_fuselage_meshmirror.theta')
    model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.left_fuselage_meshmirror.theta')
# connect set_pitch to the rotors:
for i in range(num_props): model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.p'+str(i)+'b1_mesh_rotor.theta')




# blade angle design variables (for each prop):
for i in range(int(num_props/2)):
    blade_angle = model_csdl.create_input('blade_angle_'+str(i), val=rotor_blade_angle)
    # add the design variable for each prop:
    model_csdl.add_design_variable('blade_angle_'+str(i), upper=np.deg2rad(20), lower=np.deg2rad(-20), scaler=1E1)
    model_csdl.connect('blade_angle_'+str(i), 'system_model.wig.wig.wig.operation.input_model.p'+str(i)+'b1_mesh_rotor.blade_angle')
    # symmetric blade-angle connections:
    model_csdl.register_output('other_blade_angle_'+str(i), -1*blade_angle)
    model_csdl.connect('other_blade_angle_'+str(i), 'system_model.wig.wig.wig.operation.input_model.p'+str(num_props - i - 1)+'b1_mesh_rotor.blade_angle')
    # print the blade angle during optimization:
    model_csdl.print_var(blade_angle)

# # blade angle design variables (for all props):
# blade_angle = model_csdl.create_input('blade_angle_', val=rotor_blade_angle)
# model_csdl.add_design_variable('blade_angle_', scaler=1E1)
# # print the blade angle during optimization:
# model_csdl.print_var(blade_angle)
# model_csdl.register_output('other_blade_angle_', -1*blade_angle)
# for i in range(int(num_props/2)):
#     model_csdl.connect('blade_angle_', 'system_model.wig.wig.wig.operation.input_model.p'+str(i)+'b1_mesh_rotor.blade_angle')
#     model_csdl.connect('other_blade_angle_', 'system_model.wig.wig.wig.operation.input_model.p'+str(num_props - i - 1)+'b1_mesh_rotor.blade_angle')




# rotor delta design variables:
# NOTE: negative delta_z is moving rotors down
# NOTE: negative delta_y is moving rotors outwards
# NOTE: negative delta_x is moving rotors forwards
for i in range(int(num_props/2)):
    delta_x, delta_y, delta_z, = model_csdl.create_input('delta_x_'+str(i), val=dxlist[i]), model_csdl.create_input('delta_y_'+str(i), val=dylist[i]), model_csdl.create_input('delta_z_'+str(i), val=dzlist[i])
    model_csdl.add_design_variable('delta_x_'+str(i), upper=dx_const[i][1], lower=dx_const[i][0], scaler=1E2)
    model_csdl.add_design_variable('delta_y_'+str(i), upper=dy_const[i][1], lower=dy_const[i][0], scaler=1E2)
    model_csdl.add_design_variable('delta_z_'+str(i), upper=dz_const[i][1], lower=dz_const[i][0], scaler=1E2)

    # concatenate delta_x, y, and z:
    delta = model_csdl.create_output('delta_'+str(i), shape=(3), val=0)
    delta[0], delta[1], delta[2] = delta_x, delta_y, delta_z
    # connect the delta variable to the rotor mesh:
    model_csdl.connect('delta_'+str(i), 'system_model.wig.wig.wig.operation.input_model.p'+str(i)+'b1_mesh_rotor.delta')

    # symmetric delta connections:
    other_delta = model_csdl.create_output('other_delta_'+str(i), shape=(3,), val=0)
    other_delta[0], other_delta[1], other_delta[2] = delta[0], -1*delta[1], delta[2]
    model_csdl.connect('other_delta_'+str(i), 'system_model.wig.wig.wig.operation.input_model.p'+str(num_props - i - 1)+'b1_mesh_rotor.delta')

    # print the rotor delta during optimization:
    model_csdl.print_var(delta)



# the max engine power constraint:
power_vector = model_csdl.create_output('power_vector', shape=(len(prop_fx_list)), val=0)
for i in range(len(prop_fx_list)):
    eng_pwr = model_csdl.declare_variable('system_model.wig.wig.wig.'+f'engine_{i}'+'_engine.'+f'engine_{i}'+'_pwr')
    power_vector[i] = eng_pwr
# aggregate the max power with a ks max:
max_eng_pwr = model_csdl.register_output('max_eng_pwr', csdl.max(1E-2*power_vector)/1E-2)
# add a single aggregated max power constraint:
# model_csdl.add_constraint('max_eng_pwr', upper=max_pwr, scaler=1E-3)
model_csdl.print_var(max_eng_pwr)

# lift equals weight constraint:
L_neg_y_ave = model_csdl.declare_variable('system_model.wig.wig.wig.average_op.wing_vlm_mesh_neg_y_out_L_ave')
L_pos_y_ave = model_csdl.declare_variable('system_model.wig.wig.wig.average_op.wing_vlm_mesh_pos_y_out_L_ave')
L_tot_ave = model_csdl.register_output('L_tot_ave', L_neg_y_ave + L_pos_y_ave)
model_csdl.print_var(L_tot_ave)
fz_res = model_csdl.register_output('fz_res', (L_tot_ave - m*9.81))
model_csdl.print_var(fz_res)


# compute a viscous drag estimate:
velocity = model_csdl.declare_variable('system_model.wig.wig.wig.operation.input_model.wig_ac_states_operation.u')
other_drag_coef = 0.001 #2*0.02 #0.015
other_drag = model_csdl.register_output('other_drag', 0.5*1.225*velocity**2*wing_area*other_drag_coef) # 600m^2 not 6000ft^2


# panel_fx gives the total x-axis forces for the entire mirrored system:
panel_fx = model_csdl.declare_variable('system_model.wig.wig.wig.average_op.panel_forces_x_ave', shape=(num_panels, 1))
fx_res = model_csdl.register_output('fx_res', csdl.sum(1*panel_fx) + other_drag)
model_csdl.print_var(fx_res)


trim_res_vec = model_csdl.create_output('trim_res_vec', shape=(2), val=0)
trim_res_vec[0], trim_res_vec[1] = fz_res, fx_res
trim_res = model_csdl.register_output('trim_res', csdl.pnorm(trim_res_vec)/10)

# print the trim residual during optimization:
model_csdl.print_var(trim_res)
model_csdl.add_constraint('trim_res', equals=0, scaler=1E-2)


# print the velocity during optimization:
vel = model_csdl.register_output('vel', 1*velocity)
model_csdl.print_var(vel)

# create the objective:
obj = model_csdl.register_output('obj', 1*velocity)
model_csdl.add_objective('obj', scaler=1E-1)












start = time.time()

# for paralellization:
# mpi run command: mpirun -np 2 python run_liberty_v2.py
# comm = MPI.COMM_WORLD
# sim = Simulator(model_csdl, analytics=True, display_scripts=True, comm=comm,)

# for single core:
sim = Simulator(model_csdl, analytics=True, lazy=1)
sim.run()
# sim.check_partials(compact_print=True)
# sim.check_totals()

# run an optimization with SLSQP:
# prob = CSDLProblem(problem_name='gawig', simulator=sim)
# optimizer = SLSQP(prob, maxiter=300, ftol=1E-4)
# optimizer.solve()
# optimizer.print_results()

# if SNOPT:
# prob = CSDLProblem(problem_name='gawig', simulator=sim)
# optimizer = SNOPT(prob, 
#     Major_iterations=300,
#     Major_optimality=1e-3,
#     Major_feasibility=1e-3,
#     append2file=True,
#     # Major_step_limit=0.25,
#     Print_frequency=1,
#     )
# optimizer.solve()
# optimizer.print_results()



end = time.time()
print('total run time (s): ', end - start)

print('fz res: ', sim['fz_res'])
print('fx res: ', sim['fx_res'])

# print rotor thrust for all the props:
for i in range(num_props):
    print('rotor '+str(i)+' thrust (N): ', sim['system_model.wig.wig.wig.torque_operation_rotor_'+str(i)+'.total_thrust'])

# print the blade angle for half the props (symmetric):
for i in range(int(num_props/2)):
    print('blade angle '+str(i)+' (rad): ', sim['blade_angle_'+str(i)])

# print the rotor deltas for half the props (symmetric):
for i in range(int(num_props/2)):
    print('delta '+str(i)+': ', sim['delta_'+str(i)])

# print the velocity:
print('velocity (m/s): ', sim['system_model.wig.wig.wig.operation.input_model.wig_ac_states_operation.u'])


# plot the lift distribution across the half span (left):
plot_lift_spanwise(nt=nt, 
                   n_avg=n_avg, 
                   var=sim['system_model.wig.wig.wig.operation.wing_vlm_mesh_neg_y_out_L_panel'],
                   num_chordwise_vlm=num_chordwise_vlm,
                   num_span=int((num_spanwise_vlm - 1)/2),
                   num_props=num_props,
                   rpos=np.array([87,67,47,9])/102,)

# plot the lift distribution across the half span (right):
plot_lift_spanwise(nt=nt, 
                   n_avg=n_avg, 
                   var=sim['system_model.wig.wig.wig.operation.wing_vlm_mesh_pos_y_out_L_panel'],
                   num_chordwise_vlm=num_chordwise_vlm,
                   num_span=int((num_spanwise_vlm - 1)/2),
                   num_props=num_props,
                   rpos=np.array([87,67,47,9])/102,)


# plot the uvlm result:
plot_wireframe(sim, surface_names, nt, plot_mirror=True, interactive=False, name='test')
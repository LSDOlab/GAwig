# region imports
import caddee.api as cd
import m3l
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
from mpl_toolkits.mplot3d import proj3d
from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, PostProcessor
from deflect_flap import deflect_flap
from VAST.core.profile_model import gen_profile_output_list, PPSubmodel
from last_n_average import LastNAverage
# from plot_wing_symmetry import plot_wireframe
from plot import plot_wireframe
from engine import Engine
from torque_model import TorqueModel
# endregion

# region hyperparameters
num_props = 4
num_blades = 2
rpm = 1090.
nt = 20
dt_0 = 0.003 * 1
# dt = dt_0 + 60./rpm
dt = dt_0
h = 20 # m
pitch = np.deg2rad(0) # rad
rotor_blade_angle = np.deg2rad(0) # rad
rotor_delta = np.array([0,0,0]) # m
rotation_point = np.array([0,0,0])
do_wing = True
do_flaps = False
do_fuselage = False
mirror = True
sub = True
free_wake = True
symmetry = False # only works with mirror = True
log_space = False # log spacing spanwise for wing mesh
n_avg = int(nt/((rpm/60)*nt*(dt_0))) # 10
n_avg=15
print(n_avg)
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

# region components
def build_component(name, search_names):
    # returns component with given name made from surfaces containing search_names
    primitive_names = list(spatial_rep.get_primitives(search_names=search_names).keys())
    component = LiftingSurface(name=name, spatial_representation=spatial_rep, primitive_names=primitive_names)
    sys_rep.add_component(component)
    return component

wing = build_component('wing', ['WingGeom'])
htail = build_component('htail', ['HTail'])
fuse = build_component('fuse', ['FuselageGeom'])

# props
props = [] # we go from 1-indexed to 0-indexed here
prop_indices = list(range(0,int(num_props/2))) + list(range(int(8-num_props/2),8))
for i in range(num_props):
    prop = build_component('prop_'+str(i), ['Prop'+str(prop_indices[i]+1),'Hub'+str(prop_indices[i]+1)])
    props.append(prop)
#endregion

# region meshes
# wing mesh:
num_spanwise_vlm = 21
num_chordwise_vlm = 15

if log_space:
    start = 0.001
    end = 1
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
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 5.]), direction=np.array([0., 0., -2.]), grid_search_n=50, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 5.]), direction=np.array([0., 0., 2.]), grid_search_n=50, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
# spatial_rep.plot_meshes([wing_camber_surface])
# exit()
wing_camber_surface_np = wing_camber_surface.value.reshape((num_chordwise_vlm, num_spanwise_vlm, 3))
wing_vlm_mesh_name = 'wing_vlm_mesh'

if do_flaps:
    flap_mesh = deflect_flap(wing_camber_surface_np, 30, 1)
    # spatial_rep.plot_meshes([flap_mesh])
    wing_camber_surface_np = flap_mesh

if symmetry:
    wing_camber_surface_np_neg_y = np.flip(wing_camber_surface_np[:,:int((num_spanwise_vlm+1)/2),:].copy(),1)
    # wing_camber_surface_np_neg_y[-1,0,1] -= 1.
    # wing_camber_surface_np_neg_y[:,0,1] -= 1e-8
    # wing_camber_surface_np_neg_y[-1,0,1] = wing_camber_surface_np_neg_y[-1,1,1]
    wing_camber_surface_np_pos_y = wing_camber_surface_np[:,int((num_spanwise_vlm-1)/2):,:].copy()
    # wing_camber_surface_np_pos_y[-1,0,1] += 1.
    # wing_camber_surface_np_pos_y[:,0,1] += 1e-8
    # wing_camber_surface_np_pos_y[-1,0,1] = wing_camber_surface_np_pos_y[-1,1,1]

# right fuselage mesh:
num_long_vlm = 22
num_vert_vlm = 6
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

# htail mesh:
num_spanwise_vlm_htail = 12
num_chordwise_vlm_htail = 6
htail_leading_edge = htail.project(np.linspace(np.array([112, -27, 32]), np.array([112, 27, 32]), num_spanwise_vlm_htail), direction=np.array([0., 0., -1.]), plot=False)
htail_trailing_edge = htail.project(np.linspace(np.array([126, -27, 32]), np.array([126, 27, 32]), num_spanwise_vlm_htail), direction=np.array([0., 0., -1.]), plot=False)
htail_chord_surface = am.linspace(htail_leading_edge, htail_trailing_edge, num_chordwise_vlm_htail)
# spatial_rep.plot_meshes([htail_chord_surface])
htail_upper_surface_wireframe = htail.project(htail_chord_surface.value + np.array([0., 0., 2.]), direction=np.array([0., 0., -2.]), grid_search_n=30, plot=False)
htail_lower_surface_wireframe = htail.project(htail_chord_surface.value - np.array([0., 0., 2.]), direction=np.array([0., 0., 2.]), grid_search_n=30, plot=False)
htail_camber_surface = am.linspace(htail_upper_surface_wireframe, htail_lower_surface_wireframe, 1)
# spatial_rep.plot_meshes([htail_camber_surface])
htail_vlm_mesh_name = 'htail_vlm_mesh'
sys_rep.add_output(htail_vlm_mesh_name, htail_camber_surface)

# prop meshes
num_spanwise_prop= 5
num_chordwise_prop = 2
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

    propb1_mesh_names = []
    prop_vector_names = []
    prop_point_names = []
    prop_meshes_np = [prop0b0_mesh]

# list of hub front/back for different rotors
p5_list = [p5]
p6_list = [p6]
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
wig_condition.set_module_input(name='mach_number', val=0.2, dv_flag=True, lower=0.05, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
# ptich angle is always zero, mirroring functions apply pitch by offsetting meshes
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
pitch_m3l = m3l.Variable('pitch', shape=(1,), value=np.array([np.deg2rad(pitch)]))

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
prop_loc_names = []
prop_center_loc = []
prop_blade_names = []
prop_dir_list = []
for i in range(num_props):
    direction = -1
    if i >= num_props/2: direction = 1
    
    prop_dir_list.extend([direction] * num_blades)
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
    prop_loc_names.extend([propb1_mesh_names[i] + '_point_out'] * num_blades)
    prop_blade_names.extend([propb1_mesh_names[i] + '_rotor' + str(j) + '_out' for j in range(num_blades)])
    prop_mesh_out, mirror_prop_meshes, prop_center, prop_mirror_center = prop_model.evaluate(h_m3l)
    if mirror:
        prop_meshes.append(prop_mesh_out + mirror_prop_meshes)
        prop_center_loc.append(prop_center)
        prop_center_loc.append(prop_mirror_center)
        prop_loc_names.extend([propb1_mesh_names[i] + '_point_mirror'] * num_blades)
        prop_dir_list.extend([-1*direction] * num_blades)
        prop_blade_names.extend([propb1_mesh_names[i] + '_rotor' + str(j) + '_mirror' for j in range(num_blades)])
    else:
        prop_meshes.append(prop_mesh_out)
        prop_center_loc.append(prop_center)

if mirror:
    num_blades = num_blades*2
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
    # generates sub lists for groups of full interaction with no cross-interactoin
    sub_eval_list = []
    sub_induced_list = []
    for group in interaction_groups:
        for i in group:
            for j in group:
                sub_eval_list.append(i)
                sub_induced_list.append(j)
    return sub_eval_list, sub_induced_list

# ode stuff for props
surface_names = []
surface_shapes = []
initial_conditions = []
interaction_groups = []
i = 0

symmetry_list = []

polar_bool_list = []

prop_wake_coord_IC_list = [
np.array([[ 12.2669808,  -26.92908322,  21.59442859],
 [ 12.28267797, -27.34055999,  21.66876862],
 [ 12.29033299, -27.75167039,  21.7278122 ],
 [ 12.26585929, -28.16024803,  21.74613557],
 [ 12.23862588, -28.56938872,  21.72504884]]),

np.array([[ 12.2669808,  -26.10611678,  21.45357141],
 [ 12.28267797, -25.69464001,  21.37923138],
 [ 12.29033299, -25.28352961,  21.3201878 ],
 [ 12.26585929, -24.87495197,  21.30186443],
 [ 12.23862588, -24.46581128,  21.32295116]]),

np.array([[12.2669808,  26.92908322, 21.59442859],
 [12.28267797, 27.34055999, 21.66876862],
 [12.29033299, 27.75167039, 21.7278122 ],
 [12.26585929, 28.16024803, 21.74613557],
 [12.23862588, 28.56938872, 21.72504884]]),

np.array([[12.2669808,  26.10611678, 21.45357141],
 [12.28267797, 25.69464001, 21.37923138],
 [12.29033299, 25.28352961, 21.3201878 ],
 [12.26585929, 24.87495197, 21.30186443],
 [12.23862588, 24.46581128, 21.32295116]])
]
prop_wake_coord_IC = []
for n in range(4):
    prop_wake_coord_IC_mesh = np.zeros((nt-1, 5, 3))
    for m in range(nt-1):
        prop_wake_coord_IC_mesh[m,:,:] = prop_wake_coord_IC_list[n].copy()
    # prop_wake_coord_IC.append(prop_wake_coord_IC_mesh)

j = 0
for prop_mesh in prop_meshes:
    interaction_groups.append(list(range(num_blades*i,num_blades*(i+1))))
    for var in prop_mesh:
        shape = var.shape
        nx = shape[1]
        ny = shape[2]
        name = var.name
        surface_names.append(name)
        surface_shapes.append(shape[1:4])
        polar_bool_list.append(True)
        uvlm_parameters.append((name, True, var))
        # uvlm_parameters.append((name+'_coll_vel', True, np.zeros((nt, nx-1, ny-1, 3))))
        initial_conditions.append((name+'_gamma_w_0', np.zeros((nt-1, ny-1))))
        initial_conditions.append((name+'_wake_coords_0', np.zeros((nt-1, ny, 3))))
        # initial_conditions.append((name+'_wake_coords_0', prop_wake_coord_IC[j]))
        j += 1
    i += 1
for prop_center in prop_center_loc:
    print(prop_center.shape)
    print(prop_center.name)
    uvlm_parameters.append((prop_center.name, False, prop_center))
    # uvlm_parameters.append((prop_center))
# exit()
# for prop_mesh_vel in prop_meshes_vel:
#     for vel in prop_mesh_vel:
#         shape = vel.shape
#         nx = shape[1]
#         ny = shape[2]
#         name = vel.name
#         uvlm_parameters.append((f'{name}', True, vel))
# for i in range(num_props):
#     uvlm_parameters.append((propb1_mesh_names[i] + '_point', False, prop_points[i].reshape(3,)))
# interactions for props
sub_eval_list, sub_induced_list = generate_sub_lists(interaction_groups)

# ode stuff and interactions for non-rotors:
for surface in non_rotor_surfaces:
    # ode parameters and ICs
    surface_names.append(surface.name)
    polar_bool_list.append(False)
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
pp_vars = []
# for name in surface_names:
#     pp_vars.append((name+'_L', (nt, 1)))

num_panels = 0
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
    if symmetry:
        pp_vars.append(('wing_vlm_mesh_neg_y_out_L', (nt, 1)))
        pp_vars.append(('wing_vlm_mesh_neg_y_out_D', (nt, 1)))
        pp_vars.append(('wing_vlm_mesh_pos_y_out_L', (nt, 1)))
        pp_vars.append(('wing_vlm_mesh_pos_y_out_D', (nt, 1)))
    else:
        pp_vars.append(('wing_vlm_mesh_out_L', (nt, 1)))
        pp_vars.append(('wing_vlm_mesh_out_D', (nt, 1)))

for i in range(num_props):
    for j in range(int(num_blades/2)):
        pp_vars.append((f'p{i}b1_mesh_rotor{j}_out_panel_forces_x', (nt, num_spanwise_prop-1)))

if False:
    pp_vars.append(('wing_vlm_mesh_out_L_panel', (nt, 29, 1)))


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
                          rpm=[rpm]*len(prop_loc_names),
                          rpm_dir=prop_dir_list, # +x is positive, -x is negative
                          rot_surf_names=prop_blade_names, # list of surface names for blades corresponding to prop in prop_loc_names
                          center_point_names=prop_loc_names,
                        #   use_polar=True,
                        #   polar_bool_list=polar_bool_list,
                          )

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
ave_outputs = average_op.evaluate(outputs) # time averaged qts

fx = ave_outputs[0]
fy = ave_outputs[1]
fz = ave_outputs[2]

offset = 3
if do_wing:
    if symmetry:
        wing_lift_neg_y = ave_outputs[3]
        wing_drag_neg_y = ave_outputs[4]
        wing_lift_pos_y = ave_outputs[5]
        wing_drag_pos_y = ave_outputs[6]
        offset = 7
    else:
        wing_lift = ave_outputs[3]
        wing_drag = ave_outputs[4]
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

# region assembly
# add the cruise m3l model to the cruise condition
wig_condition.add_m3l_model('wig_model', overmodel)
# # add the design condition to the design scenario
design_scenario.add_design_condition(wig_condition)
system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()
# caddee_csdl_model = overmodel.assemble_csdl()

model_csdl = caddee_csdl_model
# endregion

# region connections and DVs

for i in range(len(prop_meshes)):
    i = str(i)

    rpm_csdl = model_csdl.create_input(f'rpm_rotor_{i}', val=rpm)

    model_csdl.connect('p' + i + '_vector', 
                    'system_model.wig.wig.wig.operation.input_model.p' + i + 'b1_mesh_rotor.vector')
    
    model_csdl.connect(f'rpm_rotor_{i}', 
                    f'system_model.wig.wig.wig.torque_operation_rotor_{i}.rpm')
    
    model_csdl.connect(f'rpm_rotor_{i}',
                    f'system_model.wig.wig.wig.engine_{i}_engine.rpm')
    
    model_csdl.connect('system_model.wig.wig.wig.operation.input_model.wig_ac_states_operation.u',
                        f'system_model.wig.wig.wig.torque_operation_rotor_{i}.velocity')

# wing mirror model connections:
# caddee_csdl_model.connect('wing_vlm_mesh', 
#                           'system_model.wig.wig.wig.operation.input_model.wing_vlm_meshmirror.wing_vlm_mesh')
# endregion

# # the pitch angle design variable:
# set_pitch = model_csdl.create_input('set_pitch', val=pitch)
# # model_csdl.add_design_variable('set_pitch', lower=np.deg2rad(0), upper=np.deg2rad(10), scaler=1E2)
# model_csdl.print_var(set_pitch)
# # connect set_pitch to wing mirror:
# model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.wing_vlm_mesh_pos_ymirror.theta')
# model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.wing_vlm_mesh_neg_ymirror.theta')
# # connect set_pitch to fuselage mirrors:
# if do_fuselage:
#     model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.right_fuselage_meshmirror.theta')
#     model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.left_fuselage_meshmirror.theta')

# # connect set_pitch to rotors:
# for i in range(num_props):
#     model_csdl.connect('set_pitch', 'system_model.wig.wig.wig.operation.input_model.p'+str(i)+'b1_mesh_rotor.theta')

sim = Simulator(model_csdl, analytics=True, lazy=1)
sim.run()

if False:
    import pickle
    L_panel_forces = sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.wing_vlm_mesh_out_L_panel']
    prop_forces = {}
    for i in range(4):
        prop_name = f'prop {i}'
        sub_list = []
        for j in range(num_blades):            
            # sub_list.append(sim[f'p{i}b1_mesh_rotor{j}_out_panel_forces_x'])
            sub_list.append(sim[f'system_model.wig.wig.wig.operation.post_processor.ThrustDrag.p{i}b1_mesh_rotor{j}_out_panel_forces_x'])
            # system_model.wig.wig.wig.operation.post_processor.ThrustDrag.p4b1_mesh_rotor1_out_panel_forces_x
        prop_forces[prop_name] = sub_list

    file_1 = open('full_config_lift_data', "wb")
    pickle.dump(L_panel_forces, file_1)
    file_1.close()

    file_2 = open('full_config_thrust_data', "wb")
    pickle.dump(prop_forces, file_2)
    file_2.close()


# import time
# start = time.time()
# sim.run()
# end = time.time()
# print('Total run time:')
# print(end-start)
# print('Fx Prop 1')
# print(sim['system_model.wig.wig.wig.average_op.p0b1_mesh_rotor0_out_panel_forces_x_ave'])
# print(sim['system_model.wig.wig.wig.average_op.p0b1_mesh_rotor1_out_panel_forces_x_ave'])
# print('\n')
# print('Fy Prop 1')
# print(sim['system_model.wig.wig.wig.average_op.p0b1_mesh_rotor0_out_panel_forces_y_ave'])
# print(sim['system_model.wig.wig.wig.average_op.p0b1_mesh_rotor1_out_panel_forces_y_ave'])
# print('\n')
# print('Fz Prop 1')
# print(sim['system_model.wig.wig.wig.average_op.p0b1_mesh_rotor0_out_panel_forces_z_ave'])
# print(sim['system_model.wig.wig.wig.average_op.p0b1_mesh_rotor1_out_panel_forces_z_ave'])

# print('\n')
# print('Wing panel forces x, y, z')
# print(sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.wing_vlm_mesh_out_panel_forces_x'])
# print(sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.wing_vlm_mesh_out_panel_forces_y'])
# print(sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.wing_vlm_mesh_out_panel_forces_z'])
# print('\n')
# L = sim['system_model.wig.wig.wig.average_op.wing_vlm_mesh_out_L']
# D = sim['system_model.wig.wig.wig.average_op.wing_vlm_mesh_out_D']
# print('\n')
# L_ave = sim['system_model.wig.wig.wig.average_op.wing_vlm_mesh_out_L_ave']
# D_ave = sim['system_model.wig.wig.wig.average_op.wing_vlm_mesh_out_D_ave']
# print(L)
# print(D)
# print(L/D)

# print(sim['system_model.wig.wig.wig.torque_operation_rotor_0.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_1.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_2.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_3.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_4.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_5.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_6.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_7.total_thrust'])

'''
GRID CONVERGENCE STUDY NOTES:
for chordwise:
    - spanwise of wing: 41
    - chordwise panels: [1, 2, 5, 10, 15, 30, 45]
'''

plot_wing_distributions = False
if plot_wing_distributions:
    import pickle
    span_coords = sim['system_model.wig.wig.wig.operation.input_model.wing_vlm_meshmirror.wing_vlm_mesh_out'][0][0,:,1]
    span_panels = (span_coords[:-1] + span_coords[1:])/2
    span = span_panels[-1] - span_panels[0]
    # var shape is (num_nodes, (nx-1)*(ny-1) ,1)
    panel_nx, panel_ny = num_chordwise_vlm-1, num_spanwise_vlm-1
    L_panel = sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.wing_vlm_mesh_out_L_panel'][-1].reshape((panel_nx, panel_ny))
    D_i_panel = sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.wing_vlm_mesh_out_D_panel'][-1].reshape((panel_nx, panel_ny))

    C_L = sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.wing_vlm_mesh_out_C_L'][-1]
    C_D_i = sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.wing_vlm_mesh_out_C_D_i'][-1]

    L_spanwise = np.sum(L_panel, axis=0)
    D_i_spanwise = np.sum(D_i_panel, axis=0)

    file_data = [span_panels, L_spanwise, D_i_spanwise, C_L, C_D_i]
    if do_fuselage:
        fuse_nx, fuse_nz = num_long_vlm - 1, num_vert_vlm - 1
        file_name = f'fuse_grid_conv/wing_nc_{int(panel_nx)}_ns_{int(panel_ny)}_fuse_nc_{fuse_nx}_ns_{fuse_nz}.pkl'
    else:
        # file_name = f'fuse_grid_conv/wing_nc_{int(panel_nx)}_ns_{int(panel_ny)}.pkl'
        file_name = f'chord_grid_conv/wing_nc_{int(panel_nx)}_ns_{int(panel_ny)}.pkl'

    open_file = open(file_name, "wb")
    pickle.dump(file_data, open_file)
    open_file.close
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(span_panels/span, L_spanwise, '-*')
    ax1.grid()
    ax1.set_ylabel('Sectional Lift (N)')

    ax2.plot(span_panels/span, D_i_spanwise, '-*')
    ax2.grid()
    ax2.set_xlabel('Normalized Spanwise Location')
    ax2.set_ylabel('Sectional Induced Drag (N)')

    plt.show()



if True:
    plot_wireframe(sim, surface_names, nt, plot_mirror=True, interactive=False, side_view=False, name='wing_fuse_test', backend='ffmpeg', absolute=True)
    # plot_wireframe(sim, surface_names, nt, plot_mirror=True, interactive=False, side_view=True)





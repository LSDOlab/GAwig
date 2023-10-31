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
from plot import plot_wireframe
from engine import Engine
from torque_model import TorqueModel
# endregion

# region hyperparameters
num_props = 2
num_blades = 2
rpm = 1090.
nt = 10
dt = 0.016 * 1
h = 30                       # m
pitch = np.deg2rad(0)        # rad
blade_angle = np.deg2rad(0)  # rad
rotor_delta = [0,0,0]        # m
rotation_point = np.array([0,0,0])
do_wing = True
do_fuselage = True
mirror = True
symmetry = False # only works with mirror = True
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
num_spanwise_vlm = 12 # * 2 + 1
num_spanwise_temp = num_spanwise_vlm+1
num_chordwise_vlm = 8
log_space = False
if log_space:
    start = 0.001
    end = 1

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
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 2.]), direction=np.array([0., 0., -2.]), grid_search_n=30, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 2.]), direction=np.array([0., 0., 2.]), grid_search_n=30, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
# spatial_rep.plot_meshes([wing_camber_surface])
wing_camber_surface_np = wing_camber_surface.value.reshape((num_chordwise_vlm, num_spanwise_vlm, 3))

flap_mesh = deflect_flap(wing_camber_surface_np, 30, 1)
# spatial_rep.plot_meshes([flap_mesh])

wing_vlm_mesh_name = 'wing_vlm_mesh'
wing_camber_surface_np = flap_mesh #.reshape((1, 14, 22, 3))# wing_camber_surface.value # TODO: change this idk



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
num_spanwise_prop= 6
num_chordwise_prop = 2
offsets = [0,20,20,38,18,38,20,20]
p1 = [39.754, -88.35, 4.769]
p2 = [39.848-0.3, -93.75, 4.342-0.5]
p3 = [40.246, -88.35, 5.231]
p4 = [40.152+0.3, -93.75, 5.658+0.5]
p5 = [40., -87., 5.]
p6 = [37., -87., 5.]

if num_props > 0:
    leading_edge = props[0].project(np.linspace(np.array(p1), np.array(p2), num_spanwise_prop), direction=np.array([0., 0, -1.]), grid_search_n=50, plot=False)
    trailing_edge = props[0].project(np.linspace(np.array(p3), np.array(p4), num_spanwise_prop), direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False)
    chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_prop)
    prop0b0_mesh = chord_surface.value

    propb1_mesh_names = []
    prop_vector_names = []
    prop_point_names = []
    prop_meshes_np = [prop0b0_mesh]


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
wig_condition.set_module_input(name='mach_number', val=0.2, dv_flag=True, lower=0.1, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0))
wig_condition.set_module_input(name='flight_path_angle', val=0)
wig_condition.set_module_input(name='roll_angle', val=0)
wig_condition.set_module_input(name='yaw_angle', val=0)
wig_condition.set_module_input(name='wind_angle', val=0)
wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))

ac_states = wig_condition.evaluate_ac_states()
ac_expander = ac_expand(num_nodes=nt)
ac_states_expanded = ac_expander.evaluate(ac_states)
# endregion

# region mirroring

h_m3l = m3l.Variable('h', shape=(1,), value=np.array([h]))
pitch_m3l = m3l.Variable('pitch', shape=(1,), value=np.array([np.deg2rad(pitch)]))

non_rotor_surfaces = []
# wing mirroring
wing_mirror_model = Mirror(component=wing,mesh_name=wing_vlm_mesh_name,nt=nt,ns=num_spanwise_vlm,nc=num_chordwise_vlm,point=rotation_point, mesh=wing_camber_surface_np*0.3048)
wing_mesh_out, wing_mirror_mesh = wing_mirror_model.evaluate(pitch_m3l, h_m3l)
if do_wing:
    non_rotor_surfaces.append(wing_mesh_out)
    if mirror:
        non_rotor_surfaces.append(wing_mirror_mesh)

# right fuselage mirroring
right_fuse_mirror_model = Mirror(component=fuse,mesh_name=right_fuse_mesh_name,nt=nt,ns=num_vert_vlm,nc=num_long_vlm,point=rotation_point, mesh=right_fuse_surface_reordered*0.3048)
right_fuse_mesh_out, right_fuse_mirror_mesh = right_fuse_mirror_model.evaluate(pitch_m3l, h_m3l)

# left fuselage mirroring
left_fuse_mirror_model = Mirror(component=fuse,mesh_name=left_fuse_mesh_name,nt=nt,ns=num_vert_vlm,nc=num_long_vlm,point=rotation_point, mesh=left_fuse_surface_reordered*0.3048)
left_fuse_mesh_out, left_fuse_mirror_mesh = left_fuse_mirror_model.evaluate(pitch_m3l, h_m3l)
if do_fuselage:
    non_rotor_surfaces.append(left_fuse_mesh_out)
    non_rotor_surfaces.append(right_fuse_mesh_out)
    if mirror:
        non_rotor_surfaces.append(left_fuse_mirror_mesh)
        non_rotor_surfaces.append(right_fuse_mirror_mesh)


prop_meshes = []
for i in range(num_props):
    blade_angle_value = np.array([blade_angle])
    rotor_delta_value = np.reshape(np.array([rotor_delta]), (3,))
    if i >= num_props/2:
        blade_angle_value = -1*blade_angle_value
        rotor_delta_value[1] = -1*rotor_delta_value[1]
    blade_angle_m3l = m3l.Variable('blade_angle' + str(i), shape=(1,), value=blade_angle_value)
    delta_m3l = m3l.Variable('delta' + str(i), shape=(3,), value=rotor_delta_value)
    dir = -1
    if i >= num_props/2:
        dir = 1
    prop_model = Rotor3(mesh_name = propb1_mesh_names[i], 
                        num_blades = num_blades, 
                        ns = num_spanwise_prop, 
                        nc = num_chordwise_prop, 
                        nt = nt, 
                        dt = dt, 
                        dir = dir, 
                        r_point = rotation_point, 
                        mesh = prop_meshes_np[i],
                        rpm = rpm,
                        point = prop_points[i])
    prop_mesh_out, mirror_prop_meshes = prop_model.evaluate(h_m3l, pitch_m3l, blade_angle_m3l, delta_m3l)
    if mirror:
        prop_meshes.append(prop_mesh_out + mirror_prop_meshes)
    else:
        prop_meshes.append(prop_mesh_out)

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

    # interactions
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
            symmetry_list.append([int(i*num_blades+j),                                       # left blade
                                int((num_props-1-i)*num_blades + j),                       # right blade
                                int(i*num_blades+j + num_blades/2),                        # left mirror blade
                                int((num_props-1-i)*num_blades + j + num_blades/2)])       # right mirror blade

    # symmetry for wing
    if wing_mesh_out.name in surface_names:
        wing_index = surface_names.index(wing_mesh_out.name)
        wing_mirror_index = surface_names.index(wing_mirror_mesh.name)
        symmetry_list.append([wing_index, wing_mirror_index])

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
    pp_vars.append(('wing_vlm_mesh_out_L', (nt, 1)))
    pp_vars.append(('wing_vlm_mesh_out_D', (nt, 1)))

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
                          sub = True,
                          sub_eval_list = sub_eval_list,
                          sub_induced_list = sub_induced_list,
                        #   symmetry = False,
                        #   sym_struct_list = symmetry_list,
                          free_wake=True,)
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
average_op = LastNAverage(n=5)
ave_outputs = average_op.evaluate(outputs) # time averaged qts

fx = ave_outputs[0]
fy = ave_outputs[1]
fz = ave_outputs[2]

offset = 3
if do_wing:
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

sim = Simulator(model_csdl, analytics=True, lazy=1)
sim.run()

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

print(sim['system_model.wig.wig.wig.torque_operation_rotor_0.total_thrust'])
print(sim['system_model.wig.wig.wig.torque_operation_rotor_1.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_2.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_3.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_4.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_5.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_6.total_thrust'])
# print(sim['system_model.wig.wig.wig.torque_operation_rotor_7.total_thrust'])

if True:
    plot_wireframe(sim, surface_names, nt, plot_mirror=True, interactive=True)





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
from rotor import Rotor2
from expansion_op import ac_expand
from VAST.core.vast_solver_unsteady import VASTSolverUnsteady
from VAST.core.profile_model import gen_profile_output_list, PPSubmodel
from last_n_average import LastNAverage





file_name = 'GAwig/LibertyLifter3.stp'
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=file_name)
spatial_rep.refit_geometry(file_name=file_name)
# spatial_rep.plot(plot_types=['mesh'])



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
num_props = 2
props = [] # we go from 1-indexed to 0-indexed here
for i in range(num_props):
    prop = build_component('prop_'+str(i), ['Prop'+str(i+1),'Hub'+str(i+1)])
    props.append(prop)
#endregion

# region meshes
# wing mesh:
num_spanwise_vlm = 22
num_chordwise_vlm = 14
leading_edge = wing.project(np.linspace(np.array([30, -103, 6]), np.array([30, 103, 6]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([80, -105, 6]), np.array([80, 105, 6]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
# spatial_rep.plot_meshes([chord_surface])
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 2.]), direction=np.array([0., 0., -2.]), grid_search_n=30, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 2.]), direction=np.array([0., 0., 2.]), grid_search_n=30, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
# spatial_rep.plot_meshes([wing_camber_surface])
wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)


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
offsets = [0,20,20,38,18,38,20,20]
p1 = [39.754, -88.35, 4.769]
p2 = [39.848-0.3, -93.75, 4.342-0.5]
p3 = [40.246, -88.35, 5.231]
p4 = [40.152+0.3, -93.75, 5.658+0.5]
p5 = [40., -87., 5.]
p6 = [37., -87., 5.]

propb1_mesh_names = []
prop_vector_names = []
prop_point_names = []

for i in range(num_props):
    offset = offsets[i]
    p1[1] = p1[1] + offset
    p2[1] = p2[1] + offset
    p3[1] = p3[1] + offset
    p4[1] = p4[1] + offset
    p5[1] = p5[1] + offset
    p6[1] = p6[1] + offset
    # prop blade 1 mesh
    leading_edge = props[i].project(np.linspace(np.array(p1), np.array(p2), num_spanwise_prop), direction=np.array([0., 0, -1.]), grid_search_n=50, plot=False)
    trailing_edge = props[i].project(np.linspace(np.array(p3), np.array(p4), num_spanwise_prop), direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False)
    chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_prop)
    # spatial_rep.plot_meshes([chord_surface])
    propb1_mesh_name = 'p'+str(i)+'b1_mesh'
    sys_rep.add_output(propb1_mesh_name, chord_surface)

    # prop hub:
    hub_back, hub_front = props[i].project(np.array(p5)), props[i].project(np.array(p6))
    prop_vec = hub_front - hub_back
    prop_vector_name, prop_point_name = 'p' + str(i) + '_vector', 'p' + str(i) + '_point'
    sys_rep.add_output(prop_vector_name, prop_vec)
    sys_rep.add_output(prop_point_name, hub_back)
    propb1_mesh_names.append(propb1_mesh_name)
    prop_vector_names.append(prop_vector_name)
    prop_point_names.append(prop_point_name)
# endregion

nt = num_nodes = 5

# design scenario
design_scenario = cd.DesignScenario(name='wig')

wig_condition = cd.CruiseCondition(name='wig')
wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
wig_condition.set_module_input(name='altitude', val=0)
wig_condition.set_module_input(name='mach_number', val=0.35, dv_flag=True, lower=0.1, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0), dv_flag=False, lower=np.deg2rad(-10), upper=np.deg2rad(10))
wig_condition.set_module_input(name='flight_path_angle', val=0)
wig_condition.set_module_input(name='roll_angle', val=0)
wig_condition.set_module_input(name='yaw_angle', val=0)
wig_condition.set_module_input(name='wind_angle', val=0)
wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))
ac_states = wig_condition.evaluate_ac_states()
ac_expander = ac_expand(num_nodes=nt)
ac_states_expanded = ac_expander.evaluate(ac_states)

theta = np.deg2rad(0)
h = 20
rotation_point = np.array([0,0,0])

non_rotor_surfaces = []
# wing mirroring
wing_mirror_model = Mirror(component=wing,mesh_name=wing_vlm_mesh_name,nt=nt,ns=num_spanwise_vlm,nc=num_chordwise_vlm,point=rotation_point)
wing_mirror_model.set_module_input('theta', val=theta, dv_flag=False)
wing_mirror_model.set_module_input('h', val=h, dv_flag=False)
wing_mesh_out, wing_mirror_mesh = wing_mirror_model.evaluate()
non_rotor_surfaces.append(wing_mesh_out)
non_rotor_surfaces.append(wing_mirror_mesh)

# right fuselage mirroring
right_fuse_mirror_model = Mirror(component=fuse,mesh_name=right_fuse_mesh_name,nt=nt,ns=num_vert_vlm,nc=num_long_vlm,point=rotation_point, mesh=right_fuse_surface_reordered)
right_fuse_mirror_model.set_module_input('theta', val=theta, dv_flag=False)
right_fuse_mirror_model.set_module_input('h', val=h, dv_flag=False)
right_fuse_mesh_out, right_fuse_mirror_mesh = right_fuse_mirror_model.evaluate()
non_rotor_surfaces.append(right_fuse_mesh_out)
non_rotor_surfaces.append(right_fuse_mirror_mesh)

# left fuselage mirroring
left_fuse_mirror_model = Mirror(component=fuse,mesh_name=left_fuse_mesh_name,nt=nt,ns=num_vert_vlm,nc=num_long_vlm,point=rotation_point, mesh=left_fuse_surface_reordered)
left_fuse_mirror_model.set_module_input('theta', val=theta, dv_flag=False)
left_fuse_mirror_model.set_module_input('h', val=h, dv_flag=False)
left_fuse_mesh_out, left_fuse_mirror_mesh = left_fuse_mirror_model.evaluate()
non_rotor_surfaces.append(left_fuse_mesh_out)
non_rotor_surfaces.append(left_fuse_mirror_mesh)

dt = 0.016*2
num_blades = 6
prop_meshes = []
for i in range(num_props):
    dir = -1
    if i > num_blades/2:
        dir = 1
    prop_model = Rotor2(component=props[i], mesh_name=propb1_mesh_names[i], num_blades=num_blades, ns=num_spanwise_prop, nc=num_chordwise_prop, nt=nt, dt=dt, dir=dir, r_point=rotation_point)
    prop_model.set_module_input('rpm', val=1000, dv_flag=True)
    prop_model.set_module_input('theta', val=theta, dv_flag=True)
    prop_model.set_module_input('h', val=h, dv_flag=False)
    prop_mesh_out, mirror_prop_meshes = prop_model.evaluate()
    prop_meshes.append(prop_mesh_out + mirror_prop_meshes)
num_blades = num_blades*2

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
        uvlm_parameters.append((name+'_coll_vel', True, np.zeros((nt, nx-1, ny-1, 3))))
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
    uvlm_parameters.append((surface.name+'_coll_vel', True, np.zeros((nt, num_chordwise-1, num_spanwise-1, 3))))
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

pp_vars = []
# for name in surface_names:
#     pp_vars.append((name+'_L', (nt, 1)))

num_panels = int((num_props*num_blades/2*(num_spanwise_prop-1)*(num_chordwise_prop-1) + (num_spanwise_vlm-1)*(num_chordwise_vlm-1) + (num_long_vlm-1)*(num_vert_vlm-1)*2)*2)
print(num_panels)
pp_vars.append(('panel_forces_x',(nt,num_panels,1)))
pp_vars.append(('panel_forces_y',(nt,num_panels,1)))
pp_vars.append(('panel_forces_z',(nt,num_panels,1)))

profile_outputs = gen_profile_output_list(surface_names, surface_shapes)
ode_surface_shapes = [(num_nodes, ) + item for item in surface_shapes]
post_processor = PPSubmodel(surface_names = surface_names, ode_surface_shapes=ode_surface_shapes, delta_t=dt, nt=num_nodes+1, symmetry=False)

model = m3l.DynamicModel()
uvlm = VASTSolverUnsteady(num_nodes = num_nodes, 
                          surface_names = surface_names, 
                          surface_shapes = surface_shapes, 
                          delta_t = dt, 
                          nt = nt+1,
                          sub = True,
                          sub_eval_list = sub_eval_list,
                          sub_induced_list = sub_induced_list)
uvlm_residual = uvlm.evaluate()
model.register_output(uvlm_residual)
model.set_dynamic_options(initial_conditions=initial_conditions,
                          num_times=num_nodes,
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

average_op = LastNAverage(n=2)
ave_outputs = average_op.evaluate(outputs)

overmodel = m3l.Model()
for var in ave_outputs:
    overmodel.register_output(var)

# add the cruise m3l model to the cruise condition
wig_condition.add_m3l_model('wig_model', overmodel)
# add the design condition to the design scenario
design_scenario.add_design_condition(wig_condition)
system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()

model_csdl = caddee_csdl_model

for i in range(len(prop_meshes)):
    i = str(i)
    model_csdl.connect('p' + i + 'b1_mesh', 
                    'system_model.wig.wig.wig.operation.input_model.p' + i + 'b1_mesh_rotor.p' + i + 'b1_mesh')
    model_csdl.connect('p' + i + '_vector', 
                    'system_model.wig.wig.wig.operation.input_model.p' + i + 'b1_mesh_rotor.vector')
    model_csdl.connect('p' + i + '_point', 
                    'system_model.wig.wig.wig.operation.input_model.p' + i + 'b1_mesh_rotor.point')
    
# wing mirror model connections - can get rid of this if we input value to mirroring function:
caddee_csdl_model.connect('wing_vlm_mesh', 
                          'system_model.wig.wig.wig.operation.input_model.wing_vlm_meshmirror.wing_vlm_mesh')


sim = Simulator(model_csdl, analytics=True, lazy=1)
sim.run()

# import time
# start = time.time()
# sim.run()
# end = time.time()
# print('Total run time:')
# print(end-start)



if True:
    from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
    axs = Axes(
        xrange=(0, 80),
        yrange=(-100, 100),
        zrange=(-5, 10),
    )
    video = Video("rotor_test.gif", fps=10, backend='imageio')
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
            if 'rotor' in surface_name:
                vps = Points(np.reshape(sim['system_model.wig.wig.wig.operation.input_model.'+surface_name[0:9]+'_rotor.' + surface_name][i, :, :, :], (-1, 3)),
                            r=8,
                            c='red')
                vp += vps
                vp += __doc__
                vps = Points(np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_wake_coords'][i, 0:i, :, :],
                                        (-1, 3)),
                            r=8,
                            c='blue')
                vp += vps
                vp += __doc__
            else:
                # system_model.wig.wig.wig.operation.input_model.wing_vlm_meshmirror.wing_vlm_mesh_out
                if 'wing' in surface_name:
                    vps = Points(np.reshape(sim['system_model.wig.wig.wig.operation.input_model.wing_vlm_meshmirror.' + surface_name][i, :, :, :], (-1, 3)),
                                r=8,
                                c='red')
                elif 'right' in surface_name:
                    vps = Points(np.reshape(sim['system_model.wig.wig.wig.operation.input_model.right_fuselage_meshmirror.' + surface_name][i, :, :, :], (-1, 3)),
                                r=8,
                                c='red')
                elif 'left' in surface_name:
                    vps = Points(np.reshape(sim['system_model.wig.wig.wig.operation.input_model.left_fuselage_meshmirror.' + surface_name][i, :, :, :], (-1, 3)),
                                r=8,
                                c='red')
                vp += vps
                vp += __doc__
                vps = Points(np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_wake_coords'][i, 0:i, :, :],
                                        (-1, 3)),
                            r=8,
                            c='blue')
                vp += vps
                vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
        #         axes=False, interactive=False)  # render the scene
        # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
        #         axes=False, interactive=False, zoom=True)  # render the scene
        vp.show(axs, elevation=-45, azimuth=-45, roll=45,
                axes=False, interactive=True)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.close_window()
    vp.close_window()
    video.close()  # merge all the recorded frames


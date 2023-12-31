import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
import array_mapper as am
import lsdo_geo as lg
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
import csdl
from mirror import Mirror
from old.rotate import Rotate
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from mpl_toolkits.mplot3d import proj3d




file_name = 'GAwig/LibertyLifter2.stp'
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=file_name)
spatial_rep.refit_geometry(file_name=file_name)
# spatial_rep.plot(plot_types=['mesh'])




# wing:
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['WingGeom']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
sys_rep.add_component(wing)
# wing.plot()

# htail:
htail_primitive_names = list(spatial_rep.get_primitives(search_names=['HTail']).keys())
htail = LiftingSurface(name='htail', spatial_representation=spatial_rep, primitive_names=htail_primitive_names)
sys_rep.add_component(htail)
# htail.plot()

# fuse:
fuse_primitive_names = list(spatial_rep.get_primitives(search_names=['FuselageGeom']).keys())
fuse = LiftingSurface(name='fuse', spatial_representation=spatial_rep, primitive_names=fuse_primitive_names)
sys_rep.add_component(fuse)
# fuse.plot()

# prop 1:
prop_1_primitive_names = list(spatial_rep.get_primitives(search_names=['Prop1','Hub1']).keys())
prop_1 = LiftingSurface(name='prop_1', spatial_representation=spatial_rep, primitive_names=prop_1_primitive_names)
sys_rep.add_component(prop_1)
# prop_1.plot()




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
rtop = fuse.project(np.linspace(np.array([0, 27, -0.25]), np.array([120, 27, 9]), num_long_vlm), direction=np.array([0., 0., -1.]), plot=False)
rbot = fuse.project(np.linspace(np.array([0, 27, -10]), np.array([120, 27, -2]), num_long_vlm), direction=np.array([0., 0., -1.]), plot=False)
right_fuse_surface = am.linspace(rtop, rbot, num_vert_vlm)
# spatial_rep.plot_meshes([right_fuse_surface])

# left fuselage mesh:
ltop = fuse.project(np.linspace(np.array([0, -27, -0.25]), np.array([120, -27, 9]), num_long_vlm), direction=np.array([0., 0., -1.]), plot=False)
lbot = fuse.project(np.linspace(np.array([0, -27, -10]), np.array([120, -27, -2]), num_long_vlm), direction=np.array([0., 0., -1.]), plot=False)
left_fuse_surface = am.linspace(ltop, lbot, num_vert_vlm)
# spatial_rep.plot_meshes([left_fuse_surface])


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




# prop 1 mesh:
num_spanwise_prop= 6
num_chordwise_prop = 2
prop_1_leading_edge = prop_1.project(np.linspace(np.array([39.803, -88.35, 5.185]), np.array([39.901, -93.75, 6.528]), num_spanwise_prop), direction=np.array([0., 0, -1.]), grid_search_n=50, plot=False)
prop_1_trailing_edge = prop_1.project(np.linspace(np.array([40.197, -88.35, 4.815]), np.array([40.171, -93.259, 4.347]), num_spanwise_prop), direction=np.array([0., 0., -1.]), plot=False)
prop_1_chord_surface = am.linspace(prop_1_leading_edge, prop_1_trailing_edge, num_chordwise_prop)

spatial_rep.plot_meshes([prop_1_chord_surface])

# Configuration creations and prescribed actuations
configuration_names = ["cruise_configuration"]
system_configurations = sys_rep.declare_configurations(names=configuration_names)
cruise_configuration = system_configurations['cruise_configuration']

n = 20
dt = 0.1

cruise_configuration.set_num_nodes(num_nodes=n)
cruise_configuration.add_output('prop_1_chord_surface', prop_1_chord_surface)

hub_back = prop_1.project(np.array([28.5, -10., 8.]))
hub_front = prop_1.project(np.array([28.5, 10., 8.]))
prop1_actuation_axis = hub_front - hub_back
from caddee.core.caddee_core.system_representation.prescribed_actuations import PrescribedRotation
prop1_actuator_solver = PrescribedRotation(component=prop_1, axis_origin=hub_front, axis_vector=prop1_actuation_axis)
prop1_actuation_profile = np.linspace(0., -1., n)
prop1_actuator_solver.set_rotation(name='cruise_prop_actuation', value=np.zeros((n,)), units='radians')
cruise_configuration.actuate(transformation=prop1_actuator_solver)








sys_param.setup()


# design scenario
my_big_wig_model = ModuleCSDL()

design_scenario = cd.DesignScenario(name='wig')
wig_model = m3l.Model()
wig_condition = cd.CruiseCondition(name='wig')
wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
wig_condition.set_module_input(name='altitude', val=0)
wig_condition.set_module_input(name='mach_number', val=0.21623, dv_flag=True, lower=0.1, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0), dv_flag=False, lower=np.deg2rad(-10), upper=np.deg2rad(10))
wig_condition.set_module_input(name='flight_path_angle', val=0)
wig_condition.set_module_input(name='roll_angle', val=0)
wig_condition.set_module_input(name='yaw_angle', val=0)
wig_condition.set_module_input(name='wind_angle', val=0)
wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))
ac_states = wig_condition.evaluate_ac_states()
wig_model.register_output(ac_states)




prop_1_rotation_model = Rotate(n=n,dt=dt)



wig_model_csdl = wig_model.assemble()




my_big_wig_model.add(prop_1_rotation_model, name='rotation_model')
my_big_wig_model.add_module(sys_param.assemble_csdl(), name='system_paramterization')
my_big_wig_model.add_module(sys_rep.assemble_csdl(), name='system_representation')
my_big_wig_model.add_module(wig_model_csdl, name='m3l_model')

my_big_wig_model.connect('angles', 'cruise_prop_actuation')




my_big_wig_model.create_input('rpm', val=1000)



sim = Simulator(my_big_wig_model, analytics=True)
sim.run()





for t in range(n):
    updated_primitives_names = list(spatial_rep.primitives.keys()).copy()
    cruise_geometry = sim['cruise_configuration_geometry'][t]
    # cruise_geometry = sim['design_geometry']
    spatial_rep.update(cruise_geometry, updated_primitives_names)
    prop_1_chord_surface.evaluate(spatial_rep.control_points['geometry'])
    prop_1_chord_surface_csdl = sim['prop_1_chord_surface'][t]
    spatial_rep.plot_meshes([prop_1_chord_surface_csdl], mesh_plot_types=['wireframe'], mesh_opacity=1.)
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
from rotate import Rotate
from mpl_toolkits.mplot3d import proj3d


 

file_name = 'GAwig/LibertyLifter2.stp'
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=file_name)
spatial_rep.refit_geometry(file_name=file_name)
spatial_rep.plot(plot_types=['mesh'])




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
num_chordwise_prop = 1
prop_1_leading_edge = prop_1.project(np.linspace(np.array([39.803, -88.35, 5.185]), np.array([39.901, -93.75, 5.428]), num_spanwise_prop), direction=np.array([0., 0., -1.]), plot=True)
exit()
prop_1_trailing_edge = prop_1.project(np.linspace(np.array([80, -105, 6]), np.array([80, 105, 6]), num_spanwise_prop), direction=np.array([0., 0., -1.]), plot=True)
chord_surface = am.linspace(prop_1_leading_edge, prop_1_trailing_edge, num_chordwise_prop)













sys_param.setup()


# design scenario
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






n = 20
dt = 0.1


# create angular time-histories for rotors:
prop_1_model = Rotate(component=prop_1, n=n, dt=dt)
prop_1_model.set_module_input('rpm', val=1000, dv_flag=True)
prop_1_angles = prop_1_model.evaluate()
wig_model.register_output(prop_1_angles)









# add the cruise m3l model to the cruise condition
wig_condition.add_m3l_model('wig_model', wig_model)
# add the design condition to the design scenario
design_scenario.add_design_condition(wig_condition)
system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()






sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()
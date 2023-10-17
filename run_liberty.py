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
from rotor import Rotor
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from mpl_toolkits.mplot3d import proj3d
from caddee.core.caddee_core.system_representation.prescribed_actuations import PrescribedRotation



file_name = 'GAwig/LibertyLifter3.stp'
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

# prop 2:
prop_2_primitive_names = list(spatial_rep.get_primitives(search_names=['Prop2','Hub2']).keys())
prop_2 = LiftingSurface(name='prop_2', spatial_representation=spatial_rep, primitive_names=prop_2_primitive_names)
sys_rep.add_component(prop_2)
# prop_2.plot()

# prop 3:
prop_3_primitive_names = list(spatial_rep.get_primitives(search_names=['Prop3','Hub3']).keys())
prop_3 = LiftingSurface(name='prop_3', spatial_representation=spatial_rep, primitive_names=prop_3_primitive_names)
sys_rep.add_component(prop_3)
# prop_3.plot()

# prop 4:
prop_4_primitive_names = list(spatial_rep.get_primitives(search_names=['Prop4','Hub4']).keys())
prop_4 = LiftingSurface(name='prop_4', spatial_representation=spatial_rep, primitive_names=prop_4_primitive_names)
sys_rep.add_component(prop_4)
# prop_4.plot()

# prop 5:
prop_5_primitive_names = list(spatial_rep.get_primitives(search_names=['Prop5','Hub5']).keys())
prop_5 = LiftingSurface(name='prop_5', spatial_representation=spatial_rep, primitive_names=prop_5_primitive_names)
sys_rep.add_component(prop_5)
# prop_5.plot()

# prop 6:
prop_6_primitive_names = list(spatial_rep.get_primitives(search_names=['Prop6','Hub6']).keys())
prop_6 = LiftingSurface(name='prop_6', spatial_representation=spatial_rep, primitive_names=prop_6_primitive_names)
sys_rep.add_component(prop_6)
# prop_6.plot()

# prop 7:
prop_7_primitive_names = list(spatial_rep.get_primitives(search_names=['Prop7','Hub7']).keys())
prop_7 = LiftingSurface(name='prop_7', spatial_representation=spatial_rep, primitive_names=prop_7_primitive_names)
sys_rep.add_component(prop_7)
# prop_7.plot()

# prop 8:
prop_8_primitive_names = list(spatial_rep.get_primitives(search_names=['Prop8','Hub8']).keys())
prop_8 = LiftingSurface(name='prop_8', spatial_representation=spatial_rep, primitive_names=prop_8_primitive_names)
sys_rep.add_component(prop_8)
# prop_8.plot()











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


# prop 1 blade 1 mesh:
num_spanwise_prop= 6
num_chordwise_prop = 2
p1b1_leading_edge = prop_1.project(np.linspace(np.array([39.754, -88.35, 4.769]), np.array([39.848-0.3, -93.75, 4.342-0.5]), num_spanwise_prop), direction=np.array([0., 0, -1.]), grid_search_n=50, plot=False)
p1b1_trailing_edge = prop_1.project(np.linspace(np.array([40.246, -88.35, 5.231]), np.array([40.152+0.3, -93.75, 5.658+0.5]), num_spanwise_prop), direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False)
p1b1_chord_surface = am.linspace(p1b1_leading_edge, p1b1_trailing_edge, num_chordwise_prop)
# spatial_rep.plot_meshes([p1b1_chord_surface])
p1b1_mesh_name = 'p1b1_mesh'
sys_rep.add_output(p1b1_mesh_name, p1b1_chord_surface)

# prop 1 hub:
hub_back, hub_front = prop_1.project(np.array([40., -87., 5.])), prop_1.project(np.array([37., -87., 5.]))
prop1_vec = hub_front - hub_back
p1_vector_name, p1_point_name = 'p1_vector', 'p1_point'
sys_rep.add_output(p1_vector_name, prop1_vec)
sys_rep.add_output(p1_point_name, hub_back)


# prop 2 blade 1 mesh:
p2b1_leading_edge = prop_2.project(np.linspace(np.array([39.754, -88.35+20, 4.769]), np.array([39.848-0.3, -93.75+20, 4.342-0.5]), num_spanwise_prop), direction=np.array([0., 0, -1.]), grid_search_n=50, plot=False)
p2b1_trailing_edge = prop_2.project(np.linspace(np.array([40.246, -88.35+20, 5.231]), np.array([40.152+0.3, -93.75+20, 5.658+0.5]), num_spanwise_prop), direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False)
p2b1_chord_surface = am.linspace(p2b1_leading_edge, p2b1_trailing_edge, num_chordwise_prop)
# spatial_rep.plot_meshes([p2b1_chord_surface])
p2b1_mesh_name = 'p2b1_mesh'
sys_rep.add_output(p2b1_mesh_name, p2b1_chord_surface)

# prop 2 hub:
p2_hub_back, p2_hub_front = prop_2.project(np.array([40., -87.+20, 5.])), prop_2.project(np.array([37., -87.+20, 5.]))
prop2_vec = p2_hub_front - p2_hub_back
p2_vector_name, p2_point_name = 'p2_vector', 'p2_point'
sys_rep.add_output(p2_vector_name, prop2_vec)
sys_rep.add_output(p2_point_name, p2_hub_back)


# prop 3 blade 1 mesh:
p3b1_leading_edge = prop_3.project(np.linspace(np.array([39.754, -88.35+40, 4.769]), np.array([39.848-0.3, -93.75+40, 4.342-0.5]), num_spanwise_prop), direction=np.array([0., 0, -1.]), grid_search_n=50, plot=False)
p3b1_trailing_edge = prop_3.project(np.linspace(np.array([40.246, -88.35+40, 5.231]), np.array([40.152+0.3, -93.75+40, 5.658+0.5]), num_spanwise_prop), direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False)
p3b1_chord_surface = am.linspace(p3b1_leading_edge, p3b1_trailing_edge, num_chordwise_prop)
# spatial_rep.plot_meshes([p3b1_chord_surface])
p3b1_mesh_name = 'p3b1_mesh'
sys_rep.add_output(p3b1_mesh_name, p3b1_chord_surface)

# prop 3 hub:
p3_hub_back, p3_hub_front = prop_3.project(np.array([40., -87.+40, 5.])), prop_3.project(np.array([37., -87.+40, 5.]))
prop3_vec = p3_hub_front - p3_hub_back
p3_vector_name, p3_point_name = 'p3_vector', 'p3_point'
sys_rep.add_output(p3_vector_name, prop3_vec)
sys_rep.add_output(p3_point_name, p3_hub_back)


# prop 4 blade 1 mesh:
p4b1_leading_edge = prop_4.project(np.linspace(np.array([39.754, -88.35+40+38, 4.769]), np.array([39.848-0.3, -93.75+40+38, 4.342-0.5]), num_spanwise_prop), direction=np.array([0., 0, -1.]), grid_search_n=50, plot=False)
p4b1_trailing_edge = prop_4.project(np.linspace(np.array([40.246, -88.35+40+38, 5.231]), np.array([40.152+0.3, -93.75+40+38, 5.658+0.5]), num_spanwise_prop), direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False)
p4b1_chord_surface = am.linspace(p4b1_leading_edge, p4b1_trailing_edge, num_chordwise_prop)
# spatial_rep.plot_meshes([p4b1_chord_surface])
p4b1_mesh_name = 'p4b1_mesh'
sys_rep.add_output(p4b1_mesh_name, p4b1_chord_surface)

# prop 3 hub:
p4_hub_back, p4_hub_front = prop_4.project(np.array([40., -87.+40+38, 5.])), prop_4.project(np.array([37., -87.+40+38, 5.]))
prop4_vec = p4_hub_front - p4_hub_back
p4_vector_name, p4_point_name = 'p4_vector', 'p4_point'
sys_rep.add_output(p4_vector_name, prop4_vec)
sys_rep.add_output(p4_point_name, p4_hub_back)








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



nt = 4
dt = 0.001
num_blades = 6
prop_1_model = Rotor(component=prop_1, mesh_name=p1b1_mesh_name, num_blades=num_blades, ns=num_spanwise_prop, nc=num_chordwise_prop, nt=nt, dt=dt, dir=-1)
prop_1_model.set_module_input('rpm', val=1000, dv_flag=True)
prop_1_mesh = prop_1_model.evaluate()
wig_model.register_output(prop_1_mesh)

prop_2_model = Rotor(component=prop_2, mesh_name=p2b1_mesh_name, num_blades=num_blades, ns=num_spanwise_prop, nc=num_chordwise_prop, nt=nt, dt=dt, dir=-1)
prop_2_model.set_module_input('rpm', val=1000, dv_flag=True)
prop_2_mesh = prop_2_model.evaluate()
wig_model.register_output(prop_2_mesh)

prop_3_model = Rotor(component=prop_3, mesh_name=p3b1_mesh_name, num_blades=num_blades, ns=num_spanwise_prop, nc=num_chordwise_prop, nt=nt, dt=dt, dir=-1)
prop_3_model.set_module_input('rpm', val=1000, dv_flag=True)
prop_3_mesh = prop_3_model.evaluate()
wig_model.register_output(prop_3_mesh)

prop_4_model = Rotor(component=prop_4, mesh_name=p4b1_mesh_name, num_blades=num_blades, ns=num_spanwise_prop, nc=num_chordwise_prop, nt=nt, dt=dt, dir=-1)
prop_4_model.set_module_input('rpm', val=1000, dv_flag=True)
prop_4_mesh = prop_4_model.evaluate()
wig_model.register_output(prop_4_mesh)



# add the cruise m3l model to the cruise condition
wig_condition.add_m3l_model('wig_model', wig_model)
# add the design condition to the design scenario
design_scenario.add_design_condition(wig_condition)
system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()



caddee_csdl_model.connect('p1b1_mesh', 
                          'system_model.wig.wig.wig.p1b1_mesh_rotor.p1b1_mesh')

caddee_csdl_model.connect('p1_vector', 
                          'system_model.wig.wig.wig.p1b1_mesh_rotor.vector')

caddee_csdl_model.connect('p1_point', 
                          'system_model.wig.wig.wig.p1b1_mesh_rotor.point')

caddee_csdl_model.connect('p2b1_mesh', 
                          'system_model.wig.wig.wig.p2b1_mesh_rotor.p2b1_mesh')

caddee_csdl_model.connect('p2_vector', 
                          'system_model.wig.wig.wig.p2b1_mesh_rotor.vector')

caddee_csdl_model.connect('p2_point', 
                          'system_model.wig.wig.wig.p2b1_mesh_rotor.point')

caddee_csdl_model.connect('p3b1_mesh', 
                          'system_model.wig.wig.wig.p3b1_mesh_rotor.p3b1_mesh')

caddee_csdl_model.connect('p3_vector', 
                          'system_model.wig.wig.wig.p3b1_mesh_rotor.vector')

caddee_csdl_model.connect('p3_point', 
                          'system_model.wig.wig.wig.p3b1_mesh_rotor.point')

caddee_csdl_model.connect('p4b1_mesh', 
                          'system_model.wig.wig.wig.p4b1_mesh_rotor.p4b1_mesh')

caddee_csdl_model.connect('p4_vector', 
                          'system_model.wig.wig.wig.p4b1_mesh_rotor.vector')

caddee_csdl_model.connect('p4_point', 
                          'system_model.wig.wig.wig.p4b1_mesh_rotor.point')



sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()





# plot the meshes to see if stuff is working:
original = sim['p1b1_mesh']
p1_mesh = sim['system_model.wig.wig.wig.p1b1_mesh_rotor.rotor']
p2_mesh = sim['system_model.wig.wig.wig.p2b1_mesh_rotor.rotor']
p3_mesh = sim['system_model.wig.wig.wig.p3b1_mesh_rotor.rotor']
p4_mesh = sim['system_model.wig.wig.wig.p4b1_mesh_rotor.rotor']




fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(original[:,:,0].flatten(), original[:,:,1].flatten(), original[:,:,2].flatten(), color='orange')

for i in range(num_blades):
    for j in range(nt):
        #ax.plot_trisurf(p1_mesh[i,j,:,:,0].flatten(), p1_mesh[i,j,:,:,1].flatten(), p1_mesh[i,j,:,:,2].flatten(), color='blue')
        ax.plot_trisurf(p1_mesh[i,j,:,:,0].flatten(), p1_mesh[i,j,:,:,1].flatten(), p1_mesh[i,j,:,:,2].flatten())
        ax.plot_trisurf(p2_mesh[i,j,:,:,0].flatten(), p2_mesh[i,j,:,:,1].flatten(), p2_mesh[i,j,:,:,2].flatten())
        ax.plot_trisurf(p3_mesh[i,j,:,:,0].flatten(), p3_mesh[i,j,:,:,1].flatten(), p3_mesh[i,j,:,:,2].flatten())
        ax.plot_trisurf(p4_mesh[i,j,:,:,0].flatten(), p4_mesh[i,j,:,:,1].flatten(), p4_mesh[i,j,:,:,2].flatten())

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

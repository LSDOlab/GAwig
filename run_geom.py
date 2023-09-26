import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
import array_mapper as am
import lsdo_geo as lg
from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces
import numpy as np
import sys
sys.setrecursionlimit(100000)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import csdl




file_name = 'GAwig/amphib2.stp'

caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)

spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=file_name)
spatial_rep.refit_geometry(file_name=file_name)
#spatial_rep.plot(plot_types=['mesh'])




# wing
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['winggeom']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
sys_rep.add_component(wing)
#wing.plot()



# wing mesh
num_spanwise_vlm = 18
num_chordwise_vlm = 5
offset = 3
leading_edge = wing.project(np.linspace(np.array([12.01 - offset, -20.193, 2.051]), np.array([12.01 - offset, 20.193, 2.051]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([14.508 + offset, -20.593, 1.963]), np.array([14.508 + offset, 20.593, 1.963]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
#spatial_rep.plot_meshes([chord_surface])


wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=30, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=30, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1
#spatial_rep.plot_meshes([wing_camber_surface])


wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)
wing_oml_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
wing_oml_mesh_name = 'wing_oml_mesh'
sys_rep.add_output(wing_oml_mesh_name, wing_oml_mesh)





# design scenario
design_scenario = cd.DesignScenario(name='cruise')
cruise_model = m3l.Model()
cruise_condition = cd.CruiseCondition(name='cruise')
cruise_condition.atmosphere_model = cd.SimpleAtmosphereModel()
cruise_condition.set_module_input(name='altitude', val=0)
cruise_condition.set_module_input(name='mach_number', val=0.15, dv_flag=True, lower=0.1, upper=0.3)
cruise_condition.set_module_input(name='range', val=1000)
cruise_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0), dv_flag=True, lower=np.deg2rad(-10), upper=np.deg2rad(10))
cruise_condition.set_module_input(name='flight_path_angle', val=0)
cruise_condition.set_module_input(name='roll_angle', val=0)
cruise_condition.set_module_input(name='yaw_angle', val=0)
cruise_condition.set_module_input(name='wind_angle', val=0)
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))
ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(ac_states)







# VLM solver
vlm_model = VASTFluidSover(
    surface_names=[wing_vlm_mesh_name,],
    surface_shapes=[(1, ) + wing_camber_surface.evaluate().shape[1:],],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='ft',
    cl0=[0.25]
)
vlm_panel_forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=ac_states)
cruise_model.register_output(vlm_forces)
cruise_model.register_output(vlm_moments)

# VLM force mapping model
vlm_force_mapping_model = VASTNodalForces(
    surface_names=[wing_vlm_mesh_name,],
    surface_shapes=[(1, ) + wing_camber_surface.evaluate().shape[1:],],
    initial_meshes=[wing_camber_surface,]
)

oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=vlm_panel_forces, nodal_force_meshes=[wing_oml_mesh, wing_oml_mesh])
wing_forces = oml_forces[0]





# add the cruise m3l model to the cruise condition
cruise_condition.add_m3l_model('cruise_model', cruise_model)
# add the design condition to the design scenario
design_scenario.add_design_condition(cruise_condition)
system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()



sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()



vlm_total_forces = sim['system_model.cruise.cruise.cruise.wing_vlm_mesh_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_total_forces']
print(vlm_total_forces[:,:,2])


plt.plot(vlm_total_forces[:,:,2].flatten())
plt.show()



#print(sim[])
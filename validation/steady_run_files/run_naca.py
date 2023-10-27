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

from mpl_toolkits.mplot3d import proj3d


file_name = 'naca6409.stp'
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=file_name)
spatial_rep.refit_geometry(file_name=file_name)
# spatial_rep.plot(plot_types=['mesh'])

# wing
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['WingGeom']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
sys_rep.add_component(wing)
# wing.plot()

# wing mesh
num_spanwise_vlm = 25
num_chordwise_vlm = 25
leading_edge = wing.project(np.linspace(np.array([0., -.2, 0.01]), np.array([0., .2, 0.01]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([0.2, -.2, 0.]), np.array([0.2, .2, 0.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
# spatial_rep.plot_meshes([chord_surface])


wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 0.03]), direction=np.array([0., 0., -1.]), grid_search_n=30, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value, direction=np.array([0., 0., 1.]), grid_search_n=50, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
spatial_rep.plot_meshes([wing_camber_surface])
wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)
exit()

sys_param.setup()

# design scenario
design_scenario = cd.DesignScenario(name='wig')
wig_model = m3l.Model()
wig_condition = cd.CruiseCondition(name='wig')
wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
wig_condition.set_module_input(name='altitude', val=0)
wig_condition.set_module_input(name='mach_number', val=25.5/340., dv_flag=True, lower=0.1, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0), dv_flag=False, lower=np.deg2rad(-10), upper=np.deg2rad(10))
wig_condition.set_module_input(name='flight_path_angle', val=0)
wig_condition.set_module_input(name='roll_angle', val=0)
wig_condition.set_module_input(name='yaw_angle', val=0)
wig_condition.set_module_input(name='wind_angle', val=0)
wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))
ac_states = wig_condition.evaluate_ac_states()
wig_model.register_output(ac_states)




# create a mirrored mesh
mirror = Mirror(component=wing,mesh_name=wing_vlm_mesh_name,ns=num_spanwise_vlm,nc=num_chordwise_vlm,point=np.array([0.508, 0, 0]))
mirror.set_module_input('alpha', val=np.deg2rad(0.), dv_flag=False)
mirror.set_module_input('h', val=10., dv_flag=False)
mesh_out, mirror_mesh = mirror.evaluate()
wig_model.register_output(mirror_mesh)
wig_model.register_output(mesh_out)




# VLM solver
vlm_model = VASTFluidSover(
    surface_names=[wing_vlm_mesh_name+'_out', wing_vlm_mesh_name+'_mirror',],
    surface_shapes=[(1, ) + wing_camber_surface.evaluate().shape[1:],
                    (1, ) + wing_camber_surface.evaluate().shape[1:],],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='m',
    cl0=[0., -0.]
)
vlm_panel_forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=ac_states)
wig_model.register_output(vlm_forces)
wig_model.register_output(vlm_moments)


 

# add the cruise m3l model to the cruise condition
wig_condition.add_m3l_model('wig_model', wig_model)
# add the design condition to the design scenario
design_scenario.add_design_condition(wig_condition)
system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()




# connect the transformed wing meshes to VAST:
caddee_csdl_model.connect('system_model.wig.wig.wig.wing_vlm_meshmirror.wing_vlm_mesh_mirror',
                          'system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_system.MeshPreprocessing_comp.wing_vlm_mesh_mirror')

caddee_csdl_model.connect('system_model.wig.wig.wig.wing_vlm_meshmirror.wing_vlm_mesh_out',
                          'system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_system.MeshPreprocessing_comp.wing_vlm_mesh_out')

# connect altitude to the mirror:
#caddee_csdl_model.connect('system_model.wig.wig.wig.wig_ac_states_operation.wig_altitude',
#                          'system_model.wig.wig.wig.mirror.h')

 

sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()


L = sim['system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_out_L']
C_L = sim['system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_out_C_L']
C_D_i = sim['system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_out_C_D_i']
C_D = sim['system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_out_C_D_total']
h = sim['system_model.wig.wig.wig.wing_vlm_meshmirror.h']
b = 0.4 # wing span
S = 0.08

print('h/b: ', h/b)
print('h/sqrt(S): ', h/np.sqrt(S))
print('C_L: ', C_L)
print('C_D_i: ', C_D_i)
print('C_D: ', C_D)
print('L/D: ', C_L/C_D)
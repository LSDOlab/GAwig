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


"""
Re = 2.55E6
mu = 1.4207E-5
rho = 1.225
L = 0.508
a = 340.3
M = Re*mu/(rho*L*a)
print(M)
"""

 

#file_name = 'GAwig/clarky.stp'
file_name = 'clarky_endplate_t0d2.stp'
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

# left endplate
left_plate_primitive_names = list(spatial_rep.get_primitives(search_names=['leftplate']).keys())
left_plate = LiftingSurface(name='left_plate', spatial_representation=spatial_rep, primitive_names=left_plate_primitive_names)
sys_rep.add_component(left_plate)
# left_plate.plot()

# right endplate
right_plate_primitive_names = list(spatial_rep.get_primitives(search_names=['rightplate']).keys())
right_plate = LiftingSurface(name='right_plate', spatial_representation=spatial_rep, primitive_names=right_plate_primitive_names)
sys_rep.add_component(right_plate)
# right_plate.plot()


# wing mesh
num_spanwise_vlm = 25
num_chordwise_vlm = 25
leading_edge = wing.project(np.linspace(np.array([0, -1.016, 0.01]), np.array([0, 1.016, 0.01]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([0.508, -1.016, 0]), np.array([0.508, 1.016, 0]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
# spatial_rep.plot_meshes([chord_surface])


wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 0.5]), direction=np.array([0., 0., 1.]), grid_search_n=50, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=50, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
# spatial_rep.plot_meshes([wing_camber_surface])
wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)


# left plate mesh
num_spanwise_plate = 8
num_chordwise_plate = 8
left_plate_leading_edge = left_plate.project(np.linspace(np.array([0, -1.016, -0.051]), np.array([0, -1.016, 0]), num_spanwise_plate), direction=np.array([0., 0., -1.]), plot=False)
left_plate_trailing_edge = left_plate.project(np.linspace(np.array([0.508, -1.016, -0.051]), np.array([0.508, -1.016, 0]), num_spanwise_plate), direction=np.array([0., 0., -1.]), plot=False)
left_plate_vlm_mesh = am.linspace(left_plate_leading_edge, left_plate_trailing_edge, num_chordwise_plate)
# spatial_rep.plot_meshes([left_plate_vlm_mesh])
left_plate_vlm_mesh_name = 'left_plate_vlm_mesh'
sys_rep.add_output(left_plate_vlm_mesh_name, left_plate_vlm_mesh)

# right plate mesh
right_plate_leading_edge = right_plate.project(np.linspace(np.array([0, 1.016, -0.051]), np.array([0, 1.016, 0]), num_spanwise_plate), direction=np.array([0., 0., -1.]), plot=False)
right_plate_trailing_edge = right_plate.project(np.linspace(np.array([0.508, 1.016, -0.051]), np.array([0.508, 1.016, 0]), num_spanwise_plate), direction=np.array([0., 0., -1.]), plot=False)
right_plate_vlm_mesh = am.linspace(right_plate_leading_edge, right_plate_trailing_edge, num_chordwise_plate)
# spatial_rep.plot_meshes([right_plate_vlm_mesh])
right_plate_vlm_mesh_name = 'right_plate_vlm_mesh'
sys_rep.add_output(right_plate_vlm_mesh_name, right_plate_vlm_mesh)




sys_param.setup()

 

 

# design scenario
design_scenario = cd.DesignScenario(name='wig')
wig_model = m3l.Model()
wig_condition = cd.CruiseCondition(name='wig')
wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
wig_condition.set_module_input(name='altitude', val=0)
wig_condition.set_module_input(name='mach_number', val=0.1711, dv_flag=True, lower=0.1, upper=0.3)
wig_condition.set_module_input(name='range', val=1000)
wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0), dv_flag=False, lower=np.deg2rad(-10), upper=np.deg2rad(10))
wig_condition.set_module_input(name='flight_path_angle', val=0)
wig_condition.set_module_input(name='roll_angle', val=0)
wig_condition.set_module_input(name='yaw_angle', val=0)
wig_condition.set_module_input(name='wind_angle', val=0)
wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))
ac_states = wig_condition.evaluate_ac_states()
wig_model.register_output(ac_states)




# create a mirrored wing mesh:
alpha = np.deg2rad(5)
h = 0.04807366201
mirror = Mirror(component=wing,mesh_name=wing_vlm_mesh_name,ns=num_spanwise_vlm,nc=num_chordwise_vlm,point=np.array([0.508, 0, 0])) # chord is 0.508m
mirror.set_module_input('alpha', val=alpha, dv_flag=False)
mirror.set_module_input('h', val=h, dv_flag=False)
wing_mesh_out, wing_mirror_mesh = mirror.evaluate()
wig_model.register_output(wing_mirror_mesh)
wig_model.register_output(wing_mesh_out)

# also mirror the end plates:
left_plate_mirror = Mirror(component=left_plate,mesh_name=left_plate_vlm_mesh_name,ns=num_spanwise_plate,nc=num_chordwise_plate,point=np.array([0.508, 0, 0])) # chord is 0.508m
left_plate_mirror.set_module_input('alpha', val=alpha, dv_flag=False)
left_plate_mirror.set_module_input('h', val=h, dv_flag=False)
left_plate_mesh_out, left_plate_mirror_mesh = left_plate_mirror.evaluate()
wig_model.register_output(left_plate_mirror_mesh)
wig_model.register_output(left_plate_mesh_out)

right_plate_mirror = Mirror(component=right_plate,mesh_name=right_plate_vlm_mesh_name,ns=num_spanwise_plate,nc=num_chordwise_plate,point=np.array([0.508, 0, 0])) # chord is 0.508m
right_plate_mirror.set_module_input('alpha', val=alpha, dv_flag=False)
right_plate_mirror.set_module_input('h', val=h, dv_flag=False)
right_plate_mesh_out, right_plate_mirror_mesh = right_plate_mirror.evaluate()
wig_model.register_output(right_plate_mirror_mesh)
wig_model.register_output(right_plate_mesh_out)






# VLM solver
vlm_model = VASTFluidSover(
    surface_names=[wing_vlm_mesh_name+'_out', 
                   wing_vlm_mesh_name+'_mirror', 
                   left_plate_vlm_mesh_name+'_out', 
                   left_plate_vlm_mesh_name+'_mirror',
                   right_plate_vlm_mesh_name+'_out', 
                   right_plate_vlm_mesh_name+'_mirror',],
    surface_shapes=[(1, ) + wing_camber_surface.evaluate().shape[1:],
                    (1, ) + wing_camber_surface.evaluate().shape[1:],
                    (1,num_spanwise_plate,num_chordwise_plate,3),
                    (1,num_spanwise_plate,num_chordwise_plate,3),
                    (1,num_spanwise_plate,num_chordwise_plate,3),
                    (1,num_spanwise_plate,num_chordwise_plate,3),],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='m',
    cl0=[0,0,0,0,0,0]
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




# connect the transformed meshes to VAST:

caddee_csdl_model.connect('system_model.wig.wig.wig.wing_vlm_meshmirror.wing_vlm_mesh_mirror',
                          'system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_system.MeshPreprocessing_comp.wing_vlm_mesh_mirror')

caddee_csdl_model.connect('system_model.wig.wig.wig.wing_vlm_meshmirror.wing_vlm_mesh_out',
                          'system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_system.MeshPreprocessing_comp.wing_vlm_mesh_out')

caddee_csdl_model.connect('system_model.wig.wig.wig.left_plate_vlm_meshmirror.left_plate_vlm_mesh_out',
                          'system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_system.MeshPreprocessing_comp.left_plate_vlm_mesh_out')

caddee_csdl_model.connect('system_model.wig.wig.wig.right_plate_vlm_meshmirror.right_plate_vlm_mesh_out',
                          'system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_system.MeshPreprocessing_comp.right_plate_vlm_mesh_out')

caddee_csdl_model.connect('system_model.wig.wig.wig.left_plate_vlm_meshmirror.left_plate_vlm_mesh_mirror',
                          'system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_system.MeshPreprocessing_comp.left_plate_vlm_mesh_mirror')

caddee_csdl_model.connect('system_model.wig.wig.wig.right_plate_vlm_meshmirror.right_plate_vlm_mesh_mirror',
                          'system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_system.MeshPreprocessing_comp.right_plate_vlm_mesh_mirror')


# connect altitude to the mirror:
#caddee_csdl_model.connect('system_model.wig.wig.wig.wig_ac_states_operation.wig_altitude',
#                          'system_model.wig.wig.wig.mirror.h')

 

sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()


L = sim['system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_out_L']
C_L = sim['system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_out_C_L']
C_D_i = sim['system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_out_C_D_i']
C_D = sim['system_model.wig.wig.wig.wing_vlm_mesh_outwing_vlm_mesh_mirrorleft_plate_vlm_mesh_outleft_plate_vlm_mesh_mirrorright_plate_vlm_mesh_outright_plate_vlm_mesh_mirror_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_vlm_mesh_out_C_D_total']
h = sim['system_model.wig.wig.wig.wing_vlm_meshmirror.h']
b = 2.032 # wing span
S = 1.032256

print('h/b: ', h/b)
print('h/sqrt(S): ', h/np.sqrt(S))
print('C_L: ', C_L)
print('C_D_i: ', C_D_i)
print('C_D: ', C_D)
print('L/D: ', C_L/C_D)



"""
# plot the meshes to see if the mirror is working:
original = sim['system_model.wig.wig.wig.mirror.debug_mesh']
mirror_mesh = sim['system_model.wig.wig.wig.mirror.wing_vlm_mesh_mirror']
mesh_out = sim['system_model.wig.wig.wig.mirror.wing_vlm_mesh_out']

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(num_spanwise_vlm):
    for j in range(num_chordwise_vlm):
        ax.scatter(mirror_mesh[i,j,0], mirror_mesh[i,j,1], mirror_mesh[i,j,2], color='blue', s=50)
        ax.scatter(mesh_out[i,j,0], mesh_out[i,j,1], mesh_out[i,j,2], color='orange', s=50)
        ax.scatter(original[i,j,0], original[i,j,1], original[i,j,2], color='black', s=50)
plt.show()
"""
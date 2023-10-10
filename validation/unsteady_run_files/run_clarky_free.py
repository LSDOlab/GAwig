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


file_name = '../stp/clarky.stp'
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=file_name)
spatial_rep.refit_geometry(file_name=file_name)

# wing
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['WingGeom']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
sys_rep.add_component(wing)
# wing.plot()

# wing mesh
num_spanwise_vlm = 24
num_chordwise_vlm = 18
leading_edge = wing.project(np.linspace(np.array([0, -1.016, 0.01]), np.array([0, 1.016, 0.01]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([0.508, -1.016, 0]), np.array([0.508, 1.016, 0]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
# spatial_rep.plot_meshes([chord_surface])


wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=30, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=50, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
spatial_rep.plot_meshes([wing_camber_surface])
wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)


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

# create a mirrored mesh
h = 0.124273977
alpha = np.deg2rad(5.)
AR = 4
span = 2.032
chord = span/AR

mirror = Mirror(component=wing,mesh_name=wing_vlm_mesh_name,ns=num_spanwise_vlm,nc=num_chordwise_vlm,point=np.array([0.508, 0, 0]))
mirror.set_module_input('alpha', val=alpha, dv_flag=False)
mirror.set_module_input('h', val=h, dv_flag=False)
mesh_out, mirror_mesh = mirror.evaluate()
wig_model.register_output(mirror_mesh)
wig_model.register_output(mesh_out)


'''
UVLM
'''
from lsdo_uvlm.uvlm_operation import UVLMCore
nx = num_chordwise_vlm
ny = num_spanwise_vlm

num_nodes = 20
nt = num_nodes + 1

u_val = np.ones(num_nodes).reshape(num_nodes,1)*10.

uvlm_parameters = [('u',True,u_val),
                       ('v',True,np.zeros((num_nodes, 1))),
                       ('w',True,np.ones((num_nodes, 1))),
                       ('p',True,np.zeros((num_nodes, 1))),
                       ('q',True,np.zeros((num_nodes, 1))),
                       ('r',True,np.zeros((num_nodes, 1))),
                       ('theta',True,np.zeros((num_nodes, 1))),
                       ('psi',True,np.zeros((num_nodes, 1))),
                    #    ('x',True,np.zeros((num_nodes, 1))),
                    #    ('y',True,np.zeros((num_nodes, 1))),
                    #    ('z',True,np.zeros((num_nodes, 1))),
                    #    ('phiw',True,np.zeros((num_nodes, 1))),
                       ('gamma',True,np.zeros((num_nodes, 1))),
                       ('psiw',True,np.zeros((num_nodes, 1)))]

mesh_dict = {
    "num_y": ny,
    "num_x": nx,
    "wing_type": "rect",
    "symmetry": False,
    "span": span,
    "root_chord": chord,
    "span_cos_spacing": False,
    "chord_cos_spacing": False,
}


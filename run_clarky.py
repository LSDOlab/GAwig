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
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
import csdl
from mirror import Mirror




file_name = 'GAwig/clarky.stp'

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
num_spanwise_vlm = 18
num_chordwise_vlm = 8
leading_edge = wing.project(np.linspace(np.array([0, -1.016, 0.01]), np.array([0, 1.016, 0.01]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([0.508, -1.016, 0]), np.array([0.508, 1.016, 0]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
#spatial_rep.plot_meshes([chord_surface])


wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=30, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=30, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
spatial_rep.plot_meshes([wing_camber_surface])
wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)


sys_param.setup()
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
from rotor import Rotor, Rotor2
from expansion_op import ac_expand
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from mpl_toolkits.mplot3d import proj3d
from caddee.core.caddee_core.system_representation.prescribed_actuations import PrescribedRotation
from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, PostProcessor




file_name = 'Liberty    Lifter3.stp'
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
num_props = 8
props = [] # we go from 1-indexed to 0-indexed here
for i in range(num_props):
    prop = build_component('prop_'+str(i), ['Prop'+str(i+1),'Hub'+str(i+1)])
    props.append(prop)
#endregion

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
wing_camber_surface_np = wing_camber_surface.value.reshape((num_chordwise_vlm, num_spanwise_vlm, 3))
# print(wing_camber_surface.value[0, -2:, :, :])
spatial_rep.plot_meshes([wing_camber_surface])

flap_mesh = wing_camber_surface.value[0, -2:, :, :]

flap_mesh = np.zeros((num_chordwise_vlm, num_spanwise_vlm, 3))
flap_mesh[0:-2, :, :] =  wing_camber_surface.value[0, 0:-2, :, :]

theta = np.deg2rad(-10)
rot_mat = np.array([
    [np.cos(theta), 0, -np.sin(theta)],
    [0, 1, 0],
    [np.sin(theta), 0, np.cos(theta)],
])

nx_list = [-2, -1]

# second_to_last_row = wing_camber_surface.value[0, -2, :]
# last_row = wing_camber_surface.value[0, -1, :]
# print(second_to_last_row.shape)
# for i in range(22):
#     rot_vec = second_to_last_row[i, :]
#     rot_vec_2 = last_row[i, :]
#     flap_mesh[-2, i, :] =  rot_mat @ rot_vec
#     flap_mesh[-1, i, :] =  rot_mat @ rot_vec_2

for nx in nx_list:
    for ny in range(22):
        rot_vec = wing_camber_surface_np[nx, ny, :]
        if nx == -2:
            origin_vec = wing_camber_surface_np[nx-1, ny, :]
        elif nx == -1:
            origin_vec = wing_camber_surface_np[nx-2, ny, :]
        print((rot_vec), (rot_mat @ (rot_vec-origin_vec)))
        flap_mesh[nx, ny, :] = rot_mat @ (rot_vec-origin_vec) + origin_vec

# print(flap_mesh.shape)
spatial_rep.plot_meshes([flap_mesh])



test_vec = np.array([
    [np.sqrt(2)/2],
    [0],
    [-np.sqrt(2)/2],
])



print(test_vec)
print(rot_mat @ test_vec)

exit()
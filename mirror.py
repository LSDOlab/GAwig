import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l





class Mirror(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.num_nodes = None
        self.parameters.declare('mesh_name')
        self.parameters.declare('ns')
        self.parameters.declare('nc')
        self.parameters.declare('nt')
        self.parameters.declare('point')

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.mesh_name = self.parameters['mesh_name']
        self.ns = self.parameters['ns']
        self.nc = self.parameters['nc']
        self.nt = self.parameters['nt']
        self.point = self.parameters['point']

    def compute(self):
        mesh_name = self.parameters['mesh_name']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        point = self.parameters['point']
        csdl_model = MirrorCSDL(module=self,mesh_name=mesh_name,ns=ns,nc=nc,nt=nt,point=point,mesh_value=self.mesh)
        return csdl_model

    # def evaluate(self, theta : m3l.Variable, h : m3l.Variable):
    def evaluate(self, h : m3l.Variable):
        mesh_name = self.parameters['mesh_name']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
 
        self.name = mesh_name + 'mirror'
        self.arguments = {
            #'theta' : theta,
            'h' : h,
        }

        mirror_mesh = m3l.Variable(mesh_name + '_mirror', shape=(nt,nc,ns,3), operation=self)
        mesh_out = m3l.Variable(mesh_name + '_out', shape=(nt,nc,ns,3), operation=self)

        return mesh_out, mirror_mesh

 



class MirrorCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('mesh_name')
        self.parameters.declare('ns')
        self.parameters.declare('nc')
        self.parameters.declare('nt')
        self.parameters.declare('point')
        self.parameters.declare('mesh_value', default=None)
 
    def define(self):
        mesh_name = self.parameters['mesh_name']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        point = self.parameters['point']
        mesh_value = self.parameters['mesh_value']

        alpha = -1 * self.declare_variable('theta', shape=(1,)) # why the -1 though?
        h = self.declare_variable('h', shape=(1,))

        # the rotation matrix:
        rotation_matrix_y = self.create_output('rotation_matrix_y',shape=(3,3),val=0)
        rotation_matrix_y[0,0] = csdl.reshape(csdl.cos(alpha), (1,1))
        rotation_matrix_y[0,2] = csdl.reshape(csdl.sin(alpha), (1,1))
        rotation_matrix_y[1,1] = self.create_input('one',shape=(1,1),val=1)
        rotation_matrix_y[2,0] = csdl.reshape(-csdl.sin(alpha), (1,1))
        rotation_matrix_y[2,2] = csdl.reshape(csdl.cos(alpha), (1,1))

        mesh_in_one_by = self.declare_variable(mesh_name, shape=(nc,ns,3), val=mesh_value)
        mesh_in = csdl.expand(mesh_in_one_by, (nt,nc,ns,3), 'ijk->lijk')
        self.register_output('debug_mesh', 1*mesh_in)

        # rotate the mesh:
        translated_mesh_points = mesh_in - np.tile(point, (nt, nc, ns, 1))

        # Apply rotation to all eval_points using matrix multiplication
        rotated_mesh_points = csdl.einsum(translated_mesh_points, rotation_matrix_y, subscripts='ijkl,lm->ijkm')

        rotated_mesh = rotated_mesh_points + np.tile(point, (nt, nc, ns, 1))

        # translate the mesh based on altitude:
        dh = self.create_output('dh', shape=(nt,nc,ns,3), val=0)
        dh[:,:,:,2] = csdl.expand(h, (nt,nc,ns,1), 'i->abci')

        mesh = rotated_mesh + dh
        #mesh = rotated_mesh_points + dh
        self.register_output(mesh_name+'_out', mesh)

        # create the mirrored mesh:
        mirror = self.create_output(mesh_name+'_mirror', shape=(nt,nc,ns,3), val=0)
        mirror[:,:,:,0] = mesh[:,:,:,0]
        mirror[:,:,:,1] = mesh[:,:,:,1]
        mirror[:,:,:,2] = -1*mesh[:,:,:,2]


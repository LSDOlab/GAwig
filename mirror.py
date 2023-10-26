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

    def evaluate(self):
        mesh_name = self.parameters['mesh_name']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
 
        self.name = mesh_name + 'mirror'
        self.arguments = {}

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

        alpha = self.register_module_input('theta', shape=(1,), computed_upstream=False)
        h = self.register_module_input('h', shape=(1,), computed_upstream=False)
        #h = self.register_module_input('h', shape=(1), promotes=True)

        # the rotation matrix:
        rotation_matrix_y = self.create_output('rotation_matrix_y',shape=(3,3),val=0)
        rotation_matrix_y[0,0] = csdl.reshape(csdl.cos(alpha), (1,1))
        rotation_matrix_y[0,2] = csdl.reshape(csdl.sin(alpha), (1,1))
        rotation_matrix_y[1,1] = self.create_input('one',shape=(1,1),val=1)
        rotation_matrix_y[2,0] = csdl.reshape(-csdl.sin(alpha), (1,1))
        rotation_matrix_y[2,2] = csdl.reshape(csdl.cos(alpha), (1,1))

        mesh_in_one_by = self.declare_variable(mesh_name, shape=(nc,ns,3), val=mesh_value)
        mesh_in = csdl.expand(mesh_in_one_by, (nt,nc,ns,3), 'ijk->lijk')
        #mesh_in = self.register_module_input(mesh_name, shape=(nt,nc,ns,3), promotes=True)
        self.register_output('debug_mesh', 1*mesh_in)

        # rotate the mesh:
        translated_mesh_points = mesh_in - np.tile(point, (nt, nc, ns, 1))
        rotated_mesh_points = self.create_output('rotated_mesh_points', shape=(nt,nc,ns,3), val=0)
        for i in range(nt):
            for j in range(nc):
                for k in range(ns):
                    eval_point = csdl.reshape(translated_mesh_points[i,j,k,:], (3))
                    #eval_point = csdl.reshape(mesh_in[i,j,:], (3))
                    rotated_mesh_points[i,j,k,:] = csdl.reshape(csdl.matvec(rotation_matrix_y, eval_point), (1,1,1,3))

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


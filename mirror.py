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

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.mesh_name = self.parameters['mesh_name']
        self.ns = self.parameters['ns']
        self.nc = self.parameters['nc']

    def compute(self):
        mesh_name = self.parameters['mesh_name']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        csdl_model = MirrorCSDL(module=self,mesh_name=mesh_name,ns=ns,nc=nc)
        return csdl_model
    
    def evaluate(self):
        mesh_name = self.parameters['mesh_name']
        ns = self.parameters['ns']
        nc = self.parameters['nc']

        self.name = 'mirror'
        self.arguments = {}
        
        mirror_mesh = m3l.Variable(mesh_name + '_mirror', shape=(ns,nc,3), operation=self)

        return mirror_mesh


class MirrorCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('mesh_name')
        self.parameters.declare('ns')
        self.parameters.declare('nc')

    def define(self):
        mesh_name = self.parameters['mesh_name']
        ns = self.parameters['ns']
        nc = self.parameters['nc']

        #mesh = self.declare_variable('mesh', shape=(ns,nc,3))
        mesh = self.register_module_input(mesh_name, shape=(ns,nc,3), promotes=True)

        mirror_mesh = self.create_output(mesh_name + '_mirror', shape=(ns,nc,3), val=0)
        mirror_mesh[:,:,0] = mesh[:,:,0]
        mirror_mesh[:,:,1] = mesh[:,:,1]
        mirror_mesh[:,:,2] = -1*mesh[:,:,2]

        # self.print_var(mirror_mesh)
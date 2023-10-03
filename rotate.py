import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l





class Rotate(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.num_nodes = None
        self.parameters.declare('mesh_name')
        self.parameters.declare('shape')

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.mesh_name = self.parameters['mesh_name']
        self.shape = self.parameters['shape']

    def compute(self):
        mesh_name = self.parameters['mesh_name']
        shape = self.parameters['shape']
        csdl_model = RotateCSDL(module=self,mesh_name=mesh_name,shape=shape)
        return csdl_model

    def evaluate(self):
        mesh_name = self.parameters['mesh_name']
        shape = self.parameters['shape']
 
        self.name = mesh_name + 'rotate'
        self.arguments = {}

        rotated_mesh = m3l.Variable(mesh_name + '_rotated', shape=shape, operation=self)

        return rotated_mesh
    


class RotateCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('mesh_name')
        self.parameters.declare('shape')

    def define(self):
        mesh_name = self.parameters['mesh_name']
        shape = self.parameters['shape']



        omega = self.register_module_input('omega', shape=(1,), computed_upstream=False)
        mesh = self.register_module_input(mesh_name, shape=shape, promotes=True)
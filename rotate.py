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
        self.parameters.declare('n')
        self.parameters.declare('dt')

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.n = self.parameters['n']
        self.dt = self.parameters['dt']

    def compute(self):
        n = self.parameters['n']
        dt = self.parameters['dt']
        csdl_model = RotateCSDL(module=self,n=n,dt=dt)
        return csdl_model

    def evaluate(self):
        component = self.parameters['component']
        n = self.parameters['n']
 
        self.name = component.name + '_rotate'
        self.arguments = {}

        angles = m3l.Variable('angles', shape=(n), operation=self)

        return angles
    


class RotateCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('n')
        self.parameters.declare('dt')

    def define(self):
        n = self.parameters['n']
        dt = self.parameters['dt']


        rpm = self.register_module_input('rpm', shape=(1,), computed_upstream=False)
        rps = rpm/60
        rad_per_sec = rps*2*np.pi
        rad_per_dt = rad_per_sec*dt

        angles = self.create_output('angles', shape=(n), val=0)
        for i in range(n):
            angles[i] = i*rad_per_dt
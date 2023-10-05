import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l





class Payload(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.num_nodes = None
        self.parameters.declare('mass')

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.mass = self.parameters['mass']

    def compute(self):
        mass = self.parameters['mass']
        csdl_model = PayloadCSDL(module=self,mass=mass)
        return csdl_model

    def evaluate(self):
        mass = self.parameters['mass']
 
        self.name = 'payload'
        self.arguments = {}

        forces = m3l.Variable('forces', shape=(3), operation=self)
        moments = m3l.Variable('moments', shape=(3), operation=self)

        return forces, moments

 

 

class PayloadCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('mass', default=25000)
        self.parameters.declare('gravity', default=9.81)
 
    def define(self):
        m = self.parameters['mass']
        g = self.parameters['gravity']


        x = self.register_module_input('x', shape=(1,), computed_upstream=False) # payload position

        theta = self.declare_variable('theta', shape=(1,), val=0)

        a = x*csdl.cos(theta) # moment arm


        F = self.create_output('forces', shape=(3), val=0)
        F[1] = m*g*csdl.sin(theta)
        F[2] = m*g*csdl.cos(theta)

        M = self.create_output('moments', shape=(3), val=0)
        M[1] = a*m*g



        
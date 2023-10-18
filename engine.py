import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l


class Engine(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.num_nodes = None
        self.parameters.declare('engine_name')
        self.parameters.declare('sfc')

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.engine_name = self.parameters['engine_name']
        self.sfc = self.parameters['sfc']

    def compute(self):
        engine_name = self.parameters['engine_name']
        sfc = self.parameters['sfc']

        csdl_model = EngineCSDL(module=self,
                               engine_name=engine_name,
                               sfc=sfc,
                               )
        return csdl_model

    def evaluate(self):
        engine_name = self.parameters['engine_name']
 
        self.name = engine_name + '_engine'
        self.arguments = {}

        fc = m3l.Variable(engine_name + '_fc', shape=(1,), operation=self)

        return fc
    


# INPUTS: specific fuel consumption, torque and rpm
# OUTPUTS: fuel consumption (lb/hr)

class EngineCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('engine_name')
        self.parameters.declare('sfc')
 
    def define(self):
        engine_name = self.parameters['engine_name']
        sfc = self.parameters['sfc']




        rpm = self.declare_variable('rpm', shape=(1,))
        torque = self.declare_variable('torque', shape=(1,))
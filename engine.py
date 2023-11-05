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
        self.parameters.declare('sfc', default=0.46)

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

    def evaluate(self, torque):
        engine_name = self.parameters['engine_name']
 
        self.name = engine_name + '_engine'
        self.arguments = {
            # 'rpm': rpm,
            'torque' : torque,
        }

        fc = m3l.Variable(engine_name + '_fc', shape=(1,), operation=self)
        pwr = m3l.Variable(engine_name + '_pwr', shape=(1,), operation=self)

        return fc, pwr
    


# INPUTS: specific fuel consumption, torque and rpm
# OUTPUTS: fuel consumption (lb/hr) and power (hp)

class EngineCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('engine_name')
        self.parameters.declare('sfc', default=0.46)
 
    def define(self):
        engine_name = self.parameters['engine_name']
        sfc = self.parameters['sfc']




        rpm = self.declare_variable('rpm', shape=(1,))
        torque = (self.declare_variable('torque', shape=(1,))**2 + 1E-12)**0.5 # (N-m)

        omega = (rpm/60)*2*np.pi # (rad/sec)

        pwr = torque*omega # (W)
        hp = pwr/745.699872 # (hp)
        self.print_var(hp)
        self.register_output(engine_name + '_pwr', hp)

        fuel_consumption = sfc*hp # (lb/hr)
        self.register_output(engine_name + '_fc', fuel_consumption)
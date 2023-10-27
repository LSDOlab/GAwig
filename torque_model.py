import m3l 
import csdl
import numpy as np


class TorqueModel(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('rotor_name', types=str)


    def evaluate(self, prop_fx : m3l.Variable) -> tuple:
        rotor_name = self.parameters['rotor_name']
        self.name = f'torque_operation_{rotor_name}'

        self.arguments = {}
        self.num_blades = len(prop_fx)
        for blade_number in range(self.num_blades):
            self.arguments[f'fx_spanwise_blade_{blade_number}'] = prop_fx[blade_number]
            
        # self.arguments['rpm'] = rpm


        torque = m3l.Variable(name='torque', shape=(1, ), operation=self)
        self.fx_shape = prop_fx[blade_number].shape

        return torque

    def compute(self):
        csdl_model = TorqueModelCSDL(
            fx_shape=self.fx_shape,
            num_blades=self.num_blades,
        )

        return csdl_model
    

class TorqueModelCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('fx_shape', types=tuple)
        self.parameters.declare('num_blades', types=int)
        self.parameters.declare('num_props', types=int)
        self.parameters.declare('rotor_efficiency', types=float, default=0.8)


    def define(self):
        shape = self.parameters['fx_shape']
        num_blades = self.parameters['num_blades']
        eta = self.parameters['rotor_efficiency']

        thrust_compute = self.create_input('thrust_compute', val=0)
        omega = self.declare_variable('rpm', shape=(1, )) * 2 * np.pi / 60
        V = self.declare_variable('velocity', shape=(1, ))

        for i in range(num_blades):
            prop_fx = self.declare_variable(f'fx_spanwise_blade_{i}', shape=shape)
            blade_thrust = csdl.sum(prop_fx)

            thrust_compute = thrust_compute + blade_thrust

        
        self.register_output('total_thrust', thrust_compute * 1)

        torque  = thrust_compute * V / (omega * eta)
        self.register_output('torque', torque)

        


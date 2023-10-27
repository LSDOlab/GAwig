import m3l 
import csdl


class TorqueModel(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        return super().initialize(kwargs)
    
    def evaluate(self, prop_fx : m3l.Variable) -> tuple:
        self.name = f'torque_{prop_fx.name}_operation'

        self.arguments = {}
        self.num_blades = len(prop_fx)
        for blade_number in self.num_blades:
            self.arguments[f'fx_spanwise_blade_{blade_number}'] = prop_fx[blade_number]


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
        self.parameters.declare('rotor_efficiency', types=float, default=0.8)


    def define(self):
        shape = self.parameters['fx_shape']
        num_blades = self.parameters['num_blades']

        thrust_compute = self.create_input('thrust_compute', val=0)

        for i in range(num_blades):
            prop_fx = self.declare_variable(f'fx_spanwise_blade_{i}', shape=shape)
            blade_thrust = csdl.sum(prop_fx)

            thrust_compute = thrust_compute + blade_thrust

        
        self.register_output('total_thrust', thrust_compute * 1)
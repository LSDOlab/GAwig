import m3l 
import csdl
import numpy as np


class AirfoilLinearModel(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('airfoil_name', types=str, default='htail')
        self.parameters.declare('cl_slope', types=float, default=0.05251592)
        self.parameters.declare('cl0', types=float, default=0.00018381)
        self.parameters.declare('cd_c2', types=float, default=0.00021278)
        self.parameters.declare('cd_c1', types=float, default=0.00000117)
        self.parameters.declare('cd_c0', types=float, default=0.00015929)
        self.parameters.declare('density', types=float, default=1.1)
        self.parameters.declare('area', types=float, default=70.23059100198903)

    def evaluate(self, deflection : m3l.Variable, v_inf : m3l.Variable, theta : m3l.Variable) -> tuple:
        airfoil_name = self.parameters['airfoil_name']
        self.name = f'linear_airfoil_{airfoil_name}'

        self.arguments = {'deflection':deflection, 'v_inf':v_inf, 'theta':theta}

        L = m3l.Variable(name='L', shape=(1, ), operation=self) # tecninically I think these are Fz and -Fx
        D = m3l.Variable(name='D', shape=(1, ), operation=self)

        return L, D

    def compute(self):
        csdl_model = AirfoilLinearModelCSDL(
            cl_slope=self.parameters['cl_slope'],
            cl0=self.parameters['cl0'],
            cd_slope=self.parameters['cd_c2'],
            cd_slope=self.parameters['cd_c1'],
            cd0=self.parameters['cd_c0'],
            density=self.parameters['density'],
            area=self.parameters['area']
        )
        return csdl_model
    

class AirfoilLinearModelCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('cl_slope', types=float)
        self.parameters.declare('cl0', types=float)
        self.parameters.declare('cd_c2', types=float)
        self.parameters.declare('cd_c1', types=float)
        self.parameters.declare('cd_c0', types=float)
        self.parameters.declare('density', types=float)
        self.parameters.declare('area', types=float)

    def define(self):
        cl_slope = self.parameters['cl_slope']
        cl0 = self.parameters['cl0']
        cd_c0 = self.parameters['cd_c0']
        cd_c1 = self.parameters['cd_c1']
        cd_c2 = self.parameters['cd_c2']
        density = self.parameters['density']
        area = self.parameters['area']

        theta = self.declare_variable('theta', shape=(1,))
        deflection = self.declare_variable('deflection', shape=(1,))
        v_inf = self.declare_variable('v_inf', shape=(1,))

        aoa = theta+deflection

        C_L = aoa*cl_slope + cl0
        C_D = aoa**2*cd_c2 + aoa*cd_c1 + cd_c0

        L = 1/2*density*v_inf**2*area*C_L
        D = 1/2*density*v_inf**2*area*C_D

        self.register_output('L', L)
        self.register_output('D', D)


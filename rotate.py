import numpy as np
import csdl


class Rotate(csdl.Model):
    def initialize(self):
        self.parameters.declare('n')
        self.parameters.declare('dt')

    def define(self):
        n = self.parameters['n']
        dt = self.parameters['dt']


        rpm = self.declare_variable('rpm', shape=(1,))
        rps = rpm/60
        rad_per_sec = rps*2*np.pi
        rad_per_dt = rad_per_sec*dt

        angles = self.create_output('angles', shape=(n), val=0)
        for i in range(n):
            angles[i] = i*rad_per_dt
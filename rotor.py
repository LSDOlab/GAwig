import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l


class Rotor(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.num_nodes = None
        self.parameters.declare('mesh_name')
        self.parameters.declare('num_blades')
        self.parameters.declare('ns')
        self.parameters.declare('nc')
        self.parameters.declare('nt')
        self.parameters.declare('dt')

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.mesh_name = self.parameters['mesh_name']
        self.num_blades = self.parameters['num_blades']
        self.ns = self.parameters['ns']
        self.nc = self.parameters['nc']
        self.nt = self.parameters['nt']
        self.dt = self.parameters['dt']

    def compute(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        dt = self.parameters['dt']
        csdl_model = RotorCSDL(module=self,
                               mesh_name=mesh_name,
                               num_blades=num_blades,
                               ns=ns,
                               nc=nc,
                               nt=nt,
                               dt=dt,)
        return csdl_model

    def evaluate(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
 
        self.name = mesh_name + '_rotor'
        self.arguments = {}

        mesh = m3l.Variable(mesh_name + '_rotor', shape=(num_blades,nt,ns,nc,3), operation=self)

        return mesh

 
# creates full rotor geometry from a single blade mesh, a point, and a normal vector

class RotorCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('mesh_name')
        self.parameters.declare('num_blades', default=6)
        self.parameters.declare('ns', default=4)
        self.parameters.declare('nc', default=2)
        self.parameters.declare('nt', default=2)
        self.parameters.declare('dt', default=0.001)
 
    def define(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        dt = self.parameters['dt']


        rad_per_blade = 2*np.pi/num_blades

        # rpm = self.declare_variable('rpm', shape=(1,))
        rpm = self.register_module_input('rpm', shape=(1,), computed_upstream=False)
        rps = rpm/60
        rad_per_sec = rps*2*np.pi
        rad_per_dt = rad_per_sec*dt

        # a single blade mesh:
        mesh = self.declare_variable(mesh_name, shape=(nc,ns,3))
        # the center of the rotor disk:
        point = self.declare_variable('point', shape=(3,))
        # normal vector to the rotor disk:
        vector = self.declare_variable('vector', shape=(3,))
        normalized_vector = vector/csdl.expand(csdl.pnorm(vector, 2), (3,))
        x, y, z = normalized_vector[0], normalized_vector[1], normalized_vector[2]


        shifted_mesh = mesh - csdl.expand(point, (nc,ns,3), 'k->ijk')

        rot_mesh = self.create_output('rot_mesh', shape=(num_blades,nt,nc,ns,3), val=0)

        for i in range(num_blades):
            set_angle = rad_per_blade*i

            for j in range(nt):
                angle = set_angle + rad_per_dt*j

                rot_mat = self.create_output('rot_mat_' + str(i) + str(j), shape=(3,3), val=0)
                cos_theta, sin_theta = csdl.cos(angle), csdl.sin(angle)
                one_minus_cos_theta = 1 - cos_theta
                rot_mat[0,0] = csdl.reshape(cos_theta + x**2 * one_minus_cos_theta, (1,1))
                rot_mat[0,1] = csdl.reshape(x * y * one_minus_cos_theta - z * sin_theta, (1,1))
                rot_mat[0,2] = csdl.reshape(x * z * one_minus_cos_theta + y * sin_theta, (1,1))
                rot_mat[1,0] = csdl.reshape(y * x * one_minus_cos_theta + z * sin_theta, (1,1))
                rot_mat[1,1] = csdl.reshape(cos_theta + y**2 * one_minus_cos_theta, (1,1))
                rot_mat[1,2] = csdl.reshape(y * z * one_minus_cos_theta - x * sin_theta, (1,1))
                rot_mat[2,0] = csdl.reshape(z * x * one_minus_cos_theta - y * sin_theta, (1,1))
                rot_mat[2,1] = csdl.reshape(z * y * one_minus_cos_theta + x * sin_theta, (1,1))
                rot_mat[2,2] = csdl.reshape(cos_theta + z**2 * one_minus_cos_theta, (1,1))

                for k in range(nc):
                    for l in range(ns):
                        mesh_point = csdl.reshape(shifted_mesh[k,l,:], (3))

                        rot_mesh[i,j,k,l,:] = csdl.reshape(csdl.matvec(rot_mat, mesh_point), (1,1,1,1,3))





        rotor = rot_mesh + csdl.expand(point, (num_blades,nt,nc,ns,3), 'm->ijklm')
        self.register_output('rotor', rotor)


    


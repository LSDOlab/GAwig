import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l

"""
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
        self.parameters.declare('dir')

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.mesh_name = self.parameters['mesh_name']
        self.num_blades = self.parameters['num_blades']
        self.ns = self.parameters['ns']
        self.nc = self.parameters['nc']
        self.nt = self.parameters['nt']
        self.dt = self.parameters['dt']
        self.dir = self.parameters['dir']

    def compute(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        dt = self.parameters['dt']
        dir = self.parameters['dir']
        csdl_model = RotorCSDL(module=self,
                               mesh_name=mesh_name,
                               num_blades=num_blades,
                               ns=ns,
                               nc=nc,
                               nt=nt,
                               dt=dt,
                               dir=dir,)
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
        self.parameters.declare('dir', default=1)
 
    def define(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        dt = self.parameters['dt']
        dir = self.parameters['dir']


        rad_per_blade = 2*np.pi/num_blades

        # rpm = self.declare_variable('rpm', shape=(1,))
        rpm = self.register_module_input('rpm', shape=(1,), computed_upstream=False)
        rps = rpm/60
        rad_per_sec = rps*2*np.pi
        rad_per_dt = rad_per_sec*dt

        # a single blade mesh:
        mesh = self.declare_variable(mesh_name, shape=(nc,ns,3))
        self.print_var(mesh)
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
                angle = set_angle + rad_per_dt*j*dir

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
"""

"""
class Rotor2(m3l.ExplicitOperation):
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
        self.parameters.declare('dir')
        self.parameters.declare('r_point')

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.mesh_name = self.parameters['mesh_name']
        self.num_blades = self.parameters['num_blades']
        self.ns = self.parameters['ns']
        self.nc = self.parameters['nc']
        self.nt = self.parameters['nt']
        self.dt = self.parameters['dt']
        self.dir = self.parameters['dir']
        self.r_point = self.parameters['r_point']

    def compute(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        dt = self.parameters['dt']
        dir = self.parameters['dir']
        r_point = self.parameters['r_point']
        csdl_model = RotorCSDL2(module=self,
                                mesh=self.mesh,
                                mesh_name=mesh_name,
                                num_blades=num_blades,
                                ns=ns,
                                nc=nc,
                                nt=nt,
                                dt=dt,
                                dir=dir,
                                r_point=r_point,)
        return csdl_model

    def evaluate(self):#, h : m3l.Variable, rpm : m3l.Variable, theta : m3l.Variable):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
 
        self.name = mesh_name + '_rotor'
        self.arguments = {}
        #     'rpm' : rpm,
        #     'theta' : theta,
        #     'h' : h, 
        # }

        mesh_vars = []
        for i in range(num_blades):
            mesh_vars.append(m3l.Variable(self.name+str(i), shape=(nt,nc,ns,3), operation=self))

        mesh_out_vars = []
        for i in range(num_blades):
            mesh_out_vars.append(m3l.Variable(self.name+str(i)+'_out', shape=(nt,nc,ns,3), operation=self))

        mesh_out_velocities = []
        for i in range(num_blades):
            mesh_out_velocities.append(m3l.Variable(self.name+str(i)+'_out_velocity', shape=(nt,nc-1,ns-1,3), operation=self))
            # print(self.name+str(i)+'_out_velocity')

        mirror_mesh_vars = []
        for i in range(num_blades):
            mirror_mesh_vars.append(m3l.Variable(self.name+str(i)+'_mirror', shape=(nt,nc,ns,3), operation=self))

        debug_rotor = m3l.Variable(mesh_name + '_rotor', shape=(num_blades,nt,ns,nc,3), operation=self)

        return tuple(mesh_out_vars), tuple(mirror_mesh_vars), tuple(mesh_out_velocities)


# creates full rotor geometry from a single blade mesh, a point, and a normal vector

class RotorCSDL2(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('mesh_name')
        self.parameters.declare('num_blades', default=6)
        self.parameters.declare('ns', default=4)
        self.parameters.declare('nc', default=2)
        self.parameters.declare('nt', default=2)
        self.parameters.declare('dt', default=0.001)
        self.parameters.declare('dir', default=1)
        self.parameters.declare('r_point')
        self.parameters.declare('mesh')
 
    def define(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        dt = self.parameters['dt']
        dir = self.parameters['dir']
        r_point = self.parameters['r_point']
        mesh = self.parameters['mesh']




        alpha = self.declare_variable('theta', shape=(1,))
        h = self.declare_variable('h', shape=(1,))

        # the rotation matrix:
        rotation_matrix_y = self.create_output('rotation_matrix_y',shape=(3,3),val=0)
        rotation_matrix_y[0,0] = csdl.reshape(csdl.cos(alpha), (1,1))
        rotation_matrix_y[0,2] = csdl.reshape(csdl.sin(alpha), (1,1))
        rotation_matrix_y[1,1] = self.create_input('one',shape=(1,1),val=1)
        rotation_matrix_y[2,0] = csdl.reshape(-csdl.sin(alpha), (1,1))
        rotation_matrix_y[2,2] = csdl.reshape(csdl.cos(alpha), (1,1))






        rad_per_blade = 2*np.pi/num_blades

        rpm = self.declare_variable('rpm', shape=(1,))
        # rpm = self.register_module_input('rpm', shape=(1,), computed_upstream=False)
        rps = rpm/60
        rad_per_sec = rps*2*np.pi
        rad_per_dt = rad_per_sec*dt

        # a single blade mesh:
        mesh = self.declare_variable(mesh_name, shape=(nc,ns,3), val=mesh)*0.3048
        # the center of the rotor disk:
        point = self.declare_variable('point', shape=(3,))*0.3048
        # normal vector to the rotor disk:
        vector = self.declare_variable('vector', shape=(3,))*0.3048
        normalized_vector = vector/csdl.expand(csdl.pnorm(vector, 2), (3,))
        x, y, z = normalized_vector[0], normalized_vector[1], normalized_vector[2]


        shifted_mesh = mesh - csdl.expand(point, (nc,ns,3), 'k->ijk')
        debug_rot_mesh = self.create_output('debug_rot_mesh', shape=(num_blades,nt,nc,ns,3), val=0)
        for i in range(num_blades):
            rot_mesh = self.create_output('rot_mesh'+str(i), shape=(nt,nc,ns,3), val=0)
            set_angle = rad_per_blade*i

            for j in range(nt):
                angle = set_angle + rad_per_dt*j*dir

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

                        rot_mesh[j,k,l,:] = csdl.reshape(csdl.matvec(rot_mat, mesh_point), (1,1,1,3))
                        debug_rot_mesh[i,j,k,l,:] = csdl.reshape(csdl.matvec(rot_mat, mesh_point), (1,1,1,1,3))

            rotor = rot_mesh + csdl.expand(point, (nt,nc,ns,3), 'm->ijkm')
            self.register_output(mesh_name + '_rotor'+str(i), rotor)



            # NATIVE ROTOR MIRRORING!!!!!
            mesh_in = 1*rotor
            translated_mesh_points = mesh_in - np.tile(r_point, (nt, nc, ns, 1))
            rotated_mesh_points = self.create_output('rotated_mesh_points_'+str(i), shape=(nt,nc,ns,3), val=0)
            for j in range(nt):
                for k in range(nc):
                    for l in range(ns):
                        eval_point = csdl.reshape(translated_mesh_points[j,k,l,:], (3))
                        rotated_mesh_points[j,k,l,:] = csdl.reshape(csdl.matvec(rotation_matrix_y, eval_point), (1,1,1,3))

            rotated_mesh = rotated_mesh_points + np.tile(r_point, (nt, nc, ns, 1))

            # translate the mesh based on altitude:
            dh = self.create_output('dh'+str(i), shape=(nt,nc,ns,3), val=0)
            dh[:,:,:,2] = csdl.expand(h, (nt,nc,ns,1), 'i->abci')
            mesh_out = rotated_mesh + dh
            self.register_output(mesh_name + '_rotor' + str(i) + '_out', mesh_out)
            # print(f'{mesh_name}_rotor_{i}_out_velocity')
            coll_pts_coords = 0.25/2 * (mesh_out[:,0:nc-1, 0:ns-1, :] +mesh_out[:,0:nc-1, 1:ns, :]) +\
                                         0.75/2 * (mesh_out[:,1:, 0:ns-1, :]+mesh_out[:,1:, 1:, :])
            mesh_velocity = self.create_output(f'{mesh_name}_rotor{i}_out_velocity', shape=(nt, nc-1, ns-1, 3), val=0)
            for ii in range(nt):
                if ii == 0:
                    mesh_velocity[ii, :, :, :] = (-3 * coll_pts_coords[ii, :, :, :] + 4 * coll_pts_coords[ii + 1, :, :, :] - coll_pts_coords[ii + 2, :, :, :]) / (2 * dt)
               
                elif ii == (nt - 1):
                    mesh_velocity[ii, :, :, :] = (3 * coll_pts_coords[ii, :, :, :] - 4 * coll_pts_coords[ii - 1, :, :, :] + coll_pts_coords[ii - 2, :, :, :]) / (2 * dt)
                
                else:
                    mesh_velocity[ii, :, :, :] = (coll_pts_coords[ii+1, :, :, :] - coll_pts_coords[ii-1, :, :, :]) / (2 * dt)


            # create the mirrored mesh:
            mirror = self.create_output(mesh_name + '_rotor'+str(i)+'_mirror', shape=(nt,nc,ns,3), val=0)
            mirror[:,:,:,0] = mesh_out[:,:,:,0]
            mirror[:,:,:,1] = mesh_out[:,:,:,1]
            mirror[:,:,:,2] = -1*mesh_out[:,:,:,2]


        debug_rotor = debug_rot_mesh + csdl.expand(point, (num_blades,nt,nc,ns,3), 'm->ijklm')
        self.register_output('rotor', debug_rotor)


"""




class Rotor3(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('mesh', default=None)
        self.num_nodes = None
        self.parameters.declare('mesh_name')
        self.parameters.declare('num_blades')
        self.parameters.declare('ns')
        self.parameters.declare('nc')
        self.parameters.declare('nt')
        self.parameters.declare('dt')
        self.parameters.declare('dir')
        self.parameters.declare('r_point')
        self.parameters.declare('rpm', types=float, default=1090.)
        self.parameters.declare('point')

    def assign_attributes(self):
        self.mesh = self.parameters['mesh']
        self.mesh_name = self.parameters['mesh_name']
        self.num_blades = self.parameters['num_blades']
        self.ns = self.parameters['ns']
        self.nc = self.parameters['nc']
        self.nt = self.parameters['nt']
        self.dt = self.parameters['dt']
        self.dir = self.parameters['dir']
        self.r_point = self.parameters['r_point']
        self.point = self.parameters['point']
        self.rpm = self.parameters['rpm']

    def compute(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        dt = self.parameters['dt']
        direction = self.parameters['dir']
        r_point = self.parameters['r_point']
        csdl_model = RotorCSDL3(module=self,
                                mesh=self.mesh,
                                mesh_name=mesh_name,
                                num_blades=num_blades,
                                ns=ns,
                                nc=nc,
                                nt=nt,
                                dt=dt,
                                dir=direction,
                                r_point=r_point,
                                rpm=self.rpm,
                                point=self.point)
        return csdl_model

    # def evaluate(self, h : m3l.Variable, theta : m3l.Variable, blade_angle: m3l.Variable, delta : m3l.Variable):
    # def evaluate(self, h : m3l.Variable, theta : m3l.Variable, blade_angle: m3l.Variable):
    # def evaluate(self, h : m3l.Variable, theta : m3l.Variable):
    def evaluate(self, h : m3l.Variable):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
 
        self.name = mesh_name + '_rotor'
        self.arguments = {
            #'theta' : theta,
            'h' : h,
            #'blade_angle' : blade_angle,
            #'delta' : delta
        }

        mesh_out_vars = []
        for i in range(num_blades):
            mesh_out_vars.append(m3l.Variable(self.name+str(i)+'_out', shape=(nt,nc,ns,3), operation=self))

        mirror_mesh_vars = []
        for i in range(num_blades):
            mirror_mesh_vars.append(m3l.Variable(self.name+str(i)+'_mirror', shape=(nt,nc,ns,3), operation=self))

        # mesh_out_velocities = []
        # for i in range(num_blades):
        #     mesh_out_velocities.append(m3l.Variable(self.name+str(i)+'_out_velocity', shape=(nt,nc-1,ns-1,3), operation=self))

        # mesh_mirror_velocities = []
        # for i in range(num_blades):
        #     mesh_mirror_velocities.append(m3l.Variable(self.name+str(i)+'_mirror_velocity', shape=(nt,nc-1,ns-1,3), operation=self))

        return tuple(mesh_out_vars), tuple(mirror_mesh_vars), # tuple(mesh_out_velocities), tuple(mesh_mirror_velocities)



class RotorCSDL3(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('mesh_name')
        self.parameters.declare('num_blades', default=6)
        self.parameters.declare('ns', default=4)
        self.parameters.declare('nc', default=2)
        self.parameters.declare('nt', default=2)
        self.parameters.declare('dt', default=0.001)
        self.parameters.declare('dir', default=1)
        self.parameters.declare('r_point')
        self.parameters.declare('mesh')
        self.parameters.declare('rpm')
        self.parameters.declare('point')
 
    def define(self):
        mesh_name = self.parameters['mesh_name']
        num_blades = self.parameters['num_blades']
        ns = self.parameters['ns']
        nc = self.parameters['nc']
        nt = self.parameters['nt']
        dt = self.parameters['dt']
        dir = self.parameters['dir']
        r_point = self.parameters['r_point']
        mesh = self.parameters['mesh']
        rpm = self.parameters['rpm']
        point = self.parameters['point']




        # a single blade mesh:
        mesh = self.create_input(mesh_name, shape=(nc,ns,3), val=mesh)*0.3048
        # the center of the rotor disk:
        point = self.create_input(mesh_name + '_point', shape=(3,), val=np.reshape(point, (3,)))*0.3048

        
        delta = self.declare_variable('delta', shape=(3,), val=0)
        #self.print_var(delta)
        delta_expanded = csdl.expand(delta, (nt,nc,ns,3), 'i->abci')



        # blade pitch control
        blade_angle = self.declare_variable('blade_angle', shape=(1,), val=0)
        # self.print_var(blade_angle)
        # blade_axis = self.declare_variable('blade_axis', shape=(3,))*0.3048
        blade_axis = point + np.array([0,1,0])
        normalized_blade_axis = blade_axis/csdl.expand(csdl.pnorm(blade_axis, 2), (3,))
        xb, yb, zb = normalized_blade_axis[0], normalized_blade_axis[1], normalized_blade_axis[2]

        blade_rot_mat = self.create_output('rot_mat', shape=(3,3), val=0)
        blade_cos_theta, blade_sin_theta = csdl.cos(blade_angle), csdl.sin(blade_angle)
        blade_one_minus_cos_theta = 1 - blade_cos_theta
        blade_rot_mat[0,0] = csdl.reshape(blade_cos_theta + xb**2 * blade_one_minus_cos_theta, (1,1))
        blade_rot_mat[0,1] = csdl.reshape(xb * yb * blade_one_minus_cos_theta - zb * blade_sin_theta, (1,1))
        blade_rot_mat[0,2] = csdl.reshape(xb * zb * blade_one_minus_cos_theta + yb * blade_sin_theta, (1,1))
        blade_rot_mat[1,0] = csdl.reshape(yb * xb * blade_one_minus_cos_theta + zb * blade_sin_theta, (1,1))
        blade_rot_mat[1,1] = csdl.reshape(blade_cos_theta + yb**2 * blade_one_minus_cos_theta, (1,1))
        blade_rot_mat[1,2] = csdl.reshape(yb * zb * blade_one_minus_cos_theta - xb * blade_sin_theta, (1,1))
        blade_rot_mat[2,0] = csdl.reshape(zb * xb * blade_one_minus_cos_theta - yb * blade_sin_theta, (1,1))
        blade_rot_mat[2,1] = csdl.reshape(zb * yb * blade_one_minus_cos_theta + xb * blade_sin_theta, (1,1))
        blade_rot_mat[2,2] = csdl.reshape(blade_cos_theta + zb**2 * blade_one_minus_cos_theta, (1,1))

        blade_shifted_mesh = mesh - csdl.expand(point, (nc,ns,3), 'k->ijk')
        blade_rot_mesh = self.create_output('blade_rot_mesh', shape=(nc,ns,3), val=0)
        for i in range(nc):
            for j in range(ns):
                blade_mesh_point = csdl.reshape(blade_shifted_mesh[i,j,:], (3))
                blade_rot_mesh[i,j,:] = csdl.reshape(csdl.matvec(blade_rot_mat, blade_mesh_point), (1,1,3))

        alpha = self.declare_variable('theta', shape=(1,), val=0.)
        h = self.declare_variable('h', shape=(1,))

        

        # the rotation matrix for ground effect stuff:
        rotation_matrix_y = self.create_output('rotation_matrix_y',shape=(3,3),val=0)
        rotation_matrix_y[0,0] = csdl.reshape(csdl.cos(alpha), (1,1))
        rotation_matrix_y[0,2] = csdl.reshape(csdl.sin(alpha), (1,1))
        rotation_matrix_y[1,1] = self.create_input('one',shape=(1,1),val=1)
        rotation_matrix_y[2,0] = csdl.reshape(-csdl.sin(alpha), (1,1))
        rotation_matrix_y[2,2] = csdl.reshape(csdl.cos(alpha), (1,1))


        rad_per_blade = 2*np.pi/num_blades

        rpm = self.create_input('rpm', shape=(1,), val=np.array([rpm]))
        # rpm = self.register_module_input('rpm', shape=(1,), computed_upstream=False)
        rps = rpm/60
        rad_per_sec = rps*2*np.pi
        rad_per_dt = rad_per_sec*dt

        # a single blade mesh: (now moved inputs above)
        #mesh = self.declare_variable(mesh_name, shape=(nc,ns,3), val=mesh)*0.3048
        # the center of the rotor disk:
        #point = self.declare_variable('point', shape=(3,))*0.3048
        # normal vector to the rotor disk:
        vector = self.declare_variable('vector', shape=(3,))*0.3048
        normalized_vector = vector/csdl.expand(csdl.pnorm(vector, 2), (3,))
        x, y, z = normalized_vector[0], normalized_vector[1], normalized_vector[2]


        

        # changed to now use the blade pitch control thingy mesh computed above!
        # shifted_mesh = mesh - csdl.expand(point, (nc,ns,3), 'k->ijk')
        # shifted_mesh = blade_rot_mesh - csdl.expand(point, (nc,ns,3), 'k->ijk')
        shifted_mesh = blade_rot_mesh*1# - csdl.expand(point, (nc,ns,3), 'k->ijk')



        debug_rot_mesh = self.create_output('debug_rot_mesh', shape=(num_blades,nt,nc,ns,3), val=0)
        for i in range(num_blades):
            rot_mesh = self.create_output('rot_mesh'+str(i), shape=(nt,nc,ns,3), val=0)
            set_angle = rad_per_blade*i

            for j in range(nt):
                angle = (set_angle + rad_per_dt*j)*dir

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

                        rot_mesh[j,k,l,:] = csdl.reshape(csdl.matvec(rot_mat, mesh_point), (1,1,1,3))
                        debug_rot_mesh[i,j,k,l,:] = csdl.reshape(csdl.matvec(rot_mat, mesh_point), (1,1,1,1,3))

            rotor = rot_mesh + csdl.expand(point, (nt,nc,ns,3), 'm->ijkm')
            self.register_output(mesh_name + '_rotor'+str(i), rotor)



            # NATIVE ROTOR MIRRORING!!!!!
            mesh_in = 1*rotor
            translated_mesh_points = mesh_in - np.tile(r_point, (nt, nc, ns, 1))
            rotated_mesh_points = self.create_output('rotated_mesh_points_'+str(i), shape=(nt,nc,ns,3), val=0)
            for j in range(nt):
                for k in range(nc):
                    for l in range(ns):
                        eval_point = csdl.reshape(translated_mesh_points[j,k,l,:], (3))
                        rotated_mesh_points[j,k,l,:] = csdl.reshape(csdl.matvec(rotation_matrix_y, eval_point), (1,1,1,3))

            rotated_mesh = rotated_mesh_points + np.tile(r_point, (nt, nc, ns, 1))

            # translate the mesh based on altitude:
            dh = self.create_output('dh'+str(i), shape=(nt,nc,ns,3), val=0)
            dh[:,:,:,2] = csdl.expand(h, (nt,nc,ns,1), 'i->abci')
            mesh_out_intermediate = rotated_mesh + dh

            # translate mesh (DV)
            mesh_out = mesh_out_intermediate + delta_expanded
            self.register_output(mesh_name + '_rotor' + str(i) + '_out', mesh_out)





            # mesh out velocity
            # coll_pts_coords = 0.25/2 * (mesh_out[:,0:nc-1, 0:ns-1, :] +mesh_out[:,0:nc-1, 1:ns, :]) +\
            #                              0.75/2 * (mesh_out[:,1:, 0:ns-1, :]+mesh_out[:,1:, 1:, :])
            # mesh_velocity = self.create_output(f'{mesh_name}_rotor{i}_out_velocity', shape=(nt, nc-1, ns-1, 3), val=0)
            # for ii in range(nt):
            #     if ii == 0: mesh_velocity[ii, :, :, :] = (-3 * coll_pts_coords[ii, :, :, :] + 4 * coll_pts_coords[ii + 1, :, :, :] - coll_pts_coords[ii + 2, :, :, :]) / (2 * dt)
            #     elif ii == (nt - 1): mesh_velocity[ii, :, :, :] = (3 * coll_pts_coords[ii, :, :, :] - 4 * coll_pts_coords[ii - 1, :, :, :] + coll_pts_coords[ii - 2, :, :, :]) / (2 * dt)
            #     else: mesh_velocity[ii, :, :, :] = (coll_pts_coords[ii+1, :, :, :] - coll_pts_coords[ii-1, :, :, :]) / (2 * dt)

            # TODO: also do blade velocities for mirrored meshes

            

            # create the mirrored mesh:
            mirror = self.create_output(mesh_name + '_rotor'+str(i)+'_mirror', shape=(nt,nc,ns,3), val=0)
            mirror[:,:,:,0] = mesh_out[:,:,:,0]
            mirror[:,:,:,1] = mesh_out[:,:,:,1]
            mirror[:,:,:,2] = -1*mesh_out[:,:,:,2]


            # mesh mirror velocity
            # coll_pts_coords_mirror = 0.25/2 * (mirror[:,0:nc-1, 0:ns-1, :] + mirror[:,0:nc-1, 1:ns, :]) +\
            #                              0.75/2 * (mirror[:,1:, 0:ns-1, :] + mirror[:,1:, 1:, :])
            # mesh_velocity_mirror = self.create_output(f'{mesh_name}_rotor{i}_mirror_velocity', shape=(nt, nc-1, ns-1, 3), val=0)
            # for ii in range(nt):
            #     if ii == 0: mesh_velocity_mirror[ii, :, :, :] = (-3 * coll_pts_coords_mirror[ii, :, :, :] + 4 * coll_pts_coords_mirror[ii + 1, :, :, :] - coll_pts_coords_mirror[ii + 2, :, :, :]) / (2 * dt)
            #     elif ii == (nt - 1): mesh_velocity_mirror[ii, :, :, :] = (3 * coll_pts_coords_mirror[ii, :, :, :] - 4 * coll_pts_coords_mirror[ii - 1, :, :, :] + coll_pts_coords_mirror[ii - 2, :, :, :]) / (2 * dt)
            #     else: mesh_velocity_mirror[ii, :, :, :] = (coll_pts_coords_mirror[ii+1, :, :, :] - coll_pts_coords_mirror[ii-1, :, :, :]) / (2 * dt)


        debug_rotor = debug_rot_mesh + csdl.expand(point, (num_blades,nt,nc,ns,3), 'm->ijklm')
        self.register_output('rotor', debug_rotor)



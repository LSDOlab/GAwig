import m3l
import csdl


class ac_expand(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('num_nodes')
        self.parameters.declare('name', default='ac_expand')
    def assign_attributes(self):
        self.num_nodes = self.parameters['num_nodes']
        self.name = self.parameters['name']
    def evaluate(self, ac_states):
        self.assign_attributes()
        num_nodes = self.num_nodes
        self.arguments = {}
        self.arguments['u'] = ac_states['u']
        self.arguments['v'] = ac_states['v']
        self.arguments['w'] = ac_states['w']
        self.arguments['p'] = ac_states['p']
        self.arguments['q'] = ac_states['q']
        self.arguments['r'] = ac_states['r']
        self.arguments['theta'] = ac_states['theta']
        self.arguments['psi'] = ac_states['psi']
        self.arguments['x'] = ac_states['x']
        self.arguments['y'] = ac_states['y']
        self.arguments['z'] = ac_states['z']
        self.arguments['phi'] = ac_states['phi']
        self.arguments['gamma'] = ac_states['gamma']
        # self.arguments['psiw'] = ac_states['psiw']

        outputs = {
            'u' : m3l.Variable('u_expanded', (num_nodes,1), operation=self),
            'v' : m3l.Variable('v_expanded', (num_nodes,1), operation=self),
            'w' : m3l.Variable('w_expanded', (num_nodes,1), operation=self),
            'p' : m3l.Variable('p_expanded', (num_nodes,1), operation=self),
            'q' : m3l.Variable('q_expanded', (num_nodes,1), operation=self),
            'r' : m3l.Variable('r_expanded', (num_nodes,1), operation=self),
            'theta' : m3l.Variable('theta_expanded', (num_nodes,1), operation=self),
            'psi' : m3l.Variable('psi_expanded', (num_nodes,1), operation=self),
            'x' : m3l.Variable('x_expanded', (num_nodes,1), operation=self),
            'y' : m3l.Variable('y_expanded', (num_nodes,1), operation=self),
            'z' : m3l.Variable('z_expanded', (num_nodes,1), operation=self),
            'phi' : m3l.Variable('phi_expanded', (num_nodes,1), operation=self),
            'gamma' : m3l.Variable('gamma_expanded', (num_nodes,1), operation=self),
        }
        return outputs


    def compute(self):
        num_nodes = self.num_nodes
        model = csdl.Model()
        u = model.declare_variable(name='u',  shape=(1,))
        v = model.declare_variable(name='v',  shape=(1,))
        w = model.declare_variable(name='w',  shape=(1,))
        p = model.declare_variable(name='p',  shape=(1,))
        q = model.declare_variable(name='q',  shape=(1,))
        r = model.declare_variable(name='r',  shape=(1,))
        theta = model.declare_variable(name='theta',  shape=(1,))
        psi = model.declare_variable(name='psi',  shape=(1,))
        x = model.declare_variable(name='x',  shape=(1,))
        y = model.declare_variable(name='y',  shape=(1,))
        z = model.declare_variable(name='z',  shape=(1,))
        phi = model.declare_variable(name='phi',  shape=(1,))
        gamma = model.declare_variable(name='gamma',  shape=(1,))

        u_expanded = csdl.expand(u, (num_nodes,1))
        v_expanded = csdl.expand(v, (num_nodes,1))
        w_expanded = csdl.expand(w, (num_nodes,1))
        p_expanded = csdl.expand(p, (num_nodes,1))
        q_expanded = csdl.expand(q, (num_nodes,1))
        r_expanded = csdl.expand(r, (num_nodes,1))
        theta_expanded = csdl.expand(theta, (num_nodes,1))
        psi_expanded = csdl.expand(psi, (num_nodes,1))
        x_expanded = csdl.expand(x, (num_nodes,1))
        y_expanded = csdl.expand(y, (num_nodes,1))
        z_expanded = csdl.expand(z, (num_nodes,1))
        phi_expanded = csdl.expand(phi, (num_nodes,1))
        gamma_expanded = csdl.expand(gamma, (num_nodes,1))

        model.register_output(name='u_expanded', var=u_expanded)
        model.register_output(name='v_expanded', var=v_expanded)
        model.register_output(name='w_expanded', var=w_expanded)
        model.register_output(name='p_expanded', var=p_expanded)
        model.register_output(name='q_expanded', var=q_expanded)
        model.register_output(name='r_expanded', var=r_expanded)
        model.register_output(name='theta_expanded', var=theta_expanded)
        model.register_output(name='psi_expanded', var=psi_expanded)
        model.register_output(name='x_expanded', var=x_expanded)
        model.register_output(name='y_expanded', var=y_expanded)
        model.register_output(name='z_expanded', var=z_expanded)
        model.register_output(name='phi_expanded', var=phi_expanded)
        model.register_output(name='gamma_expanded', var=gamma_expanded)

        return model


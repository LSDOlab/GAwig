import m3l
import csdl


class BreguetRange(m3l.ExplicitOperation): 
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default="breguet_range_equation")
        self.parameters.declare('SFC', types=(int, float))

    def assign_attributes(self):
        self.name = self.parameters['name']

    def evaluate(self, lift, drag, velocity, w1, w2):
        """
        Parameters
        ----------
        dift : m3l.Variable
            UVLM lift provided by wing and tail (or total lift from all lifting surface)

        drag : m3l.Variable
            UVLM total drag

        velocity: m3l.Variable
            Aircraft velocity

        w1 : m3l.Variable
            Weight of the aircraft before cruise

        w2 : m3l.Variable
            Weight of the aircraft after cruise

        """

        self.arguments = {}
        self.arguments['lift'] = lift
        self.arguments['drag'] = drag
        self.arguments['velocity'] = velocity
        self.arguments['w1'] = w1
        self.arguments['w2'] = w2

        range = m3l.Variable('range', shape=(1, ), operation=self)

        return range
    
    def compute(self):
        SFC = self.parameters['SFC']
        csdl_model = BreguetRangeCSDL(
            SFC=SFC,
        )

        return csdl_model
    

class BreguetRangeCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('SFC', types=(int, float))

    def define(self):
        sfc = self.parameters['SFC']
        g = 9.81

        v = self.declare_variable('velocity', shape=(1, ))
        lift = self.declare_variable('lift', shape=(1, ))
        drag = self.declare_variable('drag', shape=(1, ))
        w1 = self.declare_variable('w1', shape=(1, ))
        w2 = self.declare_variable('w2', shape=(1, ))

        r = v / sfc / g * lift/drag * csdl.log(w1 / w2)

        self.register_output('range', r)


if __name__ == '__main__':
    from python_csdl_backend import Simulator
    m3l_model = m3l.Model()

    l = m3l_model.create_input('lift', val=15, shape=(1, ))
    d = m3l_model.create_input('drag', val=1, shape=(1, ))
    v = m3l_model.create_input('velocity', val=120, shape=(1, ))
    w1 = m3l_model.create_input('w1', val=1.2, shape=(1, ))
    w2 = m3l_model.create_input('w2', val=1, shape=(1, ))

    b_range = BreguetRange(SFC=0.2)
    range = b_range.evaluate(lift=l, drag=d, velocity=v, w1=w1, w2=w2)
    m3l_model.register_output(range)

    csdl_model = m3l_model.assemble_csdl()

    sim = Simulator(csdl_model, analytics=True)
    sim.run()

    print(sim['breguet_range_equation.range'])


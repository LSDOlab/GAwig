import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l


class LastNAverage(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('n', types=int, default=1)
        self.parameters.declare('end_offset', types=int, default=0)
        self.parameters.declare('name', default='average_op')

    def assign_attributes(self):
        self.n = self.parameters['n']
        self.name = self.parameters['name']

    def compute(self):
        csdl_model = LastNAverageCSDL(n=self.n, arguments = self.arguments, end_offset=self.parameters['end_offset'])
        return csdl_model

    def evaluate(self, variables):
        self.arguments = {}
        outputs = []
        for variable in variables:
            self.arguments[variable.name] = variable
            outputs.append(m3l.Variable(variable.name + '_ave', variable.shape[1:], operation=self))
        return tuple(outputs)

class LastNAverageCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('n')
        self.parameters.declare('end_offset')
        self.parameters.declare('arguments')
 
    def define(self):
        n = self.parameters['n']
        arguments = self.parameters['arguments']
        end_offset = self.parameters['end_offset']
        
        for argument in arguments.values():
            var = self.declare_variable(name=argument.name, shape=argument.shape)
            var_reshaped = csdl.reshape(var, (var.shape[0],np.prod(var.shape[1:])))
            n_var_reshaped = var_reshaped[var.shape[0]-n-end_offset:var.shape[0]-end_offset,:]
            average_reshaped = csdl.average(n_var_reshaped, axes=0)
            average = csdl.reshape(average_reshaped, var.shape[1:])
            self.register_output(argument.name + '_ave', average)

        


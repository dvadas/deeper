import numpy as np

import theano
from theano import tensor as T

# f(x) = x
def linear(input):
    return input

# f(x) = max(0, x)
def rectifiedLinearUnit(input):
    return T.clip(input, 0, np.inf)

class Layer(object):
    def __init__(self, rng, numInputs, numOutputs, activationFunction):
        maxValue = np.sqrt(6.0) / np.sqrt(numInputs + numOutputs)
        values = rng.uniform(low=-maxValue, high=maxValue, size=(numInputs, numOutputs))
        values = np.array(values, dtype=theano.config.floatX)
        self.weights = theano.shared(value=values, name='W', borrow=True)
            
        values = np.zeros((numOutputs,), dtype=theano.config.floatX)
        self.bias = theano.shared(value=values, name='b', borrow=True)
        
        # For pretraining only
        values = np.zeros((numInputs,), dtype=theano.config.floatX)
        self.biasPrime = theano.shared(value=values, name='bPrime', borrow=True)

        self.activationFunction = activationFunction

    def activate(self, input, weights=None, bias=None, activationFunction=None, reverse=False):
        if weights is None:
            weights = self.weights
        if reverse:
            weights = weights.T

        if bias is None:
            if reverse:
                bias = self.biasPrime
            else:
                bias = self.bias

        if activationFunction is None:
            activationFunction = self.activationFunction

        # result = activationFunction(input, weights, bias)
        output = T.dot(input, weights) + bias
        result = activationFunction(output)
        return result


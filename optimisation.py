import numpy as np

import theano
from theano import tensor as T

class Sgd(object):
    def __init__(self, learningRate=1e-3):
        self.learningRate = learningRate

    def update(self, param, loss):
        gradient = T.grad(loss, param)
        update = param - gradient * self.learningRate
        return [(param, update)]

# Clips an internal range, rather than an external range like clip()
# Example: innerClip([-3,-2,-1,0,1,2,3], 1.5) returns [-3,-2,0,0,0,2,3]
def innerClip(tensor, limit, minLimit=None, newValue=0):
    if minLimit is None:
        minLimit = -limit

    unclipped = (tensor > limit) + (tensor < minLimit)
    return tensor * unclipped

class AdaDelta(object):
    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon

    def update(self, param, loss):
        rho = self.rho
        epsilon = self.epsilon

        values = np.zeros(param.shape.eval(), dtype=theano.config.floatX)
        accumulatedGradient = theano.shared(value=values, name='accGradient', borrow=True)
        values = np.zeros(param.shape.eval(), dtype=theano.config.floatX)
        accumulatedDelta = theano.shared(value=values, name='accDelta', borrow=True)
         
        gradient = T.grad(loss, param)
        newAccGrad = rho * accumulatedGradient + (1 - rho) * gradient * gradient
        # newAccGrad = innerClip(newAccGrad, 1e-16)
        
        numerator = T.sqrt(accumulatedDelta + epsilon)
        denominator = T.sqrt(newAccGrad + epsilon)
        delta = -(numerator / denominator) * gradient
        newAccDelta = rho * accumulatedDelta + (1 - rho) * delta * delta
        # newAccDelta = innerClip(newAccDelta, 1e-16)

        return [
            (accumulatedGradient, newAccGrad),
            (accumulatedDelta, newAccDelta),
            (param, param + delta),
            ]


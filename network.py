import copy
import itertools

import numpy as np

import theano
from theano import tensor as T

from optimisation import AdaDelta
from layer import Layer
from minibatch import MinibatchIterator

def meanSquaredError(predictions, targets):
    squaredError = (predictions - targets) ** 2
    return squaredError.mean(axis=0)

def rootMeanSquaredError(predictions, targets):
    return T.sqrt(meanSquaredError(predictions, targets))

def l1Loss(weights):
    return T.sum(abs(weights))

def l2Loss(weights):
    return T.sum(weights ** 2)

def dropout(rng, input, dropoutRate):
    seed = rng.randint(2 ** 32)
    rngTheano = theano.tensor.shared_randomstreams.RandomStreams(seed)
    probability = 1.0 - dropoutRate
    mask = rngTheano.binomial(n=1, p=probability, size=input.shape, dtype=theano.config.floatX)
    return input * mask

addNoise = dropout

class DeepNeuralNetwork(object):
    def __init__(self, numInputs, layerSizes, activationFunctions, lossFunction=meanSquaredError,
            lossFunctionTest=None, updater=AdaDelta(), noiseRates=None,
            dropoutRates=None, dropoutFunction=dropout, l1=0.0, l2=0.0, randomState=None):
        if lossFunctionTest is None:
            lossFunctionTest = lossFunction
        if noiseRates is None:
            noiseRates = itertools.repeat(0.0)
        if dropoutRates is None:
            dropoutRates = itertools.repeat(0.0)

        if randomState is None:
            self._rng = np.random.RandomState()
        elif isinstance(randomState, np.random.RandomState):
            self._rng = randomState
        else:
            self._rng = np.random.RandomState(randomState)
        
        self._epochPretrain = 0
        self._epoch = 0
        self._loss = None
        self._layers = []
       
        lastSize = numInputs
        for size, activationFunction in zip(layerSizes, activationFunctions):
            layer = Layer(self._rng, lastSize, size, activationFunction)
            self._layers.append(layer)

            lastSize = size

        self._pretrainFuncs = self._makePretrainFuncs(lossFunction, updater, noiseRates)
        
        input = T.matrix('x')
        targets = T.matrix('y')

        self._trainFunc = self._makeTrainFunc(input, targets, lossFunction,
                updater, dropoutRates, l1, l2)
        self._testFunc, self._lossFunc = self._makeTestFuncs(input, targets,
                lossFunctionTest, dropoutRates)
        
    def _params(self):
        for layer in self._layers:
            # Don't include biasPrime, it's only for pretraining
            for param in layer.weights, layer.bias:
                yield param

    def _regularisationLoss(self, multiplier, lossFunc):
        loss = theano.shared(0.0)
        count = theano.shared(0)
        for param in self._params():
            loss += lossFunc(param)
            count += param.shape.prod()
        # Divide by shape product to get a mean, rather than a sum. This way
        # the multiplier isn't affected by layer size changes.
        return multiplier * loss / count
        
    def _makePretrainFuncs(self, lossFunction, updater, noiseRates):
        pretrainFuncs = []
        # Exclude the top output layer, we're not using the target data
        for layer, noiseRate in zip(self._layers[:-1], noiseRates):
            layerInput = T.matrix('x')
            
            if noiseRate > 0.0:
                noisyInput = addNoise(self._rng, layerInput, noiseRate)
            else:
                noisyInput = layerInput
            
            encoded = layer.activate(noisyInput)
            reconstructed = layer.activate(encoded, reverse=True)

            loss = lossFunction(reconstructed, layerInput)
            avgLoss = loss.mean()
            updates = []
            for param in layer.weights, layer.bias, layer.biasPrime:
                updates.extend(updater.update(param, avgLoss))

            pretrainFunc = theano.function(inputs=[layerInput], outputs=loss, updates=updates)
            output = layer.activate(layerInput)
            outputFunc = theano.function(inputs=[layerInput], outputs=output)
            
            pretrainFuncs.append((pretrainFunc, outputFunc))

        return pretrainFuncs

    def _makeTrainFunc(self, input, targets, lossFunction, updater, dropoutRates, l1, l2):
        layerInput = input
        for layer, dropoutRate in zip(self._layers, dropoutRates):
            if dropoutRate > 0.0:
                layerInput = dropout(self._rng, layerInput, dropoutRate)
            output = layer.activate(layerInput)
            
            layerInput = output

        loss = lossFunction(output, targets)
        if l1 != 0.0:
            loss += self._regularisationLoss(l1, l1Loss)
        if l2 != 0.0:
            loss += self._regularisationLoss(l2, l2Loss)

        avgLoss = loss.mean()
        updates = []
        for param in self._params():
            updates.extend(updater.update(param, avgLoss))

        inputs = [input, targets]
        trainFunc = theano.function(inputs=inputs, outputs=loss, updates=updates)
        return trainFunc

    def _makeTestFuncs(self, input, targets, lossFunction, dropoutRates):
        layerInput = input
        for layer, dropoutRate in zip(self._layers, dropoutRates):
            weights = layer.weights
            if dropoutRate > 0.0:
                multiplier = 1.0 - dropoutRate
                weights *= multiplier
            output = layer.activate(layerInput, weights=weights)
            
            layerInput = output

        loss = lossFunction(output, targets)

        testFunc = theano.function(inputs=[input], outputs=output)
        lossFunc = theano.function(inputs=[input, targets], outputs=loss)
        return testFunc, lossFunc

    @property
    def epochPretrain(self):
        return self._epochPretrain
    
    @property
    def epoch(self):
        return self._epoch
    
    @property
    def loss(self):
        return self._loss

    def clone(self):
        return copy.deepcopy(self)

    def _pretrain(self, features, minPretrainImprovement, maxEpochs, verbose):
        layerFeatures = features
        for i, (pretrainFunc, outputFunc) in enumerate(self._pretrainFuncs):
            if verbose:
                print 'pretraining layer', i

            featuresItr = MinibatchIterator(layerFeatures)
            epoch = 0
            prevLoss = np.inf
            while maxEpochs is None or epoch < maxEpochs:
                sumLoss = 0.0
                for batchInput in featuresItr:
                    loss = pretrainFunc(batchInput)
                    sumLoss += loss.mean() * len(batchInput)
                loss = sumLoss / len(features)
                
                if verbose:
                    print 'epoch', self._epochPretrain, 'loss', loss
                
                epoch += 1
                self._epochPretrain += 1
                
                if prevLoss - loss < minPretrainImprovement:
                    break
                prevLoss = loss

            layerFeatures = outputFunc(layerFeatures)
    
    def _train(self, features, targets, maxEpochs, verbose):
        featuresItr = MinibatchIterator(features)
        targetsItr = MinibatchIterator(targets)
        
        for epoch in xrange(maxEpochs):
            self._epoch += 1

            self._loss = np.zeros(len(targets.T))
            for batchInput, batchOutput in zip(featuresItr, targetsItr):
                loss = self._trainFunc(batchInput, batchOutput)
                self._loss += loss * len(batchInput) / len(features)
            
            if verbose:
                print 'epoch', self._epoch,
                if len(self._loss) == 1:
                    print 'loss', self._loss[0]
                else:
                    print 'loss', self._loss, self._loss.mean()

    def fit(self, features, targets, unlabelledFeatures=None,
            minPretrainImprovement=1e-4, maxPretrainEpochs=None, maxEpochs=10, verbose=False):
        if unlabelledFeatures is None:
            pretrainFeatures = features
            # pretrainFeatures will be shuffled when features is shuffled
        else:
            pretrainFeatures = np.vstack((features, unlabelledFeatures))
            self._rng.shuffle(pretrainFeatures)

        state = self._rng.get_state()
        self._rng.shuffle(features)
        # Make sure targets are shuffled the same way by restoring the RNG state
        self._rng.set_state(state)
        self._rng.shuffle(targets)
        
        self._pretrain(pretrainFeatures, minPretrainImprovement, maxPretrainEpochs, verbose)
        self._train(features, targets, maxEpochs, verbose)

    def predict(self, features):
        return self._testFunc(features)

    def score(self, features, targets):
        # Make negative, because higher loss means a lower score
        scores = -self._lossFunc(features, targets)
        return scores


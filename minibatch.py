
class MinibatchIterator(object):
    def __init__(self, data, batchSize=20):
        self._data = data
        self._batchSize = batchSize

    def __iter__(self):
        batchSize = self._batchSize
        for i in xrange(0, len(self._data), batchSize):
            yield self._data[i:i + batchSize]


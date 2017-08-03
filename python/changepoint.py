"""
The implementation borrows heavily from the algorithm documented at
http://www.variation.com/cpa/tech/changepoint.html.  A more detailed
paper is available at http://www.inference.phy.cam.ac.uk/rpa23/papers/rpa-changepoint.pdf.
The basic idea is to find abrupt variations in the generative parameters
of a data sequence.
"""
from __future__ import division

import numpy

def cusum(data):
    avg = numpy.average(data)
    m = numpy.zeros(len(data) + 1, dtype=float)
    m[0] = 0
    for i in range(len(data)):
        m[i+1] = m[i] + (data[i] - avg)
    return m

def bootstrap(data, iterations):
    c = cusum(data)
    sdiff = c.max() - c.min()

    def shuffled(x):
        y = numpy.array(x)
        numpy.random.shuffle(y)
        return y

    n = 0
    for i in range(iterations):
        b = cusum(shuffled(data))
        bdiff = b.max() - b.min()
        n += int(bdiff < sdiff)
    return n

def changepoint(data, confidence=95., iterations=1):
    stack = [(data, 0)]
    while stack:
        data, offset = stack.pop()
        if offset < 0:
            continue
        x = bootstrap(data, iterations)
        p = (x/iterations) * 100.0
        if p > confidence:
            c = cusum(data)
            mx = c.argmax()
            yield mx + offset
            stack.append((data[:mx], offset))
            stack.append((data[mx:], offset+mx-1))

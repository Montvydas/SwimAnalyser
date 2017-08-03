import numpy
import unittest
from changepoint import changepoint

data = numpy.array([
     10.7, 13.0, 11.4, 11.5, 12.5, 14.1, 14.8, 14.1,
     12.6, 16.0, 11.7, 10.6, 10.0, 11.4,  7.9,  9.5,
      8.0, 11.8, 10.5, 11.2,  9.2, 10.1, 10.4, 10.5,
])

class ChangepointTest(unittest.TestCase):

    def test_changepoint(self):
        cps = sorted(set(changepoint(data, confidence=90.)))
        self.assertEquals([10, 11], cps)
        cps = sorted(set(changepoint(data)))
        self.assertEquals([11], cps)

if __name__ == "__main__":
    cps = sorted(set(changepoint(data, confidence=90.)))
    print cps
    # unittest.main()
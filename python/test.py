import numpy as np
from peak_detector import detect_peaks
import pylab

y = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,
       1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,
       2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]

y = [-x for x in y]

print y

# Settings: lag = 30, threshold = 5, influence = 0
lag = 30
threshold = 5
influence = 0

# Run algo with settings from above
result = detect_peaks(y, lag=lag, threshold=threshold, influence=influence)

# Plot result
pylab.subplot(211)
pylab.plot(np.arange(1, len(y)+1), y)
pylab.plot(np.arange(1, len(y)+1), result["avgFilter"], color="cyan", lw=2)
pylab.plot(np.arange(1, len(y)+1), result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)
pylab.plot(np.arange(1, len(y)+1), result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)
pylab.subplot(212)
pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.ylim(-1.5, 1.5)
pylab.show()
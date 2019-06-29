import numpy as np
import smart_open as smart_open_func
import struct

def readHtk(filename):
    """
    Reads the features in a HTK file, and returns them in a 2-D numpy array.
    """

    with smart_open_func.smart_open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
            # sampPeriod and parmKind will be omitted
        #print("hi",sampPeriod,parmKind,sampSize)
        # Read data
        data = struct.unpack(">%df" % (nSamples * sampSize / 4), f.read(nSamples * sampSize))
        return np.array(data).reshape(nSamples, int(sampSize / 4))
def writeHtk(filename, feature, sampPeriod, parmKind):
    """
    Writes the features in a 2-D numpy array into a HTK file.
    """
    with smart_open_func.smart_open(filename, "wb") as f:
        # Write header
        nSamples = feature.shape[0]
        sampSize = feature.shape[1] * 4
        f.write(struct.pack(">iihh", nSamples, sampPeriod, sampSize, parmKind))

        # Write data
        f.write(struct.pack(">%df" % (nSamples * sampSize / 4), *feature.ravel()))
        
# example usage         
# x = np.random.rand(10,3)
# sampPeriod = 100000
# parmKind = 9
# writeHtk('hello.htk', x, sampPeriod, parmKind)
# y = readHtk('hello.htk')

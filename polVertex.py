import numpy as np
class polVertex ():
    def __init__(self,lonlat):
        self.vertex=np.array([lonlat[0], lonlat[1]])
        self.lon = lonlat[0]
        self.lat = lonlat[1]


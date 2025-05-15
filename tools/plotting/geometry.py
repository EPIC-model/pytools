import numpy as np

class Plane:
    def __init__(self, normal, point):
        self.normal = np.asarray(normal)
        self.normal = self.normal/ np.linalg.norm(self.normal, 2)
        self.point = np.asarray(point)
        d = - self.point.dot(self.normal)
        self.plane = np.array([self.normal[0], self.normal[1], self.normal[2], d])
        self.orientation = 'arbitrary'
        #print("Plane: ax + by + cz + d = 0")
        #print('a:', self.plane[0])
        #print('b:', self.plane[1])
        #print('c:', self.plane[2])
        #print('d:', self.plane[3])


class PlaneXY(Plane):
    def __init__(self, z=0.0):
        super().__init__(normal=[0, 0, 1], point=[0, 0, z])
        self.orientation = 'xy'
        self.height = z


class PlaneXZ(Plane):
    def __init__(self, y=0.0):
        super().__init__(normal=[0, 1, 0], point=[0, y, 0])
        self.orientation = 'xz'
        self.height = y


class PlaneYZ(Plane):
    def __init__(self, x=0.0):
        super().__init__(normal=[1, 0, 0], point=[x, 0, 0])
        self.orientation = 'yz'
        self.height = x

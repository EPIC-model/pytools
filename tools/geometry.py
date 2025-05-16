import numpy as np

class Plane:
    def __init__(self, normal: np.ndarray, point: np.ndarray):
        """
        Plane equation ax + by + cz + d = 0
        """
        _normal = np.asarray(normal)
        _point = np.asarray(point)

        if not _normal.size == 3 or not _point.size == 3:
            raise RuntimeError("Normal and/or point not 3D.")

        _normal = _normal / np.linalg.norm(_normal, 2)
        self._d = - _point.dot(_normal)
        self._a = _normal[0]
        self._b = _normal[1]
        self._c = _normal[2]
        self.orientation = 'arbitrary'

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([self._a, self._b, self._c, self._d])

    @property
    def normal(self) -> np.ndarray:
        return np.array([self._a, self._b, self._c])

    def get_x_coords(self, y: np.ndarray | float, z: np.ndarray | float):
        """
        Calculate x on plane and return value(s):
        x = - (d + by + cz) / a
        """
        if self._a == 0.0:
            raise ZeroDivisionError("Unable to obtain 'x' coordinates.")
        return -(self._d + self._b * y + self._c * z) / self._a

    def get_y_coords(self, x: np.ndarray | float, z: np.ndarray | float):
        """
        Calculate y on plane and return value(s):
        y = - (d + ax + cz) / b
        """
        if self._b == 0.0:
            raise ZeroDivisionError("Unable to obtain 'z' coordinates.")
        return -(self._d + self._a * x + self._c * z) / self._b

    def get_z_coords(self, x: np.ndarray | float, y: np.ndarray | float):
        """
        Calculate z on plane and return value(s):
        z = - (d + ax + by) / c
        """
        if self._c == 0.0:
            raise ZeroDivisionError("Unable to obtain 'z' coordinates.")
        return -(self._d + self._a * x + self._b * y) / self._c


class PlaneXY(Plane):
    def __init__(self, z:float = 0.0):
        super().__init__(normal=[0, 0, 1], point=[0, 0, z])
        self.orientation = 'xy'
        self.height = z


class PlaneXZ(Plane):
    def __init__(self, y: float = 0.0):
        super().__init__(normal=[0, 1, 0], point=[0, y, 0])
        self.orientation = 'xz'
        self.height = y


class PlaneYZ(Plane):
    def __init__(self, x: float = 0.0):
        super().__init__(normal=[1, 0, 0], point=[x, 0, 0])
        self.orientation = 'yz'
        self.height = x

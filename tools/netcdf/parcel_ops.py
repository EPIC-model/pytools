import numpy as np
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import EllipseCollection
from tools.plotting.geometry import Plane
from .parcel_dataset import ParcelDataset

#def get_aspect_ratio(self, step, indices=None):
        #if self.is_three_dimensional:
            #raise IOError("Not a 2-dimensional dataset.")

        #V = self.get_data("volume", step=step, indices=indices)
        #b11 = self.get_data("b11", step=step, indices=indices)
        #b12 = self.get_data("b12", step=step, indices=indices)
        #if self._is_compressible:
            #b22 = self.get_data("b22", step=step, indices=indices)
        #else:
            #b22 = self._get_b22(b11, b12, V)
        #a2 = self._get_eigenvalue(b11, b12, b22)
        #return a2 / V * np.pi

def _get_b33(b11: np.ndarray,
             b12: np.ndarray,
             b13: np.ndarray,
             b22: np.ndarray,
             b23: np.ndarray,
             v: np.ndarray) -> np.ndarray:
    return ((0.75 * v / np.pi) ** 2 - b13 * (b12 * b23 - b13 * b22) \
                                    + b11 * b23 ** 2                \
                                    - b12 * b13 * b23)              \
            / (b11 * b22 - b12 ** 2)

def _get_b22(b11: np.ndarray, b12: np.ndarray, volume: np.ndarray) -> np.ndarray:
    return ((volume / np.pi) ** 2 + b12 ** 2) / b11

def _get_eigenvalue(b11: np.ndarray, b12: np.ndarray, b22: np.ndarray) -> np.ndarray:
    return 0.5 * (b11 + b22) + np.sqrt(0.25 * (b11 - b22) ** 2 + b12 ** 2)

def _get_eigenvector(a2: np.ndarray,
                     b11: np.ndarray,
                     b12: np.ndarray,
                     b22: np.ndarray) -> np.ndarray:
    evec = np.array([a2 - b22, b12])
    for i in range(evec.shape[1]):
        if abs(evec[0, i]) + abs(evec[1, i]) == 0.0:
            if b11[i] > b22[i]:
                evec[0, i] = evec[0, i] + np.finfo(np.float64).eps
            else:
                evec[1, i] = evec[1, i] + np.finfo(np.float64).eps

    return evec / np.linalg.norm(evec, 2)

def _get_angle(b11: np.ndarray, b12: np.ndarray, b22: np.ndarray, a2: np.ndarray = None) -> np.ndarray:
    if a2 is None:
        a2 = _get_eigenvalue(b11, b12, b22)
    evec = _get_eigenvector(a2, b11, b12, b22)
    return np.arctan2(evec[1, :], evec[0, :])


def _calculate_intersection_ellipses(dset: ParcelDataset,
                                     step: int,
                                     plane: Plane,
                                     indices : np.ndarray):
    """
    Calculates the ellipses from all ellipsoids intersecting
    with the provided xy-, xz- or yz-plane.
    """

    b11 = dset.get_data(name='B11',        step=step, indices=indices)
    b12 = dset.get_data(name='B12',        step=step, indices=indices)
    b13 = dset.get_data(name='B13',        step=step, indices=indices)
    b22 = dset.get_data(name='B22',        step=step, indices=indices)
    b23 = dset.get_data(name='B23',        step=step, indices=indices)
    v   = dset.get_data(name='volume',     step=step, indices=indices)
    xp  = dset.get_data(name='x_position', step=step, indices=indices)
    yp  = dset.get_data(name='y_position', step=step, indices=indices)
    zp  = dset.get_data(name='z_position', step=step, indices=indices)
    b33 = _get_b33(b11, b12, b13, b22, b23, v)

    n = len(indices)

    b11e = np.empty(n)
    b12e = np.empty(n)
    area = np.empty(n)
    centres = np.empty((n, 2))
    ind = np.empty(n, dtype=int)

    # calculate the rotation matrix R that aligns the normal vector of the
    # plane with the z-axis (note: R^-1 = R^T). We use Rodrigues' rotation
    # formula:
    z = np.array([0.0, 0.0, 1.0])
    w = np.cross(plane.normal, z)
    c = np.dot(plane.normal, z)
    s = np.sqrt(1.0 - c**2)
    W = np.array([[0.0, -w[2], w[1]],
                    [w[2], 0.0, -w[0]],
                    [-w[1], w[0], 0.0]])
    W2 = np.matmul(W, W)
    R = np.eye(3) + W * s + W2 * (1.0 - c)
    iR = np.transpose(R)
    z = None

    # new location of plane
    p_aligned = np.matmul(R, plane.point)
    zplane = p_aligned[2]

    j = 0
    for i in range(n):
        B = np.array([[b11[i], b12[i], b13[i]],
                      [b12[i], b22[i], b23[i]],
                      [b13[i], b23[i], b33[i]]])

        # ellipsoid centre:
        xc = np.array([xp[i], yp[i], zp[i]])

        # rotate ellipsoid and centre:
        B = np.matmul(R, np.matmul(B, iR))
        xc = np.matmul(R, xc)

        # After rotation we make the origin (0, 0, 0) the centre of the ellipsoid
        # --> xc = (0, 0, 0), we must therefore shift the plane:
        z = zplane - xc[2]

        # Calculate the centre of the ellipse (in 3D space):
        xo = np.zeros(3)
        xo[0] = B[0, 2] / B[2, 2] * z
        xo[1] = B[1, 2] / B[2, 2] * z
        xo[2] = z

        if B[2, 2] < z ** 2:
            continue

        # Transform centre back to original coordinate system:
        xo = xo + xc
        xo = np.matmul(iR, xo)

        S = 1.0 - z ** 2 / B[2, 2]


        B2x2 = np.zeros((2,2))
        B2x2[0, 0] = (B[0, 0] * B[2, 2] - B[0, 2] ** 2) * S / B[2, 2]
        B2x2[0, 1] = (B[0, 1] * B[2, 2] - B[0, 2] * B[1, 2])  * S / B[2, 2]
        B2x2[1, 0] = B2x2[0, 1]
        B2x2[1, 1] = (B[1, 1] * B[2, 2] - B[1, 2] ** 2) * S / B[2, 2]

        B3x3 = np.array([[B2x2[0, 0], B2x2[0, 1], 0.0],
                         [B2x2[0, 1], B2x2[1, 1], 0.0],
                         [0.0, 0.0, 1.0]])
        B3x3 = np.matmul(iR, np.matmul(B3x3, R))
        B2x2[0, 0] = B3x3[0, 0]
        B2x2[0, 1] = B3x3[0, 1]
        B2x2[1, 0] = B3x3[0, 1]
        B2x2[1, 1] = B3x3[1, 1]


        if plane.orientation == 'xy':
            centres[j, 0] = xo[0]
            centres[j, 1] = xo[1]
        elif plane.orientation == 'xz':
            centres[j, 0] = xo[0]
            centres[j, 1] = xo[2]

            B2x2[0, 0] = B3x3[0, 0]
            B2x2[0, 1] = B3x3[0, 2]
            B2x2[1, 1] = B3x3[2, 2]
        elif plane.orientation == 'yz':
            centres[j, 0] = xo[1]
            centres[j, 1] = xo[2]
            B2x2[0, 0] = B3x3[1, 1]
            B2x2[0, 1] = B3x3[1, 2]
            B2x2[1, 1] = B3x3[2, 2]
        else:
            raise RuntimeError("The intersection method does not support arbitrary planes yet.")

        b11e[j] = B2x2[0, 0]
        b12e[j] = B2x2[0, 1]
        area[j] = np.pi * np.sqrt(B2x2[1, 1] * B2x2[0, 0] - B2x2[0, 1] ** 2)
        ind[j] = indices[i]
        j = j + 1

    return centres[0:j], b11e[0:j], b12e[0:j], area[0:j], ind[0:j]

def _calculate_projection_ellipses(dset: ParcelDataset,
                                   step: int,
                                   plane: Plane,
                                   indices : np.ndarray):
    """
    Calculates 2D projections of the ellipsoids onto either
    xy-, xz- or yz-plane.
    """
    b11 = dset.get_data(name='B11',    step=step, indices=indices)
    b12 = dset.get_data(name='B12',    step=step, indices=indices)
    b13 = dset.get_data(name='B13',    step=step, indices=indices)
    b22 = dset.get_data(name='B22',    step=step, indices=indices)
    b23 = dset.get_data(name='B23',    step=step, indices=indices)
    v   = dset.get_data(name='volume', step=step, indices=indices)
    b33 = _get_b33(b11, b12, b13, b22, b23, v)

    n = len(indices)
    centres = np.empty((n, 2))

    if plane.orientation == 'xy':
        b33 = 1.0 / b33
        b11_proj = b11 - b13 ** 2 * b33
        b12_proj = b12 - b13 * b23 * b33
        b22_proj = b22 - b23 ** 2 * b33
        centres[:, 0] = dset.get_data(name='x_position', step=step, indices=indices)
        centres[:, 1] = dset.get_data(name='y_position', step=step, indices=indices)
    elif plane.orientation == 'xz':
        b22 = 1.0 / b22
        b11_proj = b11 - b12 ** 2 * b22
        b12_proj = b13 - b12 * b23 * b22
        b22_proj = b33 - b23 ** 2 * b22
        centres[:, 0] = dset.get_data(name='x_position', step=step, indices=indices)
        centres[:, 1] = dset.get_data(name='z_position', step=step, indices=indices)
    elif plane.orientation == 'yz':
        b11 = 1.0 / b11
        b11_proj = b22 - b12 ** 2 * b11
        b12_proj = b23 - b12 * b13 * b11
        b22_proj = b33 - b13 ** 2 * b11
        centres[:, 0] = dset.get_data(name='y_position', step=step, indices=indices)
        centres[:, 1] = dset.get_data(name='z_position', step=step, indices=indices)
    else:
        raise RuntimeError("The projection method does not support arbitrary planes yet.")

    v_proj = np.pi * np.sqrt(b22_proj * b11_proj - b12_proj ** 2)
    return centres, b11_proj, b12_proj, v_proj, indices


def get_ellipses(dset: ParcelDataset,
                 step: int,
                 method: str = 'intersection',
                 plane: Plane = None) -> EllipseCollection:
    """
    Generate ellipses for plotting. Accepts 2D and 3D parcel datasets.
    In case of a 3D parcel dataset, there is the option to get the projected or
    intersected ellipses with a plane.

    Parameters for 3D
    -----------------
    method: 'intersection' or 'projection'
    """

    if not isinstance(dset, ParcelDataset):
        raise TypeError("Dataset 'dset' must be of type ParcelDataset")

    if dset.is_three_dimensional:
        if plane is None:
            raise RuntimeError("A plane instance must be provided.")

        if method not in ['intersection', 'projection']:
            raise RuntimeError("Method must be 'intersection' or 'projection'")

        if not plane.orientation in ('xy', 'xz', 'yz'):
            raise RuntimeError("No arbitrary planes supported yet.")

        dx = dset.extent / dset.ncells

        var = {'yz': 0, 'xz': 1, 'xy': 2}
        dim = {0: 'x', 1: 'y', 2: 'z'}

        # calculate the position of the plane
        j = var[plane.orientation]

        # lower and upper position bounds
        lo = plane.height - dx[j]
        hi = plane.height + dx[j]

        # get indices of parcels satisfying lo <= pos <= hi
        pos = dset.get_data(name=dim[j] + '_position', step=step)
        indices = np.where((pos >= lo) & (pos <= hi))[0]
        pos = None

        if method == 'intersection':
            _centres, _b11, _b12, _v, _indices = _calculate_intersection_ellipses(dset, step, plane, indices)
        elif method == 'projection':
            _centres, _b11, _b12, _v, _indices = _calculate_projection_ellipses(dset, step, plane, indices)

        _b22 = _get_b22(_b11, _b12, _v)

    else:
        _x = dset.get_data("x_position", step=step)
        _y = dset.get_data("z_position", step=step)
        _v = dset.get_data("volume", step=step)
        _b11 = dset.get_data("B11", step=step)
        _b12 = dset.get_data("B12", step=step)
        _b22 = _get_b22(_b11, _b12, _v)
        _centres = np.column_stack((_x, _y))
        _indices = np.arange(0, _x.size)

    a2 = _get_eigenvalue(_b11, _b12, _b22)
    b2 = (_v / np.pi) ** 2 / a2
    angle = _get_angle(_b11, _b12, _b22, a2)

    # 4 Feb 2022
    # https://matplotlib.org/stable/gallery/shapes_and_collections/ellipse_collection.html
    return EllipseCollection(widths=2.0 * np.sqrt(a2),
                             heights=2.0 * np.sqrt(b2),
                             angles=np.rad2deg(angle),
                             units='xy',
                             offsets=_centres), _indices

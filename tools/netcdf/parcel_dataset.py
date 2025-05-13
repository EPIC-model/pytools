import netCDF4 as nc
from .dataset import Dataset
import re
import numpy as np
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import EllipseCollection
#from tools.geometry import xy_plane, xz_plane, yz_plane
import os


class ParcelDataset(Dataset):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def open(self, filename: str):
        super().open(filename)
        self.is_three_dimensional = (self.extent.size == 3)
        self._loaded_step = -1
        self._n_parcel_files = 0

        basename = os.path.basename(filename)
        # 14 Feb 2022
        # https://stackoverflow.com/questions/15340582/python-extract-pattern-matches
        p = re.compile(r"(.*)_(\d*)_parcels.nc")
        result = p.search(basename)
        self._basename = result.group(1)
        self._dirname = os.path.dirname(filename)
        if self._dirname == '':
            self._dirname = '.'
        for ff in os.listdir(self._dirname):
            if self._basename in ff and '_parcels.nc' in ff:
                self._n_parcel_files += 1

    def get_size(self) -> int:
        """
        Get the number of parcel files.
        """
        return self._n_parcel_files

    def get_data(self, name: str, step: int, indices: numpy.ndarray = None) -> np.ndarray:

        self._load_parcel_file(step)

        data = np.array(self._ncfile.variables[name]).squeeze()

        if indices is not None:
            return data[indices, ...]
        else:
            return data

    def get_data_attribute(self, name, attr):
        if not name in self._ncfile.variables.keys():
            raise IOError("Dataset '" + name + "' unknown.")

        if not attr in self._ncfile.variables[name].ncattrs():
            raise IOError("Dataset attribute '" + name + "' unknown.")

        return self._ncfile.variables[name].getncattr(attr)

    def get_num_parcels(self, step):
        self._load_parcel_file(step)
        return self._ncfile.dimensions['n_parcels'].size

    def get_ellipses(self, step, indices=None):
        if self.is_three_dimensional:
            raise IOError("Not a 2-dimensional dataset.")

        x_pos = self.get_data("x_position", step=step, indices=indices)
        z_pos = self.get_data("z_position", step=step, indices=indices)
        position = np.empty((len(x_pos), 2))
        position[:, 0] = x_pos
        position[:, 1] = z_pos
        V = self.get_data("volume", step=step, indices=indices)
        B11 = self.get_data("B11", step=step, indices=indices)
        B12 = self.get_data("B12", step=step, indices=indices)
        if self._is_compressible:
            B22 = self.get_data("B22", step=step, indices=indices)
        else:
            B22 = self._get_B22(B11, B12, V)
        a2 = self._get_eigenvalue(B11, B12, B22)
        angle = self._get_angle(B11, B12, B22, a2)

        b2 = (V / np.pi) ** 2 / a2
        # 4 Feb 2022
        # https://matplotlib.org/stable/gallery/shapes_and_collections/ellipse_collection.html
        return EllipseCollection(widths=2 * np.sqrt(a2),
                                 heights=2 * np.sqrt(b2),
                                 angles=np.rad2deg(angle),
                                 units='xy',
                                 offsets=position)

    def get_ellipses_for_bokeh(self, step, indices=None):
        if self.is_three_dimensional:
            raise IOError("Not a 2-dimensional dataset.")

        x_pos = self.get_data("x_position", step=step, indices=indices)
        z_pos = self.get_data("z_position", step=step, indices=indices)
        V = self.get_data("volume", step=step, indices=indices)
        B11 = self.get_data("B11", step=step, indices=indices)
        B12 = self.get_data("B12", step=step, indices=indices)
        if self._is_compressible:
            B22 = self.get_data("B22", step=step, indices=indices)
        else:
            B22 = self._get_B22(B11, B12, V)
        a2 = self._get_eigenvalue(B11, B12, B22)
        angle = self._get_angle(B11, B12, B22, a2)
        b2 = (V / np.pi) ** 2 / a2
        return (
            x_pos[:],
            z_pos[:],
            2 * np.sqrt(a2[:]),
            2 * np.sqrt(b2[:]),
            angle[:],
        )

    def get_aspect_ratio(self, step, indices=None):
        if self.is_three_dimensional:
            raise IOError("Not a 2-dimensional dataset.")

        V = self.get_data("volume", step=step, indices=indices)
        B11 = self.get_data("B11", step=step, indices=indices)
        B12 = self.get_data("B12", step=step, indices=indices)
        if self._is_compressible:
            B22 = self.get_data("B22", step=step, indices=indices)
        else:
            B22 = self._get_B22(B11, B12, V)
        a2 = self._get_eigenvalue(B11, B12, B22)
        return a2 / V * np.pi

    def _get_B22(self, B11, B12, V):
        return ((V / np.pi) ** 2 + B12 ** 2) / B11

    def _get_B33(self, B11, B12, B13, B22, B23, V):
        return ((0.75 * V / np.pi) ** 2 - B13 * (B12 * B23 - B13 * B22) \
                                        + B11 * B23 ** 2                \
                                        - B12 * B13 * B23)              \
                / (B11 * B22 - B12 ** 2)

    def _get_eigenvalue(self, B11, B12, B22):
        return 0.5 * (B11 + B22) + np.sqrt(0.25 * (B11 - B22) ** 2 + B12 ** 2)

    def _get_eigenvector(self, a2, B11, B12, B22):
        evec = np.array([a2 - B22, B12])
        for i in range(evec.shape[1]):
            if abs(evec[0, i]) + abs(evec[1, i]) == 0.0:
                if B11[i] > B22[i]:
                    evec[0, i] = evec[0, i] + np.finfo(np.float64).eps
                else:
                    evec[1, i] = evec[1, i] + np.finfo(np.float64).eps

        return evec / np.linalg.norm(evec, 2)

    def _get_angle(self, B11, B12, B22, a2=None):
        if a2 is None:
            a2 = self._get_eigenvalue(B11, B12, B22)
        evec = self._get_eigenvector(a2, B11, B12, B22)
        return np.arctan2(evec[1, :], evec[0, :])

    def _get_step_string(self, step):
        return str(step).zfill(10)

    def _load_parcel_file(self, step):
        step = step + 1
        if self._loaded_step == step:
            return
        self._loaded_step = step
        self._ncfile.close()
        s = self._get_step_string(step)
        fname = os.path.join(self._dirname, self._basename + '_' + s + '_parcels.nc')
        self._ncfile = nc.Dataset(fname, "r", format="NETCDF4")

    def calculate_intersection_ellipses(self, step, plane, loc):
        """
        Calculates the ellipses from all ellipsoids intersecting
        with the provided xy-, xz- or yz-plane.

        Parameters
        ----------
        plane   'xy', 'xz' or 'yz'
        loc     grid point
        """
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError("Specified plane '", plane, "' not available.")

        origin = self.get_box_origin()
        extent = self.get_box_extent()
        ncells = self.get_box_ncells()

        dx = extent / ncells

        var = {'yz': 0, 'xz': 1, 'xy': 2}
        dim = {0: 'x', 1: 'y', 2: 'z'}

        # calculate the position of the plane
        j = var[plane]
        p = origin[j] + dx[j] * loc

        if plane == 'xy':
            pl = xy_plane(z=p)
        elif plane ==  'xz':
            pl = xz_plane(y=p)
        else:
            pl = yz_plane(x=p)

        # lower and upper position bounds
        lo = origin[j] + dx[j] * (loc - 1)
        hi = origin[j] + dx[j] * (loc + 1)

        # get indices of parcels satisfying lo <= pos <= hi
        pos = self.get_data(name=dim[j] + '_position', step=step)
        indices = np.where((pos >= lo) & (pos <= hi))[0]
        pos = None

        B11 = self.get_data(name='B11',        step=step, indices=indices)
        B12 = self.get_data(name='B12',        step=step, indices=indices)
        B13 = self.get_data(name='B13',        step=step, indices=indices)
        B22 = self.get_data(name='B22',        step=step, indices=indices)
        B23 = self.get_data(name='B23',        step=step, indices=indices)
        V   = self.get_data(name='volume',     step=step, indices=indices)
        xp  = self.get_data(name='x_position', step=step, indices=indices)
        yp  = self.get_data(name='y_position', step=step, indices=indices)
        zp  = self.get_data(name='z_position', step=step, indices=indices)
        B33 = self._get_B33(B11, B12, B13, B22, B23, V)

        n = len(indices)

        B11e = np.empty(n)
        B12e = np.empty(n)
        area = np.empty(n)
        centres = np.empty((n, 2))
        ind = np.empty(n, dtype=int)

        # calculate the rotation matrix R that aligns the normal vector of the
        # plane with the z-axis (note: R^-1 = R^T). We use Rodrigues' rotation
        # formula:
        z = np.array([0.0, 0.0, 1.0])
        w = np.cross(pl.normal, z)
        c = np.dot(pl.normal, z)
        s = np.sqrt(1.0 - c**2)
        W = np.array([[0.0, -w[2], w[1]],
                      [w[2], 0.0, -w[0]],
                      [-w[1], w[0], 0.0]])
        W2 = np.matmul(W, W)
        R = np.eye(3) + W * s + W2 * (1.0 - c)
        iR = np.transpose(R)
        z = None

        # new location of plane
        p_aligned = np.matmul(R, pl.point)
        zplane = p_aligned[2]

        j = 0
        for i in range(n):
            B = np.array([[B11[i], B12[i], B13[i]],
                          [B12[i], B22[i], B23[i]],
                          [B13[i], B23[i], B33[i]]])

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


            if plane == 'xy':
                centres[j, 0] = xo[0]
                centres[j, 1] = xo[1]
            elif plane == 'xz':
                centres[j, 0] = xo[0]
                centres[j, 1] = xo[2]

                B2x2[0, 0] = B3x3[0, 0]
                B2x2[0, 1] = B3x3[0, 2]
                B2x2[1, 1] = B3x3[2, 2]
            elif plane == 'yz':
                centres[j, 0] = xo[1]
                centres[j, 1] = xo[2]
                B2x2[0, 0] = B3x3[1, 1]
                B2x2[0, 1] = B3x3[1, 2]
                B2x2[1, 1] = B3x3[2, 2]

            B11e[j] = B2x2[0, 0]
            B12e[j] = B2x2[0, 1]
            area[j] = np.pi * np.sqrt(B2x2[1, 1] * B2x2[0, 0] - B2x2[0, 1] ** 2)
            ind[j] = indices[i]
            j = j + 1

        return centres[0:j], B11e[0:j], B12e[0:j], area[0:j], ind[0:j]


    def get_intersection_ellipses(self, step, plane, loc, use_bokeh=False):

        centres, B11, B12, V, indices = calculate_intersection_ellipses(step, plane, loc)

        B22 = self._get_B22(B11, B12, V)
        a2 = self._get_eigenvalue(B11, B12, B22)
        angle = self._get_angle(B11, B12, B22, a2)
        b2 = (V / np.pi) ** 2 / a2

        if use_bokeh:
            return (
                centres[:, 0],
                centres[:, 1],
                2 * np.sqrt(a2[:]),
                2 * np.sqrt(b2[:]),
                angle[:]), indices
        else:
            return EllipseCollection(widths=2.0 * np.sqrt(a2),
                                     heights=2.0 * np.sqrt(b2),
                                     angles=angle,
                                     units='xy',
                                     offsets=centres), indices


    def calculate_projected_ellipses(self, step, plane, loc):
        """
        Calculates 2D projections of the ellipsoids onto either
        xy-, xz- or yz-plane.

        Parameters
        ----------
        plane   'xy', 'xz' or 'yz'
        loc     grid point
        """
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError("Specified plane '", plane, "' not available.")

        origin = self.get_box_origin()
        extent = self.get_box_extent()
        ncells = self.get_box_ncells()

        dx = extent / ncells

        dim = {0: 'x', 1: 'y', 2: 'z'}

        j = 0
        if plane == 'xz':
            j = 1
        elif plane == 'xy':
            j = 2

        # lower and upper position bounds
        lo = origin[j] + dx[j] * (loc - 1)
        hi = origin[j] + dx[j] * (loc + 1)

        # get indices of parcels satisfying lo < pos < hi
        pos = self.get_data(name=dim[j] + '_position', step=step)
        indices = np.where((pos >= lo) & (pos <= hi))[0]
        pos = None


        B11 = self.get_data(name='B11',    step=step, indices=indices)
        B12 = self.get_data(name='B12',    step=step, indices=indices)
        B13 = self.get_data(name='B13',    step=step, indices=indices)
        B22 = self.get_data(name='B22',    step=step, indices=indices)
        B23 = self.get_data(name='B23',    step=step, indices=indices)
        V   = self.get_data(name='volume', step=step, indices=indices)
        B33 = self._get_B33(B11, B12, B13, B22, B23, V)

        if dim[j] == 'x':
            # yz-plane
            B11 = 1.0 / B11
            B11_proj = B22 - B12 ** 2 * B11
            B12_proj = B23 - B12 * B13 * B11
            B22_proj = B33 - B13 ** 2 * B11
            pos_1 = self.get_data(name='y_position', step=step, indices=indices)
            pos_2 = self.get_data(name='z_position', step=step, indices=indices)
        elif dim[j] == 'y':
            # xz-plane
            B22 = 1.0 / B22
            B11_proj = B11 - B12 ** 2 * B22
            B12_proj = B13 - B12 * B23 * B22
            B22_proj = B33 - B23 ** 2 * B22
            pos_1 = self.get_data(name='x_position', step=step, indices=indices)
            pos_2 = self.get_data(name='z_position', step=step, indices=indices)
        else:
            # xy-plane
            B33 = 1.0 / B33
            B11_proj = B11 - B13 ** 2 * B33
            B12_proj = B12 - B13 * B23 * B33
            B22_proj = B22 - B23 ** 2 * B33
            pos_1 = self.get_data(name='x_position', step=step, indices=indices)
            pos_2 = self.get_data(name='y_position', step=step, indices=indices)

        V_proj = np.pi * np.sqrt(B22_proj * B11_proj - B12_proj ** 2)
        return pos_1, pos_2, B11_proj, B12_proj, V_proj, indices



    #def _get_plane(self, data, plane, loc):
        #"""
        #Assumes the data in (x, y, z) ordering.
        #"""
        #if plane not in ['xy', 'xz', 'yz']:
            #raise ValueError("Specified plane '", plane "' not available.")

        #if plane == 'xy':
            #pl_data = data[:, :, loc]
        #elif plane == 'xz':
            #pl_data = data[:, loc, :]
        #else:
            ##plane = 'yz':
            #pl_data = data[loc, :, :]
        #return pl_data

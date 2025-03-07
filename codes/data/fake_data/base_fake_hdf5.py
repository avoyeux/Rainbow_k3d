"""
Parent class for the creation of the fake cubes.
"""

# IMPORTs
import os
import scipy

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field

# IMPORTs personal
from common import config

# API public
__all__ = ['BaseFakeHDF5']



@dataclass(slots=True, repr=False, eq=False)
class FakeCubeBorders:
    """
    To store the border information of the cube, be it in cartesian, in spherical or in indexes.
    """

    # CUBEs info
    dx: float
    xt: tuple[float, float]
    yt: tuple[float, float]
    zt: tuple[float, float]
    increase_factor: float
    
    # BORDERs spherical
    radius: tuple[float, float] = field(init=False)
    theta: tuple[float, float] = field(init=False)
    phi: tuple[float, float] = field(init=False)

    # BORDERs indexes
    max_indexes: tuple[int, int, int] = field(init=False)

    def __post_init__(self) -> None:
        """
        To update the cartesian and spherical border value given the increase factor.
        """

        # BORDERs increase
        self._border_increase()

        # BORDERs spherical
        self._spherical_borders()

        # BORDERs indexes
        self.max_indexes = (
            int((max(self.xt) - min(self.xt)) // self.dx),
            int((max(self.yt) - min(self.yt)) // self.dx),
            int((max(self.zt) - min(self.zt)) // self.dx),
        )

    def _border_increase(self) -> None:
        """
        To increase the borders of the cube.
        """

        # BORDERs range
        x_range = max(self.xt) - min(self.xt)
        y_range = max(self.yt) - min(self.yt)
        z_range = max(self.zt) - min(self.zt)

        # BORDERs increase
        self.xt = (
            min(self.xt) - x_range * (self.increase_factor - 1) / 2,
            max(self.xt) + x_range * (self.increase_factor - 1) / 2,
        )
        self.yt = (
            min(self.yt) - y_range * (self.increase_factor - 1) / 2,
            max(self.yt) + y_range * (self.increase_factor - 1) / 2,
        )
        self.zt = (
            min(self.zt) - z_range * (self.increase_factor - 1) / 2,
            max(self.zt) + z_range * (self.increase_factor - 1) / 2,
        )

    def _spherical_borders(self) -> None:
        """
        To get the spherical borders of the cube.
        """

        # BORDERs spherical
        self.radius = (
            np.sqrt(min(self.xt)**2 + min(self.yt)**2 + min(self.zt)**2),
            np.sqrt(max(self.xt)**2 + max(self.yt)**2 + max(self.zt)**2),
        )
        self.theta = (
            np.arccos(min(self.zt) / min(self.radius)),
            np.arccos(max(self.zt) / max(self.radius)),
        )
        self.phi = (
            np.arctan2(min(self.yt), min(self.xt)),
            np.arctan2(max(self.yt), max(self.xt)),
        )


class BaseFakeHDF5:
    """
    Parent class for the creation of the fake cubes.
    """

    def __init__(
            self,
            angle_step: float,
            sphere_radius: tuple[float, float],
            torus_radius: tuple[float, float],
            increase_factor: float,
        ) -> None:
        """
        To initialize the fake cube creation.

        Args:
            angle_step (float): the step in rad for the angles.
            sphere_radius (tuple[float, float]): the min and max radius of the sphere in km.
            increase_factor (float): the factor to increase the cube borders.
        """
        
        # ATTRIBUTES from arguments
        self.angle_step = angle_step  # in rad
        self.sphere_radius = sphere_radius  # in km
        self.torus_radius = torus_radius  # the main and width radius of the torus in km
        self.increase_factor = increase_factor

        # PATHs setup
        self.paths = self.paths_setup()

        # DATA fetch
        self.cube_info = self._get_defaults()

    def paths_setup(self) -> dict[str, str]:
        """
        To format the paths for the different directories and data files.

        Returns:
            dict[str, str]: the formatted paths.
        """

        # PATHs formatting
        paths = {
            'cubes': config.path.dir.data.cubes.karine,
        }
        return paths
    
    def _get_defaults(self) -> FakeCubeBorders:
        """
        To get the cube information and process the borders in indexes and spherical coordinates.

        Returns:
            FakeCubeBorders: the cube information.
        """

        first_cube = scipy.io.readsav(os.path.join(self.paths['cubes'], 'cube000.save'))
        info = FakeCubeBorders(
            dx=float(first_cube.dx),
            xt=(float(first_cube.xt_min), float(first_cube.xt_max)),
            yt=(float(first_cube.yt_min), float(first_cube.yt_max)),
            zt=(float(first_cube.zt_min), float(first_cube.zt_max)),
            increase_factor=self.increase_factor,
        )
        return info

    def fake_sphere_surface(self) -> np.ndarray:
        """
        To get the cartesian coordinates of the sphere surface.

        Returns:
            np.ndarray: the cartesian coordinates of the sphere surface.
        """
        
        # COORDs spherical
        radius = np.arange(min(self.sphere_radius), max(self.sphere_radius), self.cube_info.dx)
        phi = self.modulo_angle(self.cube_info.phi, step_coef=0.5)  # ? why not -np.pi?
        theta = self.modulo_angle(self.cube_info.theta)
        radius, phi, theta = np.meshgrid(radius, phi, theta)

        # COORDs cartesian
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.stack((x.ravel(), y.ravel(), z.ravel()), axis=0)

    def modulo_angle(self, angle: tuple[float, float], step_coef: float = 1.) -> np.ndarray:
        """
        To make sure the direction of the angle is always the right one.
        This translate to making sure that the modulo of the angle is taken from the right 
        reference.

        Args:
            angle (tuple[float, float]): the min and max angle in rad.
            step_coef (float, optional): the coefficient to multiply the step. Defaults to 1..

        Returns:
            np.ndarray: the corresponding angle in rad.
        """

        # MODULO from 0 to 2 * np.pi
        angle_array = np.array(angle)
        reference = angle[1]
        new_angle = (angle_array - reference) % (2 * np.pi)
        
        # ANGLEs range
        angles = np.arange(min(new_angle), max(new_angle), self.angle_step * step_coef)
        angles += reference
        return angles % (2 * np.pi)

    def fake_torus(self) -> np.ndarray:
        """
        To create a fake torus data in cartesian reprojected carrington coordinates.

        Returns:
            np.ndarray: the torus coordinates as (x, y, z) in km.
        """

        # COORDs range
        phi = np.arange(0, 2 * np.pi, self.angle_step / 2)
        theta = np.arange(0, 2 * np.pi, self.angle_step)
        theta, phi = np.meshgrid(theta, phi)

        # COORDs cartesian in km
        x = (self.torus_radius[0] + self.torus_radius[1] * np.cos(phi)) * np.cos(theta)
        y = (self.torus_radius[0] + self.torus_radius[1] * np.cos(phi)) * np.sin(theta)
        z = self.torus_radius[1] * np.sin(phi)
        return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)
    
    def fake_cube(self) -> np.ndarray:
        """
        To create a fake cube data in cartesian reprojected carrington coordinates.

        Returns:
            np.ndarray: the cube coordinates as (x, y, z) in km.
        """

        # COORDs range
        x_range = np.arange(0, 100 * self.cube_info.dx, self.cube_info.dx)
        y_range = np.arange(0, 100 * self.cube_info.dx, self.cube_info.dx)
        z_range = np.arange(0, 100 * self.cube_info.dx, self.cube_info.dx)

        # COORDs values
        x_positions = x_range + min(self.cube_info.xt)
        y_positions = y_range + min(self.cube_info.yt)
        z_positions = z_range + min(self.cube_info.zt)

        # FILL VOLUME
        X, Y, Z = np.meshgrid(x_positions, y_positions, z_positions, indexing='ij')
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)
    
    def fake_3d_cross(self) -> np.ndarray:
        """
        Creates a 3D cross with 6 branches.

        Returns:
            np.ndarray: The coordinates of the 3D cross in cartesian reprojected Carrington
                coordinates.
        """

        # COORDs range
        half_length = 200
        half_thickness = 6

        axis_range = np.arange(
            - half_length * self.cube_info.dx,
            (half_length + 1) * self.cube_info.dx,
            self.cube_info.dx,
        )
        center = np.array([
            min(self.cube_info.xt) + (50 * self.cube_info.dx),
            min(self.cube_info.yt) + (50 * self.cube_info.dx),
            min(self.cube_info.zt) + (50 * self.cube_info.dx),
        ])

        # BRANCHEs 
        x_branch = np.stack([
            axis_range,
            np.zeros(axis_range.shape),
            np.zeros(axis_range.shape),
        ], axis=0) + center  # ! not sure if I need to add center[0], center[1], ... for each axis
        y_branch = np.stack([
            np.zeros(axis_range.shape),
            axis_range,
            np.zeros(axis_range.shape),
        ], axis=0) + center
        z_branch = np.stack([
            np.zeros(axis_range.shape),
            np.zeros(axis_range.shape),
            axis_range,
        ], axis=0) + center

        # THICKNESS setup
        thickness_range = np.arange(
            - half_thickness * self.cube_info.dx,
            (half_thickness + 1) * self.cube_info.dx,
            self.cube_info.dx,
        )
        A, B = np.meshgrid(thickness_range, thickness_range)
        translations = np.stack([A.ravel(), B.ravel()], axis=0).T

        # BRANCHEs
        all_x_branches = np.copy(x_branch)
        all_y_branches = np.copy(y_branch)
        all_z_branches = np.copy(z_branch)
        for (a, b) in translations:
            all_x_branches = np.concatenate([
                all_x_branches,
                self.cross_branch_tickness(indexes=(1, 2), branch=x_branch, vals=(a, b))
            ], axis=1)
            all_y_branches = np.concatenate([
                all_y_branches,
                self.cross_branch_tickness(indexes=(0, 2), branch=y_branch, vals=(a, b))
            ], axis=1)
            all_z_branches = np.concatenate([
                all_z_branches,
                self.cross_branch_tickness(indexes=(0, 1), branch=z_branch, vals=(a, b))
            ], axis=1)

        # DUPLICATEs filtering
        all_branches = np.concatenate([all_x_branches, all_y_branches, all_z_branches], axis=1)
        return all_branches

    def cross_branch_tickness(
            self,
            indexes: tuple[int, int],
            branch: np.ndarray,
            vals: tuple[float, float],
        ) -> np.ndarray:
        """
        To add the thickness to the branches of the 3D cross.

        Args:
            indexes (list[int]): the indexes of the axes to add the thickness.
            branch (np.ndarray): the branch to add the thickness.
            vals (tuple[float, float]): the thickness values.

        Returns:
            np.ndarray: the branch with the thickness added.
        """

        new_branch = np.copy(branch)
        new_branch[indexes[0]] = branch[indexes[0]] + vals[0]
        new_branch[indexes[1]] = branch[indexes[1]] + vals[1]
        return new_branch

    def to_index(self, coords: np.ndarray) -> np.ndarray:
        """
        To convert the cartesian coordinates to cube indexes.

        Args:
            coords (np.ndarray): the cartesian coordinates.

        Returns:
            np.ndarray: the cube indexes.
        """

        # COORDs to cube
        coords[0] -= min(self.cube_info.xt)
        coords[1] -= min(self.cube_info.yt)
        coords[2] -= min(self.cube_info.zt)

        # BIN values
        coords //= self.cube_info.dx

        # FILTER values
        x_filter = (coords[0] >= 0) & (coords[0] < self.cube_info.max_indexes[0])
        y_filter = (coords[1] >= 0) & (coords[1] < self.cube_info.max_indexes[1])
        z_filter = (coords[2] >= 0) & (coords[2] < self.cube_info.max_indexes[2])
        coords = coords[:, x_filter & y_filter & z_filter]

        # UNIQUE values
        coords = np.unique(coords.astype(int), axis=1)
        return coords

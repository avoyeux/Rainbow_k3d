"""
To create and format the needed information for the final warped image integration.
"""

# IMPORTs
import os
import scipy

# IMPORTs alias
import numpy as np

# IMPORTs sub
import matplotlib.pyplot as plt

# IMPORTs personal
from common import Decorators, config
from codes.projection.helpers.warp_sdo_image import WarpSdoImage, EnvelopeProcessing
from codes.projection.format_data import (
    ImageBorders, FitWithEnvelopes, EnvelopeInformation, WarpedInformation,
)

# TYPE ANNOTATIONs
from typing import Literal, TypeGuard

# API public
__all__ = ['AllWarpedTreatment']

"""
! need to pay attention to the ordering of the data as I might be changing it in some of the codes
! most likely need to compute the middle curve of the envelope each time.
"""



class AllWarpedTreatment:
    """
    To create and format all the metadata and data needed to create the final warped SDO plot.

    Raises:
        ValueError: if the envelopes information is not provided.
    """

    def __init__(
            self,
            sdo_image: np.ndarray,
            date: str,
            integration_time: int | str | None,
            fit_n_envelopes: FitWithEnvelopes,
            borders: ImageBorders,
            pixel_interpolation_order: int = 3,
            nb_of_points: int = 1280,
            integration_type: Literal['mean', 'median'] = 'mean',
        ) -> None:
        """
        To warp the SDO image and keep the relevant information.
        The warped information is stored in the warped_information attribute.

        Args:
            sdo_image (np.ndarray): the SDO image to be warped.
            date (str): the date of the SDO image.
            integration_time (int | str | None): the integration time of the SDO image.
            fit_n_envelopes (FitWithEnvelopes): the fit and envelope information.
            borders (ImageBorders): the borders of the SDO image.
            pixel_interpolation_order (int, optional): the interpolation order used when warping
                the SDO image. Defaults to 3.
            nb_of_points (int, optional): the number of points used to compute the envelope.
                Furthermore, is will also directly define the number of pixels in the resulting
                warped image. Defaults to 1280.
            integration_type (Literal['mean', 'median'], optional): the integration method used
                when converting the warped 2D SDO image to a 1D array (for the final warped
                integration figure). Defaults to 'mean'.

        Raises:
            ValueError: if the envelope information doesn't exist.
        """

        # ATTRIBUTEs
        self.date = date
        self.integration_time = integration_time
        self.fit_order = fit_n_envelopes.fit_order
        self.integration_type: Literal['mean', 'median'] = integration_type

        # CHECKs inputs
        if not self._validate_envelopes(fit_n_envelopes.envelopes):
            raise ValueError(
                f"Need the envelope information inside {self.__class__.__name__}."
            )
        
        # WARP image
        instance = WarpSdoImage(
            sdo_image=sdo_image,
            borders=borders,
            pixel_interpolation_order=pixel_interpolation_order,
            envelopes=[fit_n_envelopes.envelopes.upper, fit_n_envelopes.envelopes.lower],
            nb_of_points=nb_of_points,
        )
        self.warped_image = instance.warped_image

        # CURVEs processed
        self.processed_envelopes = instance.envelopes_processed
        self.processed_middle = EnvelopeProcessing(
            polar_r=fit_n_envelopes.envelopes.middle.polar_r,
            polar_theta=fit_n_envelopes.envelopes.middle.polar_theta,
            nb_of_points=nb_of_points,
        )
        self.processed_fit = np.stack(
            [fit_n_envelopes.fit_polar_r, fit_n_envelopes.fit_polar_theta],
            axis=1,
        )  # ? is it already properly processed ?

        # ANGLEs
        self.angles = self.get_closest_angles(fit_n_envelopes.fit_angles)

        # RUN
        self._warped_information = self._format_warped_information()

        # TEST plot
        if config.run.test_plots: self.test_plot_angles(fit_angles=fit_n_envelopes.fit_angles)

    @property
    def warped_information(self) -> WarpedInformation: return self._warped_information
        
    def _validate_envelopes(
            self,
            envelopes: EnvelopeInformation | None,
        ) -> TypeGuard[EnvelopeInformation]:
        """
        To check that the envelopes data exist.
        It's mainly for the static type checking.

        Args:
            envelopes (EnvelopeInformation | None): the inputted envelopes information.

        Returns:
            TypeGuard[EnvelopeInformation]: True if the envelopes information is not None.
        """

        return envelopes is not None

    def normalise_coords(
            self,
            query_points: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        """  # todo update docstring
        To normalise the fit coordinates so that they are between 0 and 1. This is done so that the
        nearest neighbour search is not biased by the scale of the coordinates.

        Args:
            query_points (np.ndarray): the coordinates of the middle curve for which we want to
                find the closest fit coordinates (to get the corresponding angles).

        Returns:
            tuple[np.ndarray, np.ndarray]: the normalised coordinates of the fit and the middle
                curve.
        """

        # NORMALISE fit
        min_vals = np.min(self.processed_fit, axis=0, keepdims=True)
        max_vals = np.max(self.processed_fit, axis=0, keepdims=True)
        fit_coords = (self.processed_fit - min_vals) / (max_vals - min_vals)

        # CURVE middle change
        middle_coords = (query_points - min_vals) / (max_vals - min_vals)
        return fit_coords, middle_coords

    def get_closest_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        To get the closest angles to the processed fit.

        Args:
            angles (np.ndarray): the angles to be used.

        Returns:
            np.ndarray: the closest angles to the middle curve.
        """

        # COORDs 'normalisation'
        fit_coords, query_points = self.normalise_coords(
            query_points=np.stack([
                self.processed_middle.polar_r, self.processed_middle.polar_theta
            ], axis=1),
        )

        # ANGLEs closest to fit
        tree = scipy.spatial.cKDTree(fit_coords)
        dist, closest_indices = tree.query(query_points, k=1)
        return angles[closest_indices]
    
    def coords_to_cartesian(self, polar_r: np.ndarray, polar_theta: np.ndarray) -> np.ndarray:
        """
        To convert the polar coordinates to cartesian coordinates.

        Args:
            polar_r (np.ndarray): the polar radial coordinates in km.
            polar_theta (np.ndarray): the polar angular coordinates in degrees.

        Returns:
            np.ndarray: the cartesian coordinates in km.
        """
        
        # COORDs cartesian
        x = polar_r * np.cos(np.deg2rad(polar_theta))
        y = polar_r * np.sin(np.deg2rad(polar_theta))
        return np.stack([x, y], axis=0)
    
    def get_curve_distance(self, coords: np.ndarray) -> np.ndarray:
        """
        To compute the distance along the curve given the cartesian coordinates.

        Args:
            coords (np.ndarray): the cartesian coordinates of the curve.

        Returns:
            np.ndarray: the corresponding distance along the curve.
        """

        # COORDs km
        coords /= 1e3

        # DISTANCE curve path
        distance = np.empty((coords.shape[1],), dtype='float64')
        distance[0] = 0
        for i in range(1, distance.size):
            distance[i] = distance[i - 1] + np.sqrt(
                (coords[0, i] - coords[0, i - 1])**2 + (coords[1, i] - coords[1, i - 1])**2
            )
        return distance # in km

    def _format_warped_information(self) -> WarpedInformation:
        """
        To format the warped information.
        To actually access the data from outside, use the warped_information property.

        Returns:
            WarpedInformation: the formatted warped information.
        """

        middle_cartesian = self.coords_to_cartesian(
            polar_r=self.processed_middle.polar_r,
            polar_theta=self.processed_middle.polar_theta,
        )
        middle_distance = self.get_curve_distance(coords=middle_cartesian)

        information = WarpedInformation(
            name='warped information',
            date=self.date,
            fit_order=self.fit_order,
            arc_length=middle_distance,
            integration_time=(
                self.integration_time
                if isinstance(self.integration_time, int) else None
            ),
            integration_type=self.integration_type,
            angles=self.angles,
            warped_values=self.warped_image,
        )
        return information

    def test_plot_angles(self, fit_angles: np.ndarray) -> None:
        """
        To plot the angles and the fitted angles.
        This is only for testing purposes.

        Args:
            fit_angles (np.ndarray): the fitted angles to be plotted.
        """

        filename = f"angles_{self.date}.png"
        plt.figure(figsize=(18, 5))
        plt.plot(
            self.angles,
            label='angles',
            color='red',
        )
        plt.plot(
            fit_angles,
            label='fit angles',
            color='blue',
        )
        plt.title('Angles')
        plt.xlabel('Angle index')
        plt.ylabel('Angle (rad)')
        plt.legend()
        plt.savefig(os.path.join(config.path.dir.data.temp, filename), dpi=500)
        plt.close()

        print(f"TEST - {filename}")

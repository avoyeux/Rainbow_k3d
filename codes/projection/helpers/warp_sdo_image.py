"""
To store functions to warp the SDO image in between a given envelope.
This is to try and recreate the analysis made by Dr. Auchere in his Coronal Monsoon paper.
"""

# IMPORTs
import scipy

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field

# IMPORTs personal
from common import Decorators
from codes.data.polynomial_fit.base_fit_processing import BaseFitProcessing
from codes.projection.helpers.dataclasses.projection_dataclasses import FitEnvelopes



@dataclass(slots=True, eq=False, repr=False)
class EnvelopeProcessing(BaseFitProcessing):
    """
    To process the envelope data so that you can get the interpolation function and use it to warp
    the SDO image between the two envelopes.
    """

    # INTERPOLATION
    polar_r_interp: scipy.interpolate.interp1d = field(init=False)
    polar_theta_interp: scipy.interpolate.interp1d = field(init=False)

    def __post_init__(self) -> None:

        # COORDs re-ordered (for the cumulative distance)
        self.reorder_data()

        # COORDs normalised
        self.normalise_coords()

        # DISTANCE cumulative
        self.cumulative_distance_normalised()

        # COORDs uniform
        self.uniform_coords()

    def reorder_data(self) -> None:
        """
        To reorder the data so that the cumulative distance is calculated properly for the fit.
        That means that the first axis to order should be the polar theta one.
        """

        # RE-ORDER
        coords = np.stack([self.polar_theta, self.polar_r], axis=0)

        # SORT on first axis
        sorted_indexes = np.lexsort(coords[::-1])  # lexsort sorts the last axis first
        sorted_coords = coords[:, sorted_indexes]

        # COORDs update
        self.polar_theta, self.polar_r = sorted_coords
    
    def uniform_coords(self) -> None:
        """
        To uniformly space the coordinates on the curve.
        """

        # INTERPOLATE
        self.polar_r_interp = scipy.interpolate.interp1d(
            x=self.cumulative_distance,
            y=self.polar_r,
            kind='cubic',
        )
        self.polar_theta_interp = scipy.interpolate.interp1d(
            x=self.cumulative_distance,
            y=self.polar_theta,
            kind='cubic',
        )

    def get_coords(self, cumulative_distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        To get the coordinates of the envelope at a given cumulative distance.

        Args:
            cumulative_distance (np.ndarray): the cumulative distance at which to get the
                coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray]: the polar coordinates of the envelope for the given
                cumulative distance.
        """
    
        polar_r_coords = self.polar_r_interp(cumulative_distance)
        polar_theta_coords = self.polar_theta_interp(cumulative_distance)
        return polar_r_coords, polar_theta_coords


class WarpSdoImage:
    """
    To warp the SDO image in between the two fit envelopes.
    This is done to try and recreate the analysis made by Dr. Auchere in his Coronal Monsoon paper.
    """

    def __init__(
            self,
            sdo_image: np.ndarray,
            extent: tuple[float, float, float, float],
            image_shape: tuple[int, int],
            pixel_interpolation_order: int,
            envelopes: list[FitEnvelopes],
            nb_of_points: int,
        ) -> None:

        # ATTRIBUTEs
        self.sdo_image = sdo_image
        self.extent = extent
        self.image_shape = image_shape
        self.pixel_interpolation_order = pixel_interpolation_order
        self.envelopes = envelopes
        self.nb_of_points = nb_of_points

        # ATTRIBUTEs processed
        self.envelopes_processed = self.envelope_pre_processing()
        self.warped_image = self.warp_image()

    def envelope_pre_processing(self) -> list[EnvelopeProcessing]:
        """
        To process the envelopes so that you can get the interpolation values of the envelope and
        hence get the corresponding polar coordinates for any given cumulative distance.

        Returns:
            list[EnvelopeProcessing]: the processed envelopes.
        """

        processed: list[EnvelopeProcessing] = [None] * len(self.envelopes)  #type:ignore
        for i, envelope in enumerate(self.envelopes):
            processed[i] = EnvelopeProcessing(
                polar_r=envelope.polar_r,
                polar_theta=envelope.polar_theta,
                nb_of_points=self.nb_of_points,
            )
        return processed

    @Decorators.running_time
    def warp_image(self) -> np.ndarray:
        """
        To warp the SDO image in between the two envelopes.

        Returns:
            np.ndarray: the warped SDO image.
        """

        # NEW IMAGE setup
        columns = np.linspace(0, 1, self.image_shape[1])
        rows = np.linspace(0, 1, self.image_shape[0])
        C, R = np.meshgrid(columns, rows)

        # COORDs transformation
        polar_r, polar_theta = self.transform(C, R)
        coords = np.stack([polar_r.ravel(), polar_theta.ravel()], axis=0)

        # INTERPOLATE image
        warped_image = scipy.ndimage.map_coordinates(  # todo need to check possible arguments
            self.sdo_image,
            coords,
            order=self.pixel_interpolation_order,
        ).reshape(self.image_shape)
        return warped_image

    def transform(
            self,
            cumulative_distance: np.ndarray,
            vertical_axis: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        To transform the SDO image coordinates to the polar coordinates of the envelopes.

        Args:
            cumulative_distance (np.ndarray): the cumulative distance at which to get the
                coordinates.
            vertical_axis (np.ndarray): the vertical axis to get the coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray]: the polar coordinates of the envelope for the given
                cumulative distance.
        """

        # ENVELOPE points
        r1, theta1 = self.envelopes_processed[0].get_coords(cumulative_distance)
        r2, theta2 = self.envelopes_processed[1].get_coords(cumulative_distance)

        # INTERPOLATE
        polar_r = r1 + vertical_axis * (r2 - r1)
        polar_theta = theta1 + vertical_axis * (theta2 - theta1)

        # COORDs pixel indices
        r_pixels, theta_pixels = self.polar_to_index(polar_r, polar_theta)
        return r_pixels, theta_pixels

    def polar_to_index(
            self,
            polar_r: np.ndarray,
            polar_theta: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        To convert the polar coordinates to the pixel indices.

        Args:
            polar_r (np.ndarray): the radial coordinate.
            polar_theta (np.ndarray): the angular coordinate.

        Returns:
            tuple[np.ndarray, np.ndarray]: the pixel indices.
        """

        # RESOLUTION pixels
        km_per_pixel = (abs(self.extent[3] - self.extent[2]) * 1e3) / self.sdo_image.shape[0]
        angle_per_pixel = abs(self.extent[1] - self.extent[0]) / self.sdo_image.shape[1]

        # COORDs to pixel indices
        r_pixels = (polar_r - self.extent[2] * 1e3) / km_per_pixel
        theta_pixels = (polar_theta - self.extent[0]) / angle_per_pixel
        return r_pixels, theta_pixels

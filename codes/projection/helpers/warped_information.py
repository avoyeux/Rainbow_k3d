"""
To create and format the needed information to create the final warped image information.
"""

# IMPORTs
import scipy 

# IMPORTs alias
import numpy as np

# IMPORTs sub

# IMPORTs personal
from common import Decorators
from .warp_sdo_image import WarpSdoImage, EnvelopeProcessing
from ..format_data import ImageBorders, FitWithEnvelopes, EnvelopeInformation

# TYPE ANNOTATIONs
from typing import Literal, TypeGuard


# todo get the arc length
# todo create a name to be used in the corresponding dataclass
# todo compute the corresponding angles for each pixel of the warped integration

"""
* I should use the middle curve of the envelope to get the closest angle to the pixel.
"""

# ! most likely need to compute the middle curve of the envelope each time.



class AllWarpedTreatments:

    @Decorators.running_time
    def __init__(
            self,
            sdo_image: np.ndarray,
            fit_n_envelopes: FitWithEnvelopes,
            borders: ImageBorders,
            image_shape: tuple[int, int],
            pixel_interpolation_order: int,
            nb_of_points: int,
            processed_fit: np.ndarray,  # ? should I change it to a dataclass
            integration_type: Literal['mean', 'median'] = 'mean',
        ) -> None:

        # ATTRIBUTEs
        self.integration_type = integration_type

        # CHECKs inputs
        if not self._validate_envelopes(fit_n_envelopes.envelopes):
            raise ValueError(
                f"Need the envelope information inside {self.__class__.__name__}."
            )
        
        # WARP image
        instance = WarpSdoImage(
            sdo_image=sdo_image,
            borders=borders,
            image_shape=image_shape,
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
        self.processed_fit = processed_fit  # ! is it already properly processed.

        # ANGLEs
        self.angles = self.get_closest_angles()
        
    def _validate_envelopes(
            self,
            envelopes: EnvelopeInformation | None,
        ) -> TypeGuard[EnvelopeInformation]:
        """
        To check that the envelopes data exist.

        Args:
            envelopes (EnvelopeInformation | None): the inputted envelopes information.

        Returns:
            TypeGuard[EnvelopeInformation]: True if the envelopes information is not None.
        """

        return envelopes is not None

    def get_closest_angles(self) -> np.ndarray:

        # KDTree angles
        tree = scipy.spatial.cKDTree(angles)  # todo need to decide how to get the angles
        
        # COORDs closest angle  
        dist, closest_indices = tree.query(
            np.stack([self.processed_middle.polar_r, self.processed_middle.polar_theta], axis=0),
        )
        return closest_indices

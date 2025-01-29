"""
To store the projection of the fake dataset to a fits file.
This is done to then create fake .save files from Dr. Auchere's IDL code to be able to tell if the
projection is correct.
"""

# IMPORTs
import os

# IMPORTs alias
import numpy as np

# IMPORTs personal
from Projection.polynomial_projection import OrthographicalProjection
from Projection.projection_dataclasses import ProcessConstants, ProjectionData



class FakeFits(OrthographicalProjection):
    """
    To store the projection of the fake dataset to a fits file.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # RUN
        self.run()

    def create_fake_fits(
            self,
            process_constants: ProcessConstants,
            projection_data: ProjectionData,
        ) -> None:
        
        # CHECKs
        assert projection_data.test_cube is not None

        # IMAGE N CONTOURs
        image, lines = self.cube_contour(
            rho=projection_data.test_cube[0],
            theta=projection_data.test_cube[1],
            dx=self.constants.dx,
            d_theta=self.constants.d_theta,
            # todo need to understand if I actually need to use image_shape or get the image one
            #step earlier.
        )










if __name__ == "__main__":
    
    FakeFits(
        filename='sig1e20_leg20_lim0_03.h5',
        with_feet=False,
        verbose=2,
        processes=2,
        polynomial_order=[4],
        plot_choices=[
            'test',
        ],
        flush=True,
    )

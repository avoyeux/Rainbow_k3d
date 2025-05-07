"""
To store the code to create the initial reprojection plots.
"""

# IMPORTs
import os

# IMPORTs alias
import numpy as np

# IMPORTs sub
import matplotlib.pyplot as plt

# IMPORTs personal
from common import Decorators
from codes.projection.format_data import (
    ProcessConstants, ProjectionData, ProjectedData, FitEnvelopes,
)
from codes.projection.sdo_reprojection import OrthographicalProjection
from codes.projection.envelope_distance import EnvelopeDistanceAnnotation

# TYPE ANNOTATIONs
from typing import cast, Any
from matplotlib.collections import PathCollection



class Plotting(OrthographicalProjection):
    """
    To plot the SDO's point of view image.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        To plot the SDO's point of view image.
        """

        # PARENT CLASS initialisation
        super().__init__(*args, **kwargs)

        # RUN code
        self.run()

    @Decorators.running_time
    def plotting(
            self,
            process_constants: ProcessConstants,
            projection_data: ProjectionData,
        ) -> None:
        """
        To plot the SDO's point of view image.
        What is plotted is dependent on the chosen choices when running the class.

        Args:
            process_constants (ProcessConstants): the constants for each cube..
            projection_data (ProjectionData): the data for each cube.
        """

        # IMAGE shape
        image_shape = (
            int(
                (
                    max(self.projection_borders.radial_distance) - 
                    min(self.projection_borders.radial_distance)
                ) * 1e3 / self.constants.dx
            ),
            int(
                (
                    max(self.projection_borders.polar_angle) -
                    min(self.projection_borders.polar_angle)
                ) / self.constants.d_theta
            ),
        )

        # SDO polar projection plotting
        plt.figure(num=1, figsize=(18, 5))
        if self.Auchere_envelope is not None: 
            # PLOT Auchere's envelope
            self.plt_plot(
                coords=self.Auchere_envelope.middle,
                colour='blue',
                label='Middle path',
                kwargs=cast(dict[str, Any], self.plot_kwargs['envelope']),
            )
            self.plt_plot(
                coords=self.Auchere_envelope.upper,
                colour='black',
                label='Envelope',
                kwargs=cast(dict[str, Any], self.plot_kwargs['envelope']),
            )
            self.plt_plot(
                coords=self.Auchere_envelope.lower,
                colour='black',
                kwargs=cast(dict[str, Any], self.plot_kwargs['envelope']),
            )

        if projection_data.sdo_mask is not None:
            # CONTOURS get
            lines = self.image_contour(
                image=projection_data.sdo_mask.image,
                d_theta=projection_data.sdo_mask.resolution_angle,
            )

            # CONTOURS plot
            if lines is not None:
                line = lines[0]
                self.plt_plot(
                    coords=(line[1], line[0]),
                    colour=projection_data.sdo_mask.colour,
                    label='SDO mask contours',
                    kwargs=cast(dict[str, Any], self.plot_kwargs['contour']),
                )
                for line in lines[1:]:
                    self.plt_plot(
                        coords=(line[1], line[0]),
                        colour=projection_data.sdo_mask.colour,
                        kwargs=cast(dict[str, Any], self.plot_kwargs['contour']),
                    )

        if projection_data.sdo_image is not None:
            plt.imshow(
                projection_data.sdo_image.image,
                **cast(dict[str, Any], self.plot_kwargs['image']),
            )

        sc = None  # for the colorbar
        if projection_data.integration is not None:
            
            for integration in projection_data.integration:
                # PLOT contours time integrated
                sc = self.plot_projected_data(
                    data=integration,
                    process_constants=process_constants,
                    image_shape=image_shape,
                )
        
        if projection_data.all_data is not None:
            # PLOT contours all data
            self.plot_contours(
                projection=projection_data.all_data,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        if projection_data.no_duplicates is not None:
            # PLOT contours no duplicates
            self.plot_contours(
                projection=projection_data.no_duplicates,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )
        
        if projection_data.full_integration_no_duplicates is not None:
            # PLOT contours full integration
            sc = self.plot_projected_data(
                data=projection_data.full_integration_no_duplicates,
                process_constants=process_constants,
                image_shape=image_shape,
            )

        if projection_data.line_of_sight is not None:
            # PLOT contours line of sight
            self.plot_contours(
                projection=projection_data.line_of_sight,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        # PLOT fake data
        if projection_data.fake_data is not None:
            # PLOT contours fake data
            self.plot_contours(
                projection=projection_data.fake_data,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        # PLOT test cube
        if projection_data.test_cube is not None:
            # PLOT contours test cube
            self.plot_contours(
                projection=projection_data.test_cube,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        # # COLORBAR add
        # if sc is not None:
        #     cbar = plt.colorbar(sc)
        #     cbar.set_label(r'$\theta$ (degrees)')

        # PLOT settings
        plt.xlim(
            min(self.projection_borders.polar_angle),
            max(self.projection_borders.polar_angle),
        )
        plt.ylim(
            min(self.projection_borders.radial_distance),
            max(self.projection_borders.radial_distance),
        )
        ax = plt.gca()
        ax.minorticks_on()
        ax.set_aspect('auto')
        plt.title(f"SDO polar projection - {process_constants.date}")
        plt.xlabel('Polar angle [degrees]')
        plt.ylabel('Radial distance [Mm]')
        plt.legend(loc='upper right')

        # PLOT save
        plot_name = f"reprojection_{process_constants.date}.png"
        plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=200)
        plt.close()

        if self.verbose > 1: 
            print(
                f'SAVED - nb {process_constants.time_index:03d} - {plot_name}',
                flush=self.flush,
            )

    def plot_projected_data(
            self,
            data: ProjectedData,
            process_constants: ProcessConstants,
            image_shape: tuple[int, int],
        ) -> PathCollection | None:
        """
        To plot the projected data and the corresponding fit and envelope if they exist.

        Args:
            data (ProjectedData): the projected data to be plotted.
            process_constants (ProcessConstants): the constants for each cube.
            image_shape (tuple[int, int]): the final image shape used in the contours plotting.

        Returns:
            PathCollection | None: the scatter plot of the projected polynomial fit. It is set to
                None if there is no polynomial fit to be plotted. This is later used to add a
                colorbar to the final plot.
        """

        # CONTOURS image
        self.plot_contours(
            projection=data,
            d_theta=self.constants.d_theta,
            image_shape=image_shape,
        )

        # FIT plot
        sc = None
        if data.fit_n_envelopes is not None:

            for fit_n_envelope in data.fit_n_envelopes:
                
                # PLOT
                sc = plt.scatter(
                    fit_n_envelope.fit_polar_theta,
                    fit_n_envelope.fit_polar_r / 1e3,
                    label=fit_n_envelope.name,
                    c=np.rad2deg(fit_n_envelope.fit_angles),
                    **cast(dict[str, Any], self.plot_kwargs['fit']),
                )

                # ENVELOPE fit
                if fit_n_envelope.envelopes is not None:
                    # PLOT envelope  # ? should I also add the middle path ?
                    self.plt_plot(
                        coords=fit_n_envelope.envelopes.upper,
                        colour=fit_n_envelope.colour,
                        label=(
                            f'Envelope ({fit_n_envelope.envelopes.upper.order}th) for '
                            + fit_n_envelope.name.lower()
                        ),
                        kwargs=cast(dict[str, Any], self.plot_kwargs['fit envelope']),
                    )
                    self.plt_plot(
                        coords=fit_n_envelope.envelopes.lower,
                        colour=fit_n_envelope.colour,
                        label=None,
                        kwargs=cast(dict[str, Any], self.plot_kwargs['fit envelope']),
                    )

                    # WARP image
                    if fit_n_envelope.warped_information is not None:
                        # PLOT inside a new figure
                        self.plot_warped_image(
                            warped_image=fit_n_envelope.warped_information.warped_values,
                            integration_time=data.integration_time,
                            date=process_constants.date,
                            fit_order=fit_n_envelope.fit_order,
                            envelope_order=fit_n_envelope.envelopes.upper.order,
                        )

                        # WARPED INTEGRATION to save RAM
                        fit_n_envelope.warped_information.warped_integration()
        return sc
    
    def plot_contours(
            self,
            projection: ProjectedData,
            d_theta: float, 
            image_shape: tuple[int, int],
        ) -> None:
        """
        To plot the contours of the image for the protuberance as seen from SDO's pov.

        Args:
            projection (ProjectedCube): the cube containing the information for the protuberance
                as seen from SDO's pov.
            d_theta (float): the theta angle resolution (as a function of the disk's perimeter) in
                degrees.
            image_shape (tuple[int, int]): the image shape needed for the image of the protuberance
                as seen from SDO.
        """

        # POLAR coordinates
        rho, theta = projection.cube.coords
        colour = projection.colour

        # CONTOURS cube
        _, lines = self.cube_contour(
            rho=rho,
            theta=theta,
            image_shape=image_shape,
            d_theta=d_theta,
        )

        # PLOT
        if lines is not None:
            line = lines[0]
            self.plt_plot(
                coords=(line[1], line[0]),
                colour=colour,
                label=projection.name + ' contour',
                kwargs=cast(dict[str, Any], self.plot_kwargs['contour']),
            )
            for line in lines:
                self.plt_plot(
                    coords=(line[1], line[0]),
                    colour=colour,
                    kwargs=cast(dict[str, Any], self.plot_kwargs['contour']),
                )

    def plot_warped_image(
            self,
            warped_image: np.ndarray,
            integration_time: int | str | None,
            date: str,
            fit_order: int | None, 
            envelope_order: int,
        ) -> None:
        """
        To plot the warped SDO image inside the fit envelope.

        Args:
            warped_image (np.ndarray): the warped SDO image to be plotted.
            integration_time (int | str | None): the integration time of the data. If the value is
                None, it means that the data doesn't have an integration time. If the value is a
                string, it means that the used data is the full integration one.
            date (str): the date of the data.
            fit_order (int | None): the order of the polynomial fit. If the value is None, it means
                that the fit doesn't have a polynomial order per se (like the middle path of 
                Auchere's envelope).
            envelope_order (int): the polynomial order of the envelope used to warp the SDO image.
        """

        # PLOT
        plot_name = (
            "warped_"
            f"{f'{integration_time}h_' if integration_time is not None else ''}"
            f"{f'{fit_order}fit_' if fit_order is not None else ''}"
            f"{envelope_order}envelope_{date}.png"
        )
        plt.figure(num=2, figsize=(10, 10))
        plt.imshow(
            X=warped_image.T,
            interpolation='none',
            cmap='gray',
            origin='lower',
        )
        plt.title(f'Warped SDO image - {date}')
        plt.savefig(os.path.join(self.paths['save warped'], plot_name), dpi=200)
        plt.close(2)
        plt.figure(1)

        if self.verbose > 0: print(f'SAVED - {plot_name}', flush=self.flush)

    def plt_plot(
            self,
            coords: tuple[np.ndarray, np.ndarray] | tuple[list, list] | FitEnvelopes,
            colour: str,
            label: str | None = None,
            kwargs: dict[str, Any] = {},
        ) -> None:
        """
        As I am using plt.plot a lot in this code, I put the usual plotting parameters in a method.

        Args:
            coords (tuple[np.ndarray, np.ndarray] | FitEnvelopes): the coordinates of the points to
                be plotted.
            colour (str): the colour of the plot lines.
            label (str | None, optional): the label of the plot lines. Defaults to None.
            kwargs (dict[str, Any], optional): additional arguments to be added to the plt.plot().
                Defaults to {}.
        """

        # CHECK
        if isinstance(coords, tuple):
            # PLOT
            plt.plot(coords[0], coords[1], label=label, color=colour, **kwargs)
        else:
            # PLOT
            plt.plot(
                coords.polar_theta,
                coords.polar_r / 1e3,
                label=label,
                color=colour,
                **kwargs,
            )

            # ANNOTATE
            EnvelopeDistanceAnnotation(
                fit_envelope=coords,
                colour=colour,
            )



if __name__ == '__main__':
    Plotting(
        integration_time=[24],
        polynomial_order=[4],
        plot_choices=[
            'no duplicates',
            'integration',
            'full integration',
            'fit',
            'sdo image', 'envelope',
            'warp',
            'all sdo images',
        ],
        with_fake_data=False,
    )

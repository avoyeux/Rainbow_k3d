"""
To get the fully processed fit results and the corresponding processed envelopes.
The processed polynomial fit was made so that the extremities stop at the Sun surface (when
possible). The envelope is processed so that the extremities stop at the Sun disk surface as seen
by SDO (when possible).
"""

# IMPORTs
import os
import h5py
import scipy

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import Protocol
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# IMPORTs personal
from common import config
from codes.projection.base_reprojection import BaseReprojection
from codes.projection.helpers.extract_envelope import CreateFitEnvelope
from codes.projection.helpers.projection_dataclasses import FitWithEnvelopes, FitEnvelopes
from codes.data.polynomial_fit.base_fit_processing import BaseFitProcessing
from codes.data.polynomial_fit.polynomial_bordered import ProcessedBorderedPolynomialFit

# API public
__all__ = ['ReprojectionProcessedPolynomial']



# STATIC type hints
class PolynomialCallable(Protocol):
    """
    To define the type hint for the polynomial callable as the function takes an undefined number
    of arguments and, as such, cannot be strictly defined with typing.Callable.
    """

    def __call__(self, t: np.ndarray, *coeffs: int | float) -> np.ndarray: ...


@dataclass(slots=True, repr=False, eq=False)
class ProcessedFit(BaseFitProcessing):
    """
    To process the fit results so that the coordinates are uniformly positioned on the curve.
    """

    # DATA unprocessed
    angles: np.ndarray

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
        coords = np.stack([self.polar_theta, self.polar_r, self.angles], axis=0)

        # SORT on first axis
        sorted_indexes = np.lexsort(coords[::-1])  # as lexsort sorts on last axis.
        sorted_coords = coords[:, sorted_indexes]

        # COORDs update
        self.polar_theta, self.polar_r, self.angles = sorted_coords

    def uniform_coords(self) -> None:
        """
        To uniformly space the coordinates on the curve.
        """

        # INTERPOLATE
        polar_r_interp = scipy.interpolate.interp1d(self.cumulative_distance, self.polar_r)
        polar_theta_interp = scipy.interpolate.interp1d(self.cumulative_distance, self.polar_theta)
        angles_interp = scipy.interpolate.interp1d(self.cumulative_distance, self.angles)

        # UNIFORM
        t_fine = np.linspace(0, 1, self.nb_of_points)
        self.polar_r = polar_r_interp(t_fine)
        self.polar_theta = polar_theta_interp(t_fine)
        self.angles = angles_interp(t_fine)


@dataclass(slots=True, repr=False, eq=False)
class ProcessedEnvelope(BaseFitProcessing):
    """
    To process the fit envelope results so that the coordinates are uniformly positioned on the
    and are bordered.
    """

    # PARAMETERs
    feet_sigma: float
    feet_threshold: float
    polynomial_order: int

    # PLACEHOLDERs
    success: bool = field(init=False)
    polynomial_parameters: np.ndarray = field(init=False)
    polynomial_callable: PolynomialCallable = field(init=False)

    def __post_init__(self) -> None:

        # SETUP polynomial
        self.polynomial_callable = self.nth_order_polynomial_generator()

        # COORDs re-ordered (for the cumulative distance)
        self.reorder_data()

        # COORDs normalised
        self.normalise_coords()

        # DISTANCE cumulative
        self.cumulative_distance_normalised()

        # FIT parameters
        self.scipy_curve_fit()

        # BORDERs cut
        self.bordered_envelope()

    def reorder_data(self) -> None:
        """
        To reorder the data so that the cumulative distance is calculated properly for the fit.
        That means that the first axis to order should be the polar theta one.
        """
        
        # RE-ORDER
        coords = np.stack([self.polar_theta, self.polar_r], axis=0)

        # SORT on first axis
        sorted_indexes = np.lexsort(coords[::-1])  # as lexsort sorts on last axis.
        sorted_coords = coords[:, sorted_indexes]

        # COORDs update
        self.polar_theta, self.polar_r = sorted_coords

    def scipy_curve_fit(self) -> None:
        """
        To fit a polynomial on the data using scipy's curve_fit.
        """

        # SETUP
        params_init = np.random.rand(self.polynomial_order + 1)
        sigma = np.ones(self.cumulative_distance.size, dtype='float64')
        feet_mask = (
            self.cumulative_distance < self.feet_threshold
            ) | (
            self.cumulative_distance > 1 - self.feet_threshold
        )
        sigma[feet_mask] = self.feet_sigma

        try:
            # FITTING scipy
            params_x, _ = scipy.optimize.curve_fit(
                f=self.polynomial_callable,
                xdata=self.cumulative_distance,
                ydata=self.polar_r, 
                p0=params_init,
                sigma=sigma,
            )
            params_y, _ = scipy.optimize.curve_fit(
                f=self.polynomial_callable,
                xdata=self.cumulative_distance,
                ydata=self.polar_theta,
                p0=params_init,
                sigma=sigma,
            )
            params = np.stack([params_x, params_y], axis=0).astype('float64')

            # FLAG success
            self.success = True

        except Exception:
            # FAIL save
            params = np.empty((2, 0))  # todo change this to None later

            # FLAG fail
            self.success = False

        # PARAMs update
        self.polynomial_parameters = params
    
    def bordered_envelope(self) -> None:
        """
        To cut the envelope at the sun disk.
        """

        # CURVE longer
        t_fine = np.linspace(-0.2, 1.2, int(1e4))
        self.polar_r = self.polynomial_callable(t_fine, *self.polynomial_parameters[0])
        self.polar_theta = self.polynomial_callable(t_fine, *self.polynomial_parameters[1])

        # BORDERs cut
        conditions = (
            (self.polar_r < 698 * 1e3) |  # * final plot borders
            (self.polar_r > 870 * 1e3) |
            (self.polar_theta < 245) |
            (self.polar_theta > 295)
        )
        new_t = t_fine[~conditions]

        # CURVE bordered
        new_t_fine = np.linspace(new_t.min(), new_t.max(), self.nb_of_points)
        self.polar_r = self.polynomial_callable(new_t_fine, *self.polynomial_parameters[0])
        self.polar_theta = self.polynomial_callable(
            new_t_fine,
            *self.polynomial_parameters[1],
        )

    def nth_order_polynomial_generator(self) -> PolynomialCallable:
        """
        To generate a polynomial function given a polynomial order.

        Returns:
            typing.Callable[[np.ndarray, * int | float], np.ndarray]: the polynomial
                function.
        """

        def nth_order_polynomial(t: np.ndarray, *coeffs: int | float) -> np.ndarray:
            """
            Polynomial function given a 1D ndarray and the polynomial coefficients. The polynomial
            order is defined before hand.

            Args:
                t (np.ndarray): the 1D array for which you want the polynomial results.
                coeffs (int | float): the coefficient(s) for the polynomial in the order a_0 + 
                    a_1 * t + a_2 * t**2 + ...

            Returns:
                np.ndarray: the polynomial results.
            """

            # INIT
            result: np.ndarray = 0 #type:ignore

            # POLYNOMIAL
            for i in range(self.polynomial_order + 1): result += coeffs[i] * t ** i
            return result
        return nth_order_polynomial


class Fitting2D:
    """
    To fit a polynomial on the 2D envelopes.
    To fit is done so that the extremities of the envelope are elongated and can be cut when
    intersecting the Sun disk as seen from SDO.
    """

    def __init__(
            self,
            polar_coords: np.ndarray,
            polynomial_order: int,
            nb_of_points: int,
            feet_sigma: float,
            feet_threshold: float,
        ) -> None:
        """
        To initialise the class.
        The fit_data method should be called to get the results.

        Args:
            polar_coords (np.ndarray): the polar coordinates of the envelope.
            polynomial_order (int): the order of the polynomial to fit.
            feet_sigma (float): the sigma to use for the feet of the envelope.
            feet_threshold (float): the threshold representing how much of the data to consider as
                the feet. E.G. 0.1 means that the first 10% and last 10% of the data are considered
                as the feet.
            nb_of_points (int): the number of points to have in the final fit.
        """
        
        # ATTRIBUTEs from args
        self.feet_sigma = feet_sigma
        self.polar_coords = polar_coords
        self.nb_of_points = nb_of_points
        self.feet_threshold = feet_threshold
        self.polynomial_order = polynomial_order

        # NEW attributes
        self.fit_success: bool
        self.params_init = np.random.rand(polynomial_order + 1)

        # CHECK params
        self.enough_params = self.params_init.size < self.polar_coords.shape[1]

    def fit_data(self) -> FitEnvelopes:
        """
        To fit a polynomial on the data.
        The polynomial is processed so that it has a pre-defined number of points and borders
        defined in polar coordinates from SDO's perspective.

        Returns:
            FitEnvelopes: the polar coordinates of the processed fit envelope.
        """

        # CHECK nb of coords
        if not self.enough_params:
            
            # FIT not possible
            coords = np.empty((2, 0))
            params = np.empty((2, 0))

            # DATA format
            envelope = FitEnvelopes(
                order=self.polynomial_order,
                polar_r=coords[0],  # ? does that even work ?
                polar_theta=coords[1],
            )
        else:
            # ENVELOPE processing
            processed_envelope = ProcessedEnvelope(
                polar_r=self.polar_coords[0],
                polar_theta=self.polar_coords[1],
                nb_of_points=self.nb_of_points,
                polynomial_order=self.polynomial_order,
                feet_sigma=self.feet_sigma,
                feet_threshold=self.feet_threshold,
            )
            coords = np.stack([processed_envelope.polar_r, processed_envelope.polar_theta], axis=0)

            # ENVELOPE fit
            envelope = FitEnvelopes(
                order=self.polynomial_order,
                polar_r=processed_envelope.polar_r,
                polar_theta=processed_envelope.polar_theta,
            )

            # SAVE success
            self.fit_success = processed_envelope.success
        return envelope 


class ReprojectionProcessedPolynomial(ProcessedBorderedPolynomialFit, BaseReprojection):
    """
    To get the processed polynomial 3D fit results and create the corresponding 2D envelopes.
    The processed polynomial fit was made so that the extremities stop at the Sun surface (when 
    possible). The envelope is processed so that the extremities stop at the Sun disk surface as
    seen by SDO (when possible).
    """

    def __init__(
            self,
            filepath: str,
            dx: float,
            index: int,
            colour: str,
            sdo_pos: np.ndarray,
            polynomial_order: int,
            integration_time: int,
            number_of_points: int,
            feet_sigma: float,
            feet_threshold: float,
            data_type: str = 'No duplicates',
            with_fake_data: bool = False,
            create_envelope: bool = True,
            verbose: bool | None = None,
            flush: bool | None = None,
            test_plots: bool | None = None,
        ) -> None:
        """
        To initialise the class.
        The reprocessed_fit_n_envelopes method should be called to get the results.

        Args:
            filepath (str): the filepath to the hdf5 file containing the fit results.
            dx (float): the voxel resolution in km.
            index (int): the index of the cube to consider.
            colour (str): the colour of the fit and corresponding envelope in the final plot.
            sdo_pos (np.ndarray): the position of SDO in heliographic coordinates.
            polynomial_order (int): the order of the polynomial fit to consider.
            integration_time (int): the integration time to consider when choosing the polynomial
                fit parameters (in hours).
            number_of_points (int): the number of points to consider in the final polynomial fit.
            feet_sigma (float): the sigma to use for the feet of the envelope.
            feet_threshold (float): the threshold representing how much of the data to consider as
                the feet. E.G. 0.1 means that the first 10% and last 10% of the data are considered
                as the feet.
            data_type (str, optional): the data type to consider when looking for the corresponding
                polynomial fit. Defaults to 'No duplicates'.
            with_fake_data (bool, optional): when the HDF5 file also contains fake data (as the
                default path change). Defaults to False.
            create_envelope (bool, optional): to create the envelope or not. Defaults to True.
            verbose (bool | None, optional): the verbosity level in the prints. When None, the
                config file value is taken. Defaults to None.
            flush (bool | None, optional): to flush the prints or not. When None, the config file
                value is taken. Defaults to None.
            test_plots (bool | None, optional): to plot the results or not. When None, the config
                file value is taken. Defaults to None.
        """

        # CONFIGURATION attributes
        self.flush: bool = config.run.flush if flush is None else flush  #type:ignore
        self.verbose: bool = config.run.verbose if verbose is None else verbose  #type:ignore
        self.plots: bool = config.run.test_plots if test_plots is None else test_plots #type:ignore

        # PARENTs
        super().__init__(
            filepath=filepath,
            polynomial_order=polynomial_order,
            integration_time=integration_time,
            number_of_points=250,
            dx=dx,
            data_type=data_type,
            with_fake_data=with_fake_data,
        )
        BaseReprojection.__init__(self)

        # ATTRIBUTEs
        self.paths = self.paths_setup()
        self.dx = dx
        self.index = index
        self.colour = colour
        self.sdo_pos = sdo_pos
        self.feet_sigma = feet_sigma
        self.feet_threshold = feet_threshold
        self.nb_of_points = number_of_points
        self.polynomial_order = polynomial_order
        self.create_envelope = create_envelope
    
    def paths_setup(self) -> dict[str, str]:
        """
        To format the needed paths for the class.

        Returns:
            dict[str, str]: the formatted paths.
        """

        # PATHs formatting
        paths = {
            'save': config.path.dir.data.temp, #type:ignore # todo change this later or take it way.
        }
        return paths

    def reprocessed_fit_n_envelopes(self) -> FitWithEnvelopes:
        """
        To reprocess the fit results and create the corresponding envelopes.

        Returns:
            FitWithEnvelopes: the processed fit results and the corresponding envelopes.
        """

        # FIT processed borders 3D
        fit_processed_3d = self.reprocessed_polynomial(self.index)

        # FIT polar 2D
        fit_2D_polar = self.get_polar_image_angles(
            self.matrix_rotation(
                data=self.cartesian_pos(fit_processed_3d, self.dx).coords,
                sdo_pos=self.sdo_pos,
            ))
        
        # FIT uniform processing
        fit_2D_uniform = ProcessedFit(
            polar_r=fit_2D_polar[0],
            polar_theta=fit_2D_polar[1],
            angles=fit_2D_polar[2],
            nb_of_points=self.nb_of_points,
        )
        coords_uniform = np.stack([fit_2D_uniform.polar_r, fit_2D_uniform.polar_theta], axis=0)

        if self.create_envelope:
            # ENVELOPE
            envelopes_polar = CreateFitEnvelope.get(
                coords=coords_uniform,
                radius=3e4,
            )
            
            envelopes: list[FitEnvelopes] = []
            for envelope in envelopes_polar:
                
                # FIT envelope
                fitting_method = Fitting2D(
                    polar_coords=envelope,
                    polynomial_order=6,
                    nb_of_points=self.nb_of_points,
                    feet_sigma=self.feet_sigma,
                    feet_threshold=self.feet_threshold,
                )
                envelope = fitting_method.fit_data()
                envelopes.append(envelope)

                # LOG success
                if self.verbose > 0:
                    log = self.success_log(fitting_method)
                    if log != '': print(self.success_log(fitting_method), flush=self.flush)
            
            # PLOT for testing
            if self.plots:
                cube_date = self.get_cube_date()
                self.plot(
                    date=cube_date,
                    fit=coords_uniform,
                    old_envelope=[envelope for envelope in envelopes_polar],
                    new_envelope=envelopes,
                )
        else:
            envelopes = None  #type:ignore
        
        # DATA format
        fit_n_envelopes = FitWithEnvelopes(
            colour=self.colour,
            integration_time=self.integration_time,
            fit_order=self.polynomial_order,
            fit_polar_r=fit_2D_uniform.polar_r,
            fit_polar_theta=fit_2D_uniform.polar_theta,
            fit_angles=fit_2D_uniform.angles,
            envelopes=envelopes,
        )
        return fit_n_envelopes

    def success_log(self, instance: Fitting2D) -> str:
        """
        To log the success of the fit.

        Args:
            instance (Fitting2D): the instance of the Fitting2D class.

        Returns:
            str: the log message.
        """

        # LOG setup
        log: str = ''

        # CHECK params
        if not instance.enough_params:
                log += (
                    f"\033[1;31mFor cube {self.index:03d}, not enough points for the polynomial "
                    f"fit (shape: {instance.polar_coords.shape[0]}). Going to next cube.\033[0m\n"
                )

        # CHECK fit completion
        if not instance.fit_success:
            log += (
                f"\033[1;31mFor cube {self.index:03d}, the polynomial fit failed. Going to next "
                "cube.\033[0m"
            )
        return log
    
    def get_cube_date(self) -> str:
        """
        To get the date of the cube.

        Returns:
            str: the date of the cube.
        """

        # DATA from file
        dates: h5py.Dataset = self.file['Dates']  #type:ignore
        cube_numbers: h5py.Dataset = self.file['Real/Time indexes']  #type:ignore     

        # DATA cube
        cube_number = cube_numbers[self.index]
        date: str = dates[cube_number].decode('utf-8')
        return date

    def plot(
            self,
            date: str,
            fit: np.ndarray,
            old_envelope: list[np.ndarray],
            new_envelope: list[FitEnvelopes],
        ) -> None:
        """
        To visualise the results of the fitting (only used during the results).

        Args:
            date (str): the date of the cube.
            fit (np.ndarray): the fit results.
            old_envelope (list[np.ndarray]): the old envelope gotten directly from the fit.
            new_envelope (list[np.ndarray]): the new envelope gotten from processing the old one.
        """

        # SETUP
        filename = f'envelope_fit_results_{date}.png'
        plt.figure(figsize=(10, 20))

        # PLOT envelopes
        self.plot_sub(old_envelope,'black', 'Old envelope')
        self.plot_sub(
            curves=[
                np.stack([envelope.polar_r, envelope.polar_theta], axis=0)
                for envelope in new_envelope
            ],
            colour='green',
            label='New envelope',
        )

        # PLOT fit
        plt.scatter(fit[1], fit[0], color='red', label='Fit points')
        plt.legend()
        plt.savefig(os.path.join(self.paths['save'], filename), dpi=500)
        plt.close()
        print(f'SAVED - {filename}', flush=self.flush)

    def plot_sub(self, curves: list[np.ndarray], colour: str, label: str) -> None:
        """
        To plot the curves so that only one label is shown for the two plt.scatter.

        Args:
            curves (list[np.ndarray]): the curves to plot.
            colour (str): the colour of the points.
            label (str): the label to show.
        """

        curve = curves[0]
        plt.scatter(curve[1], curve[0], color=colour, label=label)
        curve = curves[1]
        plt.scatter(curve[1], curve[0], color=colour)

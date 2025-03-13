"""
To border the polynomial fit results given positions in the polar reprojected plot.
# todo will need to change this docstring when I start properly structuring the corresponding code.
"""

# IMPORTs
import os
import scipy

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import Protocol
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# IMPORTs personal
from common import config, Decorators
from codes.projection.base_reprojection import BaseReprojection
from codes.projection.extract_envelope import CreateFitEnvelope
from codes.projection.projection_dataclasses import FitWithEnvelopes, FitEnvelopes
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
class BaseProcessing:
    """
    Base to create and easily access the uniform coordinates for the fit and the envelope.
    """

    # DATA unprocessed
    polar_r: np.ndarray
    polar_theta: np.ndarray

    # PARAMETERs
    nb_of_points: int

    # PLACEHOLDERs
    cumulative_distance: np.ndarray = field(init=False)
    polar_r_normalised: np.ndarray = field(init=False)
    polar_theta_normalised: np.ndarray = field(init=False)

    def normalise_coords(self) -> None:
        """
        Normalise the coordinates so that they are between 0 and 1. As such, the cumulative
        distance won't only depend on one axis.
        """

        # COORDs
        coords = np.stack([self.polar_r, self.polar_theta], axis=0)

        # NORMALISE
        min_vals = np.min(coords, axis=1, keepdims=True)
        max_vals = np.max(coords, axis=1, keepdims=True)
        coords = (coords - min_vals) / (max_vals - min_vals)

        # COORDs update
        self.polar_r_normalised, self.polar_theta_normalised = coords
    
    def cumulative_distance_normalised(self) -> None:
        """
        To calculate the cumulative distance of the data and normalise it.
        """
        
        # COORDs
        coords = np.stack([self.polar_theta_normalised, self.polar_r_normalised], axis=0)

        # DISTANCE cumulative
        t = np.empty(coords.shape[1], dtype='float64')
        t[0] = 0
        for i in range(1, coords.shape[1]):
            t[i] = t[i - 1] + np.linalg.norm(coords[:, i] - coords[:, i - 1])
        t /= t[-1]  # normalise

        # DISTANCE update
        self.cumulative_distance = t


@dataclass(slots=True, repr=False, eq=False)
class ProcessedFit(BaseProcessing):
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

        # INTERPOLATE  # ? do I need to use the normalised values here ?
        polar_r_interp = scipy.interpolate.interp1d(self.cumulative_distance, self.polar_r)
        polar_theta_interp = scipy.interpolate.interp1d(self.cumulative_distance, self.polar_theta)
        angles_interp = scipy.interpolate.interp1d(self.cumulative_distance, self.angles)

        # UNIFORM
        t_fine = np.linspace(0, 1, self.nb_of_points)
        self.polar_r = polar_r_interp(t_fine)
        self.polar_theta = polar_theta_interp(t_fine)
        self.angles = angles_interp(t_fine)


@dataclass(slots=True, repr=False, eq=False)
class ProcessedEnvelope(BaseProcessing):
    """
    To process the fit envelope results so that the coordinates are uniformly positioned on the
    and are bordered.
    """

    # PARAMETERs
    polynomial_order: int

    # PLACEHOLDERs
    success: bool = field(init=False)
    new_t_fine: np.ndarray = field(init=False)  # ! take it away after the plot tests are done
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

        # PARAMs initialisation
        params_init = np.random.rand(self.polynomial_order + 1)

        try:
            # FITTING scipy
            params_x, _ = scipy.optimize.curve_fit(
                f=self.polynomial_callable,
                xdata=self.cumulative_distance,
                ydata=self.polar_r, 
                p0=params_init,
            )
            params_y, _ = scipy.optimize.curve_fit(
                f=self.polynomial_callable,
                xdata=self.cumulative_distance,
                ydata=self.polar_theta,
                p0=params_init,
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
            (self.polar_theta < 255) |
            (self.polar_theta > 295)
        )
        new_t = t_fine[~conditions]

        # CURVE bordered
        self.new_t_fine = np.linspace(new_t.min(), new_t.max(), self.nb_of_points)
        self.polar_r = self.polynomial_callable(self.new_t_fine, *self.polynomial_parameters[0])
        self.polar_theta = self.polynomial_callable(
            self.new_t_fine,
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


class ReprojectionProcessedPolynomial(ProcessedBorderedPolynomialFit, BaseReprojection):
    """
    # todo change this docstring later
    This is to get the params of the polynomial fit envelope to then cut the envelope at the sun
    disk and recreate the image that is in the coronal monsoon paper.
    """

    def __init__(
            self,
            filepath: str,
            dx: float,
            index: int,
            sdo_pos: np.ndarray,
            polynomial_order: int,
            integration_time: int,
            number_of_points: int,
            data_type: str = 'No duplicates',
            with_fake_data: bool = False,
            create_envelope: bool = True
        ) -> None:
        
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
        self.sdo_pos = sdo_pos
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
            'save': config.path.dir.code.fit, #type:ignore # todo change this later or take it way.
        }
        return paths

    @Decorators.running_time
    def reprocessed_fit_n_envelopes(self, process: int) -> FitWithEnvelopes:

        # FIT processed borders 3D
        fit_processed_3d = self.reprocessed_polynomial(process)

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
            
            envelopes = []
            new_envelopes = []
            for i, envelope in enumerate(envelopes_polar):

                fitting_method = Fitting2D(
                    polar_coords=envelope,
                    index=self.index,
                    polynomial_order=9,
                    nb_of_points=self.nb_of_points,
                    identifier=i,
                )
                envelope = fitting_method.fit_data()
                envelopes.append(envelope)
        else:
            envelopes = None
        
        # DATA format
        fit_n_envelopes = FitWithEnvelopes(
            fit_order=self.polynomial_order,
            fit_polar_r=fit_2D_uniform.polar_r,
            fit_polar_theta=fit_2D_uniform.polar_theta,
            fit_angles=fit_2D_uniform.angles,
            envelopes=envelopes,
        )
        return fit_n_envelopes
        # self.plot(coords_uniform, old_envelope=envelopes, new_envelope=new_envelopes)
    
    def plot(
            self,
            fit: np.ndarray,
            old_envelope: list[np.ndarray],
            new_envelope: list[np.ndarray],
        ) -> None:
        """
        To visualise the results of the fitting (only used during the results).

        Args:
            fit (np.ndarray): the fit results.
            old_envelope (list[np.ndarray]): the old envelope gotten directly from the fit.
            new_envelope (list[np.ndarray]): the new envelope gotten from processing the old one.
        """

        filename = f'envelope_fit_results_{self.index:03d}.png'
        plt.figure(figsize=(10, 20))
        self.plot_sub(old_envelope,'black', 'Old envelope')
        self.plot_sub(new_envelope, 'purple', 'New envelope')
        plt.scatter(fit[1], fit[0], color='red', label='Fit points')
        plt.legend()
        plt.savefig(os.path.join(self.paths['save'], filename), dpi=500)
        plt.close()
        print(f'SAVED - {filename}', flush=True)

    def plot_sub(self, curves: list[np.ndarray], colour: str, label: str) -> None:

        curve = curves[0]
        plt.scatter(curve[1], curve[0], color=colour, label=label)
        curve = curves[1]
        plt.scatter(curve[1], curve[0], color=colour)


class Fitting2D:
    """
    # todo change this docstring later.
    To fit a polynomial on the 2D envelopes.
    """

    def __init__(
            self,
            polar_coords: np.ndarray,
            index: int,
            polynomial_order: int,
            nb_of_points: int,
            verbose: bool | None = None,
            flush: bool | None = None,
            identifier: int = 0,
            step: int = 10,
        ) -> None:
        
        # CONFIG attributes
        self.verbose = config.run.verbose if verbose is None else verbose #type:ignore
        self.flush = config.run.flush if flush is None else flush #type:ignore
        
        # ATTRIBUTEs from args
        self.index = index
        self.polar_coords = polar_coords
        self.nb_of_points = nb_of_points
        self.polynomial_order = polynomial_order
        self.identifier = identifier
        self.step = step

        # NEW attributes
        self.paths = self.paths_setup()
        self.params_init = np.random.rand(polynomial_order + 1)

    def paths_setup(self) -> dict[str, str]:

        # PATHs formatting
        paths = {
            'save': config.path.dir.code.fit, #type:ignore # todo change this later or take it way.
        }
        return paths

    @Decorators.running_time
    def fit_data(self) -> FitEnvelopes:
        """ #todo update docstring
        To fit a polynomial on the data.
        The polynomial is processed so that it has a pre-defined number of points and borders
        defined in polar coordinates from SDO's perspective.

        Returns:
            tuple[np.ndarray, np.ndarray]: the parameters and the coordinates of the polynomial
                fit.
        """

        # SETUP
        nb_parameters = self.params_init.size

        # CHECK nb of coords
        if nb_parameters >= self.polar_coords.shape[1]:
            if self.verbose > 0:
                print(
                    f"\033[1;31mFor cube {self.index:03d}, not enough points for the polynomial "
                    f"fit (shape: {self.polar_coords.shape[0]}). Going to next cube.\033[0m",
                    flush=self.flush,
                )
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
            )
            coords = np.stack([processed_envelope.polar_r, processed_envelope.polar_theta], axis=0)
            params = processed_envelope.polynomial_parameters

            envelope = FitEnvelopes(
                order=self.polynomial_order,
                polar_r=processed_envelope.polar_r,
                polar_theta=processed_envelope.polar_theta,
            )

            # # PLOT tests
            # self.plot_fitting_results(
            #     t=processed_envelope.new_t_fine,
            #     t_init=processed_envelope.cumulative_distance,
            #     fit_x=processed_envelope.polar_r,
            #     fit_y=processed_envelope.polar_theta,
            # )
            # CHECK success
            if not processed_envelope.success:
                if self.verbose > 0:
                    print(
                        f"\033[1;31mFor cube {self.index:03d}, the polynomial fit failed. Going "
                        "to next cube.\033[0m",
                        flush=self.flush,
                    )
        return envelope 

    def plot_fitting_results(
            self,
            t: np.ndarray,
            t_init: np.ndarray,
            fit_x: np.ndarray,
            fit_y: np.ndarray,
        ) -> None:

        # PLOT
        filename = f'fit_results_{self.identifier}_{self.index:03d}_x.png'
        self.plotting_sub(filename, t_init, t, self.polar_coords[0], fit_x)

        filename = f'fit_results_{self.identifier}_{self.index:03d}_y.png'
        self.plotting_sub(filename, t_init, t, self.polar_coords[1], fit_y)

    def plotting_sub(
            self,
            filename: str,
            t_init: np.ndarray,
            t: np.ndarray,
            points: np.ndarray,
            fit: np.ndarray,
        ) -> None:

        plt.figure(figsize=(10, 20))
        plt.scatter(t_init, points, label='Points')
        plt.scatter(t, fit, label='Fit')
        plt.legend()
        plt.savefig(os.path.join(self.paths['save'], filename), dpi=500)
        plt.close()

        print(f'SAVED - {filename}', flush=True)

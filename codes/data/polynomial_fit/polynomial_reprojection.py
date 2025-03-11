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
import matplotlib.pyplot as plt

# IMPORTs personal
from common import config, Decorators
from codes.projection.base_reprojection import BaseReprojection
from codes.data.polynomial_fit.base_polynomial_fit import GetPolynomialFit
from codes.projection.projection_dataclasses import CubeInformation
from codes.projection.extract_envelope import CreateFitEnvelope

# API public
__all__ = ['ReprojectionProcessedPolynomial', 'Fitting2D']



# STATIC type hints
class PolynomialCallable(Protocol):
    """
    To define the type hint for the polynomial callable as the function takes an undefined number
    of arguments and, as such, cannot be strictly defined with typing.Callable.
    """

    def __call__(self, t: np.ndarray, *coeffs: int | float) -> np.ndarray: ...


class ReprojectionProcessedPolynomial(GetPolynomialFit, BaseReprojection):
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
        ) -> None:
        
        # PARENTs
        super().__init__(
            filepath=filepath,
            polynomial_order=polynomial_order,
            integration_time=integration_time,
            number_of_points=number_of_points,  #maybe change this to 0 if not needed.
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
        self.polynomial_order = 8

        # ! I need to redefine self.t_fine
    
    def paths_setup(self) -> dict[str, str]:

        # PATHs formatting
        paths = {
            'save': config.path.dir.code.fit, # todo change this later or take it way.
        }
        return paths

    @Decorators.running_time
    def reprocessed_fit(self, process: int) -> CubeInformation:  # ? do I need processes as arg ?

        # PARAMs polynomial
        params = self.get_params(cube_index=process)

        # COORDs cartesian
        coords = self.get_coords(self.t_fine, params)
        coords = CubeInformation(
            xt_min=self.polynomial_info.xt_min,
            yt_min=self.polynomial_info.yt_min,
            zt_min=self.polynomial_info.zt_min,
            coords=coords,
        )
        coords = self.cartesian_pos(coords, self.dx)
        print(f'coords shape is {coords.coords.shape}', flush=True)

        # TEST PLOTTING
        self.get_envelope_params(coords.coords)

        # ! I am still not sure how to take into account the angles nor the interpolation for when
        # ! recreating the image inside the envelope (c.f. Coronal monsoon paper).


    def get_envelope_params(self, coords: np.ndarray) -> None:
        
        # COORDs cartesian to polar  # ? add somewhere else as it could be reused later ?
        polar_r, polar_theta = self.get_polar_image(self.matrix_rotation(coords, self.sdo_pos))

        # ENVELOPE
        envelopes_polar = CreateFitEnvelope.get(
            coords=np.stack([polar_r, polar_theta], axis=0),  # * be careful, order changed
            radius=3e4,
        )
        print(f"the shapes of the envelope curves are {' - '.join([str(envelope.shape) for envelope in envelopes_polar])}", flush=True)

        envelopes = []
        new_envelopes = []
        for i, envelope in enumerate(envelopes_polar):
            fitting_method = Fitting2D(
                polar_coords=envelope,
                index=self.index,
                polynomial_order=self.polynomial_order,
                nb_of_points=self.nb_of_points,
            )
            params, coords = fitting_method.fit_data()
            envelopes.append(envelope)
            new_envelopes.append(coords)
        self.plot(np.stack([polar_r, polar_theta], axis=0), old_envelope=envelopes, new_envelope=new_envelopes)

            # todo will need to add stuff here but for now lets just plot the result
        
    def plot(
            self,
            fit: np.ndarray,
            old_envelope: list[np.ndarray],
            new_envelope: list[np.ndarray],
        ) -> None:

        print(f'starting plot for {self.index:03d}', flush=True)
        filename = f'd_envelope_fit_results_{self.index:03d}.png'
        plt.figure(figsize=(10, 20))
        self.plot_sub(old_envelope, 'Old envelope')
        self.plot_sub(new_envelope, 'New envelope')
        plt.plot(fit[0], fit[1], label='Fit points')
        plt.legend()
        plt.savefig(os.path.join(self.paths['save'], filename), dpi=500)
        plt.close()

    def plot_sub(self, curves: list[np.ndarray], label: str) -> None:

        curve = curves[0]
        plt.plot(curve[0], curve[1], label=label)
        curve = curves[1]
        plt.plot(curve[0], curve[1])


class Fitting2D:
    """
    # todo change this docstring later.
    To fit a polynomial on the 2D envelopes to then use the result in the 
    reprojectionprocessedpolynomial class.
    """

    def __init__(
            self,
            polar_coords: np.ndarray,
            index: int,
            polynomial_order: int,
            nb_of_points: int,
            verbose: bool | None = None,
            flush: bool | None = None,
        ) -> None:
        
        # CONFIG attributes
        self.verbose = config.run.verbose if verbose is None else verbose
        self.flush = config.run.flush if flush is None else flush
        
        # ATTRIBUTEs from args
        self.index = index
        self.polar_coords = self.reorder_data(polar_coords)
        self.nb_of_points = nb_of_points
        self.polynomial_order = polynomial_order

        # NEW attributes
        self.params_init = np.random.rand(polynomial_order + 1)
        self.polynomial_callable = self.nth_order_polynomial_generator()

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
            result: np.ndarray = 0

            # POLYNOMIAL
            for i in range(self.polynomial_order + 1): result += coeffs[i] * t ** i
            return result
        return nth_order_polynomial

    def reorder_data(self, coords: np.ndarray) -> np.ndarray:

        return coords[[1, 0], :]
        # ! the axis representing the polar angle needs to come first.

    @Decorators.running_time
    def fit_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
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
        else:
            # DISTANCE cumulative
            t = np.empty(self.polar_coords.shape[1], dtype='float64')
            t[0] = 0
            for i in range(1, self.polar_coords.shape[1]):
                t[i] = t[i - 1] + np.linalg.norm(
                    self.polar_coords[:, i] - self.polar_coords[:, i - 1]
                )
            t /= t[-1]  # normalise

            # FITTING processed
            params, coords = self.processed_fit(t)
        return params, coords[[1, 0], :]

    def processed_fit(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        To process the polynomial fit positions so that the final result is a curve with a set
        number of points and borders defined in polar coordinates from SDO's perspective.

        Args:
            t (np.ndarray): the cumulative distance.

        Returns:
            tuple[np.ndarray, np.ndarray]: the parameters and the coordinates of the polynomial
                fit.
        """

        # PARAMs polynomial fit
        params = self.scipy_curve_fit(t=t)

        # CURVE re-creation
        t_fine = np.linspace(-0.3, 1.3, int(1e5))
        curve = self.get_coords(t_fine, params)
        print(f'curve shape is {curve.shape}')
        print(f'curve[0] min is {curve[0].min()} and max is {curve[0].max()}')
        print(f'curve[1] min is {curve[1].min()} and max is {curve[1].max()}', flush=True)

        # BORDERs cut
        conditions = (  # ! make sure that the values are in degrees and not rad
            (curve[0] < 245) |  # * final plot borders
            (curve[0] > 295) |
            (curve[1] < 698 * 1e3) | 
            (curve[1] > 870 * 1e3)
        )
        new_t = t_fine[~conditions]

        # CURVE final (has a defined number of points)
        new_t_fine = np.linspace(new_t.min(), new_t.max(), self.nb_of_points)
        curve = self.get_coords(new_t_fine, params)
        return params, curve

    def scipy_curve_fit(self, t: np.ndarray) -> np.ndarray:
        """
        To fit a polynomial on the data using scipy's curve_fit.

        Args:
            t (np.ndarray): the cumulative distance.

        Returns:
            np.ndarray: the parameters of the polynomial fit.
        """

        try:
            # FITTING scipy
            x, y = self.polar_coords
            params_x, _ = scipy.optimize.curve_fit(
                f=self.polynomial_callable,
                xdata=t,
                ydata=x, 
                p0=self.params_init,
            )
            params_y, _ = scipy.optimize.curve_fit(
                f=self.polynomial_callable,
                xdata=t,
                ydata=y,
                p0=self.params_init,
            )
            params = np.stack([params_x, params_y], axis=0).astype('float64')
        except Exception:
            # FAIL print
            if self.verbose > 1:
                print(
                    f"\033[1;31mFor cube {self.index:03d}, the polynomial fit failed. Going to " 
                    "next cube.\033[0m",
                    flush=self.flush,
                )
            params = np.empty((2, 0))
        return params

    def get_coords(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Gives the coordinates of the polynomial fit given the time and the parameters.

        Args:
            t (np.ndarray): the cumulative distance.
            params (np.ndarray): the parameters of the polynomial fit.

        Returns:
            np.ndarray: the coordinates of the polynomial fit.
        """

        # PARAMs
        params_x, params_y = params

        # COORDs
        x = self.polynomial_callable(t, *params_x)
        y = self.polynomial_callable(t, *params_y)
        return np.stack([x, y], axis=0)

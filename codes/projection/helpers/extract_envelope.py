"""
To extract the envelope created by Dr. Auchere and used in his Coronal Monsoon paper.
"""

# IMPORTs
import os
import scipy

# IMPORTs alias
import numpy as np

# IMPORTs sub
from PIL import Image
from typing import cast
import scipy.interpolate

# IMPORTs personal
from common import config
from codes.projection.format_data import ImageBorders, FitEnvelopes, EnvelopeInformation


class ExtractEnvelope:
    """
    To plot the envelope (and the corresponding middle path) created by Dr. Auchere and which was
    used in his Coronal Monsoon paper.
    """

    def __init__(
            self,
            polynomial_order: int,
            number_of_points: int,
            borders: ImageBorders,
            verbose: int | None = None,
        ) -> None:
        """
        To get the curve equations of the two PNGs (created by Dr. Auchere) of the envelope
        encompassing the Rainbow protuberance. From there, it also creates the middle path of that
        envelope. 
        There is also the possibility to create an intermediate plot just to check the result.

        Args:
            polynomial_order (int): the order of the polynomial used for the fit of the two curves
                (i.e. the envelope PNGs).
            number_of_points (int): the number of points used in the recreation of the envelopes
                and hence the number of points in the middle path curve.
            borders (ImageBorders): the radial distance and polar angle to borders consider for the
                image.
            image_shape (tuple[int, int]): the shape (in pixels) of the envelope image.
            verbose (int | None, optional): decides on the details in the prints. When None, it
                takes the value from the config file. Defaults to None.
        """

        # CONFIG attributes
        self.verbose: int = config.run.verbose if verbose is None else verbose

        # ATTRIBUTES setup
        self.polynomial_order = polynomial_order
        self.number_of_points = number_of_points
        self.borders = borders
        
        # PATHs setup
        self.paths = self.path_setup()

        # RUN
        self.main()

    @classmethod
    def get(
            cls,
            polynomial_order: int,
            number_of_points: int,
            borders: ImageBorders,
            verbose: int | None = None,
        ) -> EnvelopeInformation:
        """
        This classmethod directly gives the envelope and middle path positions in polar
        coordinates.

        Args:
            polynomial_order (int): the order of the polynomial used to fit the envelope png
                curves.
            number_of_points (int | float): the number of points used in the fitting result of the
                envelope and the middle path.
            borders (ImageBorders): the radial distance and polar angle to borders consider for the
                image.
            verbose (int | None, optional): decides on the details in the prints. When None, it
                takes the value from the config file. Defaults to None.

        Returns:
            EnvelopeInformation: the envelope information in polar coordinates.
        """

        instance = cls(
            polynomial_order=polynomial_order,
            number_of_points=number_of_points,
            borders=borders,
            verbose=verbose,
        )
        return instance.envelope_information
    
    def path_setup(self) -> dict[str, str]:
        """
        To get the paths to the needed directories and files.

        Returns:
            dict[str, str]: the needed paths.
        """

        # PATHs save
        paths = {'envelope': config.path.dir.data.result.envelope}
        return paths
    
    def main(self) -> None:
        """
        To get the envelope and corresponding middle path information.
        """

        # ENVELOPE processing
        lower_envelope = self.envelope_processing(
            filepath=os.path.join(self.paths['envelope'], 'rainbow_lower_path_v2.png'),
        )
        upper_envelope = self.envelope_processing(
            filepath=os.path.join(self.paths['envelope'], 'rainbow_upper_path_v2.png'),
        )

        # CURVE middle
        middle_path = FitEnvelopes(
            order=self.polynomial_order,
            polar_r=(lower_envelope.polar_r + upper_envelope.polar_r) / 2,
            polar_theta=(lower_envelope.polar_theta + upper_envelope.polar_theta) / 2,
        )

        # RESULTs
        self.envelope_information = EnvelopeInformation(
            middle=middle_path,
            upper=upper_envelope,
            lower=lower_envelope,
        )

    def envelope_processing(self, filepath: str) -> FitEnvelopes:
        """
        To process the envelope PNG files and get the curve equations and coordinates of each
        envelope path. 

        Args:
            filepath (str): the filepath to the envelope PNG file.

        Returns:
            FitEnvelopes: the corresponding envelope information in polar coordinates.
        """

        # IMAGE open
        im = Image.open(filepath)
        image = np.array(im)

        # DATA process
        arc_length = np.linspace(0, 1, self.number_of_points)
        theta_t_interp, r_theta_coeffs = self.get_image_coeffs(image)

        # DATA save
        polar_theta = theta_t_interp(arc_length)
        polar_r = self.get_polynomial_array(r_theta_coeffs, polar_theta)
        im.close()

        # DATA formatting
        envelope = FitEnvelopes(
            order=self.polynomial_order,
            polar_r=self.polar_positions(
                arr=polar_r,
                max_index=image.shape[0],
                borders=cast(
                    tuple[int, int],
                    tuple(int(distance * 1e3) for distance in self.borders.radial_distance)
                ),
            ),  # in km
            polar_theta=self.polar_positions(
                arr=polar_theta,
                max_index=image.shape[1],
                borders=self.borders.polar_angle,
            ),  # in degrees
        )
        return envelope

    def polar_positions(
            self,
            arr: np.ndarray,
            max_index: int,
            borders: tuple[int, int],
        ) -> np.ndarray:
        """
        Converts am array of index image positions to the corresponding polar positions.

        Args:
            arr (np.ndarray): the index image positions of a given fit or interpolation.
            max_index (int): the maximum value possible for that index (i.e. the border which is
                also the shape - 1 for that axis).
            borders (tuple[int, int]): the border values in polar for the given axis.

        Returns:
            np.ndarray: the corresponding polar positions for the given inputted ndarray.
        """

        # INDEXEs max = 1
        arr /= max_index 
        return arr * abs(borders[0] - borders[1]) + min(borders)

    def get_image_coeffs(
            self,
            image: np.ndarray,
        ) -> tuple[scipy.interpolate.interp1d, np.ndarray]:
        """
        To get the curve functions and coefficients of a given image of the envelope.

        Args:
            image (np.ndarray): the image of the top or bottom section of the envelope.

        Returns:
            tuple[scipy.interpolate.interp1d, np.ndarray]: the curve functions and coefficients of
                the polynomial fit.
        """

        # COORDs
        theta, radial_distance = np.where(image.T == 0)

        # AXES swap (Python reads from top left corner)
        radial_distance = -(radial_distance - image.shape[0])

        # CUMULATIVE DISTANCE
        cumulative_distance = np.empty(theta.shape, dtype='float64')
        cumulative_distance[0] = 0
        for i in range(1, len(theta)):
            cumulative_distance[i] = cumulative_distance[i - 1] + np.sqrt(
                (theta[i] - theta[i - 1])**2 + (radial_distance[i] - radial_distance[i - 1])**2
            )
        cumulative_distance /= cumulative_distance[-1]  # normalised

        # FITs + INTERPOLATION
        theta_t_interp = scipy.interpolate.interp1d(cumulative_distance, theta, kind='cubic')
        r_theta_coeffs: np.ndarray = cast(
            np.ndarray,
            np.polyfit(theta, radial_distance, self.polynomial_order),
        )
        results = (
            theta_t_interp,
            r_theta_coeffs[::-1],
        )
        return results

    def get_polynomial_array(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        To recreate a curve given the polynomial coefficients in the order p(x) = a + b * x +
        c * x**2 + ...

        Args:
            coeffs (np.ndarray): the coefficients of the polynomial function in the order
                p(x) = a + b * x + c * x**2 + ...
            x (np.ndarray): the x array for which the polynomial is a function of.

        Returns:
            np.ndarray: the results of the polynomial equation for the x values.
        """

        result: np.ndarray = cast(np.ndarray, 0)
        for i in range(self.polynomial_order + 1): result += coeffs[i] * x ** i
        return result


class CreateFitEnvelope:
    """
    To create the envelope of the 3D polynomial fit.
    """

    def __init__(self, coords: np.ndarray, radius: int | float):

        self.coords = coords  # (r, theta)
        self.radius = radius  # in km

    @classmethod
    def get(cls, coords: np.ndarray, radius: int | float) -> tuple[np.ndarray, np.ndarray]:
        """
        Class method to get the upper and lower limit of the envelope without needing to initialise
        the class when using it.

        Args:
            coords (np.ndarray): the coordinates (r, theta) of the polynomial fit as seen from SDO.
            radius (int | float): the radius to consider for the envelope (in km).

        Returns:
            tuple[np.ndarray, np.ndarray]: the upper and lower limits of the envelope.
        """

        instance = cls(coords=coords, radius=radius)        
        return instance.get_envelope()

    def get_envelope(self) -> tuple[np.ndarray, np.ndarray]:
        """
        To get the upper and lower limits of the envelope.

        Returns:
            tuple[np.ndarray, np.ndarray]: the upper and lower limits of the envelope.
        """
        
        # COORDs polar
        r, theta = self.coords

        # COORDs cartesian
        x = r * np.cos(np.deg2rad(theta))
        y = r * np.sin(np.deg2rad(theta))
        coords_cartesian = np.stack([x, y], axis=0)

        # VECTORs
        x_direction = x[2:] - x[:-2]
        y_direction = y[2:] - y[:-2]

        # VECTORs envelope
        solutions = np.stack([
            self.envelope_vectors(np.array([x, y]))
            for (x, y) in zip(x_direction, y_direction)
        ], axis=0)

        envelope_one = self.envelope_setup(coords_cartesian, solutions)
        envelope_two = self.envelope_setup(coords_cartesian, - solutions)
        envelope_up, envelope_down = self.get_up_down((envelope_one, envelope_two))
        return envelope_up, envelope_down
    
    def get_up_down(self, coords: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        To differentiate between the upper limit of the envelope and the lower one.

        Args:
            coords (tuple[np.ndarray, np.ndarray]): the coords separated into two curves which are
                a mix of the upper and lower limit.

        Returns:
            tuple[np.ndarray, np.ndarray]: the upper and lower coordinates for the envelope limit.
        """

        # DATA open
        one, two = coords

        # MASK on rho
        mask = one[0] >= two[0]

        # RESULTs init
        up_array = np.empty(one.shape)
        down_array = np.empty(one.shape)

        # POPULATE result
        up_array[:, mask] = one[:, mask]
        up_array[:, ~mask] = two[:, ~mask]
        down_array[:, mask] = two[:, mask]
        down_array[:, ~mask] = one[:, ~mask]
        return up_array, down_array
    
    def envelope_setup(self, coords_cartesian: np.ndarray, solutions: np.ndarray) -> np.ndarray:
        """
        To get the envelope coordinates from the solutions.

        Args:
            coords_cartesian (np.ndarray): the cartesian coordinates of the polynomial fit.
            solutions (np.ndarray): the solutions to the envelope problem.

        Returns:
            np.ndarray: the envelope coordinates.
        """
        
        # COORDs cartesian
        x, y = coords_cartesian

        # COORDs envelope
        envelope_x_down = [None] * len(solutions)  
        envelope_y_down = [None] * len(solutions)
        for index, (x_init, y_init) in enumerate(zip(x[1:-1], y[1:-1])):
            
            solution = solutions[index]

            envelope_x_down[index] = x_init + self.radius * solution[0]
            envelope_y_down[index] = y_init + self.radius * solution[1]

        # NDARRAY coords
        envelope_x_down = np.array(envelope_x_down)
        envelope_y_down = np.array(envelope_y_down)
        envelope_r = np.sqrt(envelope_x_down**2 + envelope_y_down**2)
        envelope_theta = np.rad2deg(np.atan2(envelope_y_down, envelope_x_down)) + 360

        envelope_coords = np.stack([envelope_r, envelope_theta], axis=0)
        return envelope_coords

    def envelope_vectors(self, vector: np.ndarray) -> np.ndarray:
        """
        To get the envelope vectors from the polynomial fit.

        Args:
            vector (np.ndarray): the vector representing the polynomial direction.

        Returns:
            np.ndarray: the envelope vectors.
        """

        # COMPONENTs vector
        a_0, a_1 = vector

        # SOLUTIONs
        b_0, b_1 = 1, 1
        if a_0 == 0:
            b_1 = 0
        elif a_1 == 0:
            b_0 = 0
        else:
            b_1 = - a_0 / a_1 
        
        # NORMALISATION
        N_b = np.sqrt(b_0**2 + b_1**2)

        # VECTOR
        solution = 1 / N_b * np.array([b_0, b_1])
        return solution

"""
To extract the envelope created by Dr. Auchere and used in his Coronal Monsoon paper.
"""

# IMPORTs
import os
import scipy

# IMPORTs sub
import PIL.Image
import scipy.interpolate

# IMPORTs alias
import PIL as pil
import numpy as np
import matplotlib.pyplot as plt

# IMPORTs personal
from common import root_path
from projection.projection_dataclasses import (
    ImageBorders, EnvelopeInformation, EnvelopeLimitInformation, EnvelopeMiddleInformation
)



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
            image_shape: tuple[int, int],
            plot: bool,
            verbose: int = 0,
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
            plot (bool, optional): to decide to plot the paths and middle path. Defaults to False.
            verbose (int, optional): decides on the details in the prints. Defaults to 0.
        """

        # ATTRIBUTES setup
        self.polynomial_order = polynomial_order
        self.number_of_points = number_of_points
        self.borders = borders
        self.image_shape = image_shape
        self.create_plot = plot        
        self.verbose = verbose
        
        # PATHs setup
        self.paths = self.path_setup()

        # RUN
        self.processing()

    @classmethod
    def get(
            cls,
            polynomial_order: int,
            number_of_points: int,
            borders: ImageBorders,
            image_shape: tuple[int, int] = (400, 1250),
            plot: bool = False,
            verbose: int = 0,
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
            image_shape (tuple[int, int]): the shape (in pixels) of the envelope image.
                Defaults to (400, 1250).
            plot (bool, optional): to decide to plot the results of the envelope fitting and middle
                path computation. Defaults to False.
            verbose (int, optional): decides on the details in the prints. Defaults to 0.

        Returns:
            EnvelopeInformation: the envelope information in polar coordinates.
        """

        instance = cls(
            polynomial_order=polynomial_order,
            number_of_points=number_of_points,
            borders=borders,
            image_shape=image_shape,
            plot=plot,
            verbose=verbose,
        )
        envelope = instance.get_envelope_in_polar()
        return envelope
    
    def path_setup(self) -> dict[str, str]:
        """
        To get the paths to the needed directories and files.

        Returns:
            dict[str, str]: the needed paths.
        """

        # PATHs setup
        main_path = os.path.join(root_path, '..')

        # PATHs save
        paths = {
            'main': main_path,
            'codes': root_path,
            'envelope': os.path.join(main_path, 'Work_done', 'Envelope'),
            'results': os.path.join(main_path, 'Work_done', 'Envelope', 'Extract_envelope'),
        }
        if self.create_plot: os.makedirs(paths['results'], exist_ok=True)
        return paths
    
    def processing(self) -> None:
        """
        To process the two envelope images and get the corresponding middle curve.
        """

        # SETUP
        masks: list[np.ndarray] = [None] * 2
        lower_path = EnvelopeLimitInformation()
        upper_path = EnvelopeLimitInformation()
        limit_paths = [lower_path, upper_path]
        envelope_filenames = ['rainbow_lower_path_v2.png', 'rainbow_upper_path_v2.png']

        # MIDDLE path
        x_t_curves: list[np.ndarray] = [None] * 2
        y_t_curves: list[np.ndarray] = [None] * 2
        for i, path in enumerate(os.path.join(
            self.paths['envelope'], filename) for filename in envelope_filenames
            ):
            # IMAGE open
            im = pil.Image.open(path)
            image = np.array(im)

            # DATA process
            x_normalised = np.linspace(0, 1, self.number_of_points)
            mask, x_t_function, y_x_coefs, x_range = self.get_image_coeffs(image)
            x = np.linspace(x_range[0], x_range[1], self.number_of_points)

            # DATA save
            masks[i] = mask
            limit_paths[i].x = x
            limit_paths[i].y = self.get_polynomial_array(y_x_coefs, x)
            x_t_curves[i] = x_t_function(x_normalised)
            y_t_curves[i] = self.get_polynomial_array(y_x_coefs, x_t_curves[i])
            im.close()
        # CURVE middle
        middle_path = EnvelopeMiddleInformation(
            x_t=(x_t_curves[0] + x_t_curves[1]) / 2,
            y_t=(y_t_curves[0] + y_t_curves[1]) / 2,
        )

        # RESULTs
        self.masks = masks[1] + masks[0] 
        self.envelope_information = EnvelopeInformation(
            middle=middle_path,
            upper=upper_path,
            lower=lower_path,
        )

        # PLOT
        if self.create_plot: self.plot(saving_name='extract_envelope_raw.png')

    def get_envelope_in_polar(self) -> EnvelopeInformation:
        """
        To get the change the envelope information from the image reference frame to polar
        coordinates.

        Returns:
            EnvelopeInformation: the envelope information in polar coordinates.
        """

        image_axis_curves = []
        middle_path = EnvelopeMiddleInformation()

        # MIDDLE path
        x_t_curves: list[np.ndarray] = [None] * 2
        y_t_curves: list[np.ndarray] = [None] * 2
        for axis in range(2):

            # DATA re-order
            axis_t_curve = self.envelope_information.middle[axis]
            image_axis_curve = [self.envelope_information[i][axis] for i in range(2)]
            
            # DATA new
            final_data = []
            for data in [axis_t_curve] + image_axis_curve:
                final_data.append(self.polar_positions(
                    arr=data,
                    max_index=self.image_shape[axis - 1] - 1,
                    borders=self.borders.polar_angle if axis==0 else self.borders.radial_distance,
                ))

            # RESULTs save   
            middle_path[axis] = final_data[0]
            image_axis_curves.append([final_data[index] for index in [1, 2]])

        # DATA formatting
        upper_path = EnvelopeLimitInformation(
            x=image_axis_curves[0][0],
            y=image_axis_curves[1][0],
        )
        lower_path = EnvelopeLimitInformation(
            x=image_axis_curves[0][1],
            y=image_axis_curves[1][1],
        )
        envelope_information = EnvelopeInformation(
            middle=middle_path,
            upper=upper_path,
            lower=lower_path,
        )

        # PLOT
        if self.create_plot:
            self.plot(
                saving_name='extract_envelope_final.png',
                extent=(
                    min(self.borders.polar_angle), max(self.borders.polar_angle),
                    min(self.borders.radial_distance), max(self.borders.radial_distance),
                ))
        return envelope_information

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
        ) -> tuple[np.ndarray, scipy.interpolate.interp1d, np.ndarray, tuple[float, float]]:
        """
        To get the curve functions and coefficients of a given image of the envelope.

        Args:
            image (np.ndarray): the image of the top or bottom section of the envelope.

        Returns:
            tuple[np.ndarray, scipy.interpolate.interp1d, np.ndarray, tuple[float, float]]: the
                mask gotten from the png file, the x(t) interpolation function with the y(x) fit
                coefficients with the range of the horizontal image indexes (i.e. the min and max
                of those values).
        """

        # COORDs
        x, y = np.where(image.T == 0)
        mask = np.zeros(image.shape, dtype='uint8')

        # AXES swap (Python reads from top left corner)
        y = -(y - image.shape[0])
        mask[y, x] = 1

        # CUMULATIVE DISTANCE
        cumulative_distance = np.empty((len(x),), dtype='float64')
        cumulative_distance[0] = 0
        for i in range(1, len(x)):
            cumulative_distance[i] = cumulative_distance[i - 1] + np.sqrt(
                (x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2
            )
        cumulative_distance /= cumulative_distance[-1]  # normalised

        # FITs + INTERPOLATION
        x_t_function = scipy.interpolate.interp1d(cumulative_distance, x, kind='cubic')
        y_x_coefs = np.polyfit(x, y, self.polynomial_order)
        return mask, x_t_function, y_x_coefs[::-1], (np.min(x), np.max(x))

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

        result = 0
        for i in range(self.polynomial_order + 1): result += coeffs[i] * x ** i
        return result

    def plot(
            self,
            saving_name: str,
            extent: tuple[float, float, float, float] | None = None,
        ) -> None:
        """
        To plot the results of the processing of the envelope and computation of the middle path.

        Args:
            saving_name (str): the name of the png file to be saved.
            extent (tuple[float, float, float, float] | None, optional): the extent to consider
                when doing the plt.imshow(). The extent values should be in polar coordinates
                (r, theta) when specified. Defaults to None.
        """

        # FIGURE setup
        plt.figure(figsize=(12, 5))
        plt.title('Envelope mask (yellow) vs 6th order fit.')

        # CURVE middle
        plt.plot(
            self.middle_t_curve[0],
            self.middle_t_curve[1],
            linestyle='--',
            color='black',
            label='middle path',
        )

        # ENVELOPE
        for path in self.envelope_y_x_curves:
            plt.plot(
                path[0],
                path[1],
                linestyle='--',
                linewidth=0.7,
                color='red',
                label='envelope',
            )

        # PNG
        plt.imshow(
            self.masks,
            alpha=0.5,
            origin='lower',
            interpolation='none',
            aspect='auto',
            label='mask',
            extent=extent,
        )
        plt.legend()
        plt.savefig(os.path.join(self.paths['results'], saving_name), dpi=1000)
        plt.close()
        if self.verbose > 0: print(f"File {saving_name} saved.")


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

        # VECTORs envelope #TODO: need to change this to use vector operations
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



if __name__=='__main__':
    instance = Envelope(
        polynomial_order=6,
        number_of_points=int(1e5),
        plot=True,
    )
    instance.get_curves_results()

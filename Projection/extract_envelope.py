"""
To extract the envelope used in Dr. Auchere's Coronal Monsoon paper.
"""

# Imports
import os
import glob
import scipy

# Aliases
import PIL as pil
import numpy as np
import matplotlib.pyplot as plt

# Sub-imports
import PIL.Image
import scipy.interpolate

# Personal imports
from common import Decorators

class Envelope:

    def __init__(
            self,
            polynomial_order: int,
            number_of_points: int,
            plot: bool = False,
        ) -> None:
        """
        To get the curve equations of the two PNGs (created by Dr. Auchere) of the envelope encompassing the Rainbow protuberance. 
        From there, it also creates the middle path of that envelope. 
        There is also the possibility to check the result inside a plot.

        Args:
            polynomial_order (int): the order of the polynomial used for the fit of the two curves (i.e. the envelope PNGs).
            number_of_points (int): the number of points used in the recreation of the envelopes and hence the number of points in the middle path curve.
            plot (bool, optional): to decide to plot the paths and middle path.
        """

        self.polynomial_order = polynomial_order
        self.number_of_points = int(number_of_points)
        self.create_plot = plot        
        # 
        self.borders = self.get_borders()
        self.paths = self.path_setup()

        self.processing()

    def get_borders(self):

        borders = {
            'polar angle': (245, 295),
            'height [Mm]': (690, 870),
            'image shape': (400, 1250)
        }
        return borders
    
    def path_setup(self) -> dict[str, str]:
        """
        To get the paths to the needed directories and files.

        Raises:
            ValueError: if the main path is not found.

        Returns:
            dict[str, str]: the needed paths.
        """

        # Check main path
        main_path = '/home/avoyeux/Documents/avoyeux/'
        if not os.path.exists(main_path): main_path = '/home/avoyeux/old_project/avoyeux/'
        if not os.path.exists(main_path): raise ValueError(f"\033[1;31mThe main path {main_path} not found.")
        code_path = os.path.join(main_path, 'python_codes')

        # Save paths
        paths = {
            'main': main_path,
            'codes': code_path,
            'envelope': os.path.join(main_path, 'Work_done', 'Envelope'),
            'results': os.path.join(main_path, 'Work_done', 'Envelope', 'Extract_envelope')
        }
        if self.create_plot: os.makedirs(paths['results'], exist_ok=True)
        return paths
    
    def processing(self) -> None:
        """
        To process the two envelope images and get the corresponding middle curve.
        """

        # Initialisation
        x_t_curves = [None] * 2
        y_t_curves = [None] * 2
        y_x_curves = [None] * 2
        for i, path in enumerate(glob.glob(os.path.join(self.paths['envelope'], '*.png'))):
            # Get image data
            im = pil.Image.open(path)
            image = np.array(im)
            print(image.shape)

            # Process data
            x_normalised = np.linspace(0, 1, self.number_of_points)
            x_t_function, y_x_coefs, x_range = self.get_image_coefs(image)
            x = np.linspace(x_range[0], x_range[1], self.number_of_points)

            # Save data
            y_x_curves[i] = (x, self.get_polynomial_array(y_x_coefs, x))
            x_t_curves[i] = x_t_function(x_normalised)
            y_t_curves[i] = self.get_polynomial_array(y_x_coefs, x_t_curves[i])
            im.close()

        # Compute middle path
        middle_x_t_curve = (x_t_curves[0] + x_t_curves[1]) / 2
        middle_y_t_curve = (y_t_curves[0] + y_t_curves[1]) / 2

        self.middle_t_curve = [middle_x_t_curve, middle_y_t_curve]
        self.envelope_y_x_curves = y_x_curves 

        # Plotting the results
        if self.create_plot: self.plot()

    def get_curves_results(self):


        image_axis_curves = []
        for axis in range(2):

            # Constants init
            if axis==0:
                borders = 'polar angle'
                axis_opos = 1
            else:
                borders = 'height [Mm]'
                axis_opos = 0

            axis_t_curve = self.middle_t_curve[axis]

            image_axis_curve= []
            for envelope in self.envelope_y_x_curves:
                image_axis_curve.append(envelope[axis])
            
            final_data = []
            for data in [axis_t_curve] + image_axis_curve:

                final_data.append(self.polar_positions(
                    arr=data,
                    max_index=self.borders['image shape'][axis_opos],
                    borders=self.borders[borders],
                ))

            # Keep new data    
            self.middle_t_curve[axis] = final_data[0]
            image_axis_curves.append([final_data[index] for index in [1, 2]])
        
        self.envelope_y_x_curves = [
            (image_axis_curves[0][image], image_axis_curves[1][image])
            for image in range(2)
        ]

        self.plot()
        # Plot new data

    @classmethod
    def get(
            cls,
            polynomial_order: int,
            number_of_points: int | float,
            plot: bool,
        ) -> tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:

        instance = cls(
            polynomial_order=polynomial_order,
            number_of_points=number_of_points,
            plot=plot,
        )
        instance.get_curves_results()
        return instance.middle_t_curve, instance.envelope_y_x_curves

    def polar_positions(self, arr: np.ndarray, max_index: int, borders: tuple[int, int]) -> np.ndarray:

        # Normalise image indexes to 1
        arr /= max_index
        return arr * abs(borders[0] - borders[1]) + min(borders)

    def get_image_coefs(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        """
        To get the curve functions and coefficients of a given image of the envelope.

        Args:
            image (np.ndarray): the image of the top or bottom section of the envelope.

        Returns:
            tuple[np.ndarray, np.ndarray, tuple[float, float]]: the x(t) interpolation function with the y(x) fit coefficients with the range of the 
                horizontal image indexes (i.e. the min and max of those values).
        """

        x, y = np.where(image.T == 0)

        # Swap the image axes as python reads an image from the top left corner
        y = -(y - np.max(y))  

        cumulative_distance = np.empty((len(x),), dtype='float64')
        cumulative_distance[0] = 0
        for i in range(1, len(x)):
            cumulative_distance[i] = cumulative_distance[i - 1] + np.sqrt(
                (x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2
            )
        cumulative_distance /= cumulative_distance[-1]  # normalised

        x_t_function = scipy.interpolate.interp1d(cumulative_distance, x, kind='cubic')
        y_x_coefs = np.polyfit(x, y, self.polynomial_order)
        return x_t_function, y_x_coefs[::-1], (np.min(x), np.max(x))

    def get_polynomial_array(self, coeffs: list[int | float], x: np.ndarray) -> np.ndarray:
        """
        To recreate a curve given the polynomial coefficients in the order p(x) = a + b * x + c * x**2 + ...

        Args:
            coeffs (list[int  |  float]): the coefficients of the polynomial function in the order p(x) = a + b * x + c * x**2 + ...
            x (np.ndarray): the x array for which the polynomial is a function of.

        Returns:
            np.ndarray: the results of the polynomial equation for the x values.
        """

        # Initialisation
        result = 0

        # Calculating the polynomial
        for i in range(self.polynomial_order + 1): result += coeffs[i] * x ** i
        return result

    def plot(self) -> None:
        """
        To plot the results of the processing of the envelope and computation of the middle path.

        Args:
            middle_path (tuple[np.ndarray, np.ndarray]): the x and y position arrays for the middle of the envelope path.
            y_x_curve (list[tuple[np.ndarray, np.ndarray]]): two tuples each representing one of the envelope images by containing
                the y(x) values of the envelope interpolation and the corresponding x values. 
        """

        # Set up
        plt.figure()
        plt.scatter(self.middle_t_curve[0], self.middle_t_curve[1], label='middle')
        for path in self.envelope_y_x_curves:
            plt.scatter(path[0], path[1])
        plt.legend()
        plt.savefig(os.path.join(self.paths['results'], 'extract_middle_path.png'), dpi=500)
        plt.close()


if __name__=='__main__':
    instance = Envelope(
        polynomial_order=6,
        number_of_points=1e5,
        plot=False,
    )

    instance.get_curves_results()
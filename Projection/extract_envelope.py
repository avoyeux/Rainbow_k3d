"""
To extract the envelope created by Dr. Auchere and used in his Coronal Monsoon paper.
"""

# IMPORTs
import os
import glob
import scipy

# IMPORTs sub
import PIL.Image
import scipy.interpolate

# IMPORTs alias
import PIL as pil
import numpy as np
import matplotlib.pyplot as plt



class Envelope:
    """
    To plot the envelope (and the corresponding middle path) created by Dr. Auchere and which was
    used in his Coronal Monsoon paper.
    
    Raises:
        ValueError: if the main path to the code directories is not found.
    """

    # Image information
    borders = {
        'polar angle': (245, 295),
        'radial distance': (690, 870),
        'image shape': (400, 1250)
    }

    def __init__(
            self,
            polynomial_order: int,
            number_of_points: int,
            plot: bool = False,
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
            plot (bool, optional): to decide to plot the paths and middle path. Defaults to False.
            verbose (int, optional): decides on the details in the prints. Defaults to 0.
        """

        # ATTRIBUTES setup
        self.polynomial_order = polynomial_order
        self.number_of_points = number_of_points
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
            number_of_points: int | float,
            plot: bool,
        ) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
        """
        To get the results of the envelope processing by just class this classmethod function.

        Args:
            polynomial_order (int): the order of the polynomial used to fit the envelope png
                curves.
            number_of_points (int | float): the number of points used in the fitting result of the
                envelope and the middle path.
            plot (bool): to decide to plot the results of the envelope fitting and middle path
                computation.

        Returns:
            tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]: the first object in the
                tuple is a list with the x_t values and y_t values of the middle path curve. The
                second object in the tuple is a list for the upper and lower part of the envelope
                and containing the x values and corresponding y_x values for those curves.
        """

        instance = cls(
            polynomial_order=polynomial_order,
            number_of_points=number_of_points,
            plot=plot,
        )
        instance.get_curves_results()
        return instance.middle_t_curve, instance.envelope_y_x_curves
    
    def path_setup(self) -> dict[str, str]:
        """
        To get the paths to the needed directories and files.

        Raises:
            ValueError: if the main path is not found.

        Returns:
            dict[str, str]: the needed paths.
        """

        # CHECK path
        main_path = '/home/avoyeux/Documents/avoyeux/'
        if not os.path.exists(main_path): main_path = '/home/avoyeux/old_project/avoyeux/'
        if not os.path.exists(main_path):
            raise ValueError(f"\033[1;31mThe main path {main_path} not found.")
        code_path = os.path.join(main_path, 'python_codes')

        # PATHs save
        paths = {
            'main': main_path,
            'codes': code_path,
            'envelope': os.path.join(main_path, 'Work_done', 'Envelope'),
            'results': os.path.join(main_path, 'Work_done', 'Envelope', 'Extract_envelope'),
        }
        if self.create_plot: os.makedirs(paths['results'], exist_ok=True)
        return paths
    
    def processing(self) -> None:
        """
        To process the two envelope images and get the corresponding middle curve.
        """

        # INIT
        masks = [None] * 2
        x_t_curves = [None] * 2
        y_t_curves = [None] * 2
        y_x_curves = [None] * 2
        for i, path in enumerate(glob.glob(os.path.join(self.paths['envelope'], '*.png'))):
            # IMAGE open
            im = pil.Image.open(path)
            image = np.array(im)

            # DATA process
            x_normalised = np.linspace(0, 1, self.number_of_points)
            mask, x_t_function, y_x_coefs, x_range = self.get_image_coeffs(image)
            x = np.linspace(x_range[0], x_range[1], self.number_of_points)

            # DATA save
            masks[i] = mask
            y_x_curves[i] = (x, self.get_polynomial_array(y_x_coefs, x))
            x_t_curves[i] = x_t_function(x_normalised)
            y_t_curves[i] = self.get_polynomial_array(y_x_coefs, x_t_curves[i])
            im.close()
        # CURVE middle
        middle_x_t_curve = (x_t_curves[0] + x_t_curves[1]) / 2
        middle_y_t_curve = (y_t_curves[0] + y_t_curves[1]) / 2

        # RESULTs
        mask += masks[0]
        self.masks = mask
        self.middle_t_curve = [middle_x_t_curve, middle_y_t_curve]
        self.envelope_y_x_curves = y_x_curves 

        # PLOT
        if self.create_plot: self.plot(saving_name='extract_envelope_raw.png')

    def get_curves_results(self) -> None:
        """
        To get the change the envelope information from the image reference frame to polar
        coordinates.
        """

        image_axis_curves = []
        for axis in range(2):

            # CONSTANTs
            if axis==0:
                borders = 'polar angle'
                axis_opos = 1
            else:
                borders = 'radial distance'
                axis_opos = 0

            # DATA re-order
            axis_t_curve = self.middle_t_curve[axis]
            image_axis_curve= []
            for envelope in self.envelope_y_x_curves:
                image_axis_curve.append(envelope[axis])
            
            # DATA new
            final_data = []
            for data in [axis_t_curve] + image_axis_curve:
                final_data.append(self.polar_positions(
                    arr=data,
                    max_index=self.borders['image shape'][axis_opos] - 1,
                    borders=self.borders[borders],
                ))

            # RESULTs save   
            self.middle_t_curve[axis] = final_data[0]
            image_axis_curves.append([final_data[index] for index in [1, 2]])
        self.envelope_y_x_curves = [
            (image_axis_curves[0][image], image_axis_curves[1][image])
            for image in range(2)
        ]

        # PLOT
        if self.create_plot:
            self.plot(
                saving_name='extract_envelope_final.png',
                extent=(
                    min(self.borders['polar angle']), max(self.borders['polar angle']),
                    min(self.borders['radial distance']), max(self.borders['radial distance']),
                ))

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
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
        """
        To get the curve functions and coefficients of a given image of the envelope.

        Args:
            image (np.ndarray): the image of the top or bottom section of the envelope.

        Returns:
            tuple[np.ndarray, np.ndarray, tuple[float, float]]: the mask gotten from the png file,
                the x(t) interpolation function with the y(x) fit coefficients with the range of
                the horizontal image indexes (i.e. the min and max of those values).
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



if __name__=='__main__':
    instance = Envelope(
        polynomial_order=6,
        number_of_points=int(1e5),
        plot=True,
    )
    instance.get_curves_results()
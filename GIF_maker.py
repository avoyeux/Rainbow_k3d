import os
import imageio.v3 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class DataFinder:
    """
    To get the data that we want to use.
    """

    def __init__(self, time_interval='2h', versions=['v0', 'v1', 'v2']):

        # Arguments
        self.time_interval = time_interval
        self.versions = versions

        # Initial class arguments
        self.numbers = np.arange(1, 78)

        # Functions
        self.Paths()

    def Paths(self):
        """
        Creating the needed dictionary paths.
        """

        main_path = '/home/avoyeux/Desktop/avoyeux/'
        self.paths = {'Main': main_path,
                      'Screenshots': os.path.join(main_path, 'Screenshots3')}


class Figure_making(DataFinder):
    """
    Making figures with the different point of views saved.
    """

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        # Functions
        self.Updating_paths()
        self.Screenshot_paths()
        self.All_plots()


    def Updating_paths(self):
        """
        Creating the paths dictionary.
        """

        self.paths['Plots'] = os.path.join(self.paths['Main'], 'Plots_3D_3')
        os.makedirs(self.paths['Plots'], exist_ok=True)

    def Screenshot_paths(self):
        """
        To get the paths to the necessary images.
        """

        self.image_names = np.empty((len(self.versions), len(self.numbers)), dtype='U30') 

        for index1, version in enumerate(self.versions):  # TODO: need to change this but don't remember how
            for index2, number in enumerate(self.numbers):
                # filename = f'interval{self.time_interval}_{number}_{version}.png'
                filename = f'Plot_{number}_{version}.png'
                self.image_names[index1, index2] = filename


    def All_plots(self):
        """
        Creating all the plots.
        """

        for loop, number in enumerate(self.numbers):
            images = [mpimg.imread(os.path.join(self.paths['Screenshots'], self.image_names[version, loop])) 
                           for version in range(len(self.versions))]
                        
            self.Plotting_func(images, number)

    def Plotting_func(self, images, number):
        """
        Plotting the data.
        """

        fig, axs = plt.subplots(2, len(self.versions), figsize=(4, 4))
        plt.title(f'For {self.time_interval}')

        for loop, version in enumerate(self.versions):
            image = images[loop]
            if image.shape[0] < image.shape[1]:
                image = np.rot90(image)
            axs[0, loop].imshow(image, interpolation='none')
            axs[0, loop].axis('off')
            axs[0, loop].set_title(f'img{number}_{version}')

        for loop in range(len(self.versions)):
            # add the images of SDO and stereo  
        
        plt.tight_layout()
        # fig_name = f'Plot_{self.time_interval}_{number:02d}.png'
        fig_name = f'Plot_daytrace_{number:02d}.png'
        plt.savefig(os.path.join(self.paths['Plots'], fig_name), bbox_inches='tight', pad_inches=0.05, dpi=1000)
        plt.close()
  

class GIF_for_figures(Figure_making):
    """
    To make the GIFs for the plots.
    """

    def __init__(self, fps=0.5, **kwargs):

        super().__init__(**kwargs)

        # Arguments
        self.fps = fps

        # Functions
        self.Updating_paths2()
        self.Making_gifs()

    def Updating_paths2(self):
        """
        Updating the paths again.
        """
        
        self.paths['GIFs'] = os.path.join(self.paths['Main'], 'GIFs3')
        os.makedirs(self.paths['GIFs'], exist_ok=True)

    def Making_gifs(self):
        """
        Creating the GIF.
        """

        # imgs_paths = [os.path.join(self.paths['Plots'], f'Plot_{self.time_interval}_{number:02d}.png') for number in self.numbers]
        imgs_paths = [os.path.join(self.paths['Plots'], f'Plot_daytrace_{number:02d}.png') for number in self.numbers]

        images = [imageio.imread(img) for img in imgs_paths]
        print(f'total size is {round(np.array(images).nbytes / 2**20)}MB')
        # imageio.imwrite(os.path.join(self.paths['GIFs'], f'GIF_plot_{self.time_interval}.gif'), images, duration=self.fps*1000)
        imageio.imwrite(os.path.join(self.paths['GIFs'], 'GIF_daytrace.gif'), images, duration=self.fps*1000)

        print('Gif is done.')

class UniqGIFs(DataFinder):

    def __init__(self, fps=0.5, **kwargs):
        super().__init__(**kwargs)

        self.fps = fps

        self.Updating_paths()
        self.Gif()

    def Updating_paths(self):
        self.paths['GIFs'] = os.path.join(self.paths['Main'], 'GIFs2')
        os.makedirs(self.paths['GIFs'], exist_ok=True)

    def Gif(self):
        for version in self.versions:
            imgs_paths = [os.path.join(self.paths['Screenshots'], f'interval{self.time_interval}_{number}_{version}.png')
                        for number in self.numbers]
            images = []
            for img in imgs_paths:
                image = imageio.imread(img)
                images.append(image)
            imageio.imwrite(os.path.join(self.paths['GIFs'], f'GIF_{self.time_interval}_{version}.gif'), images,
                            duration=self.fps*1000)


def making_gifs():
    """
    To make the GIF if the plots have already been created.
    """
    
    paths = {'Plots': os.path.join('/home/avoyeux/Desktop/avoyeux/', 'Plots_3D'),
             'GIFs': os.path.join('/home/avoyeux/Desktop/avoyeux/', 'GIFs')}
    
    numbers = np.arange(2, 78)

    imgs_paths = [os.path.join(paths['Plots'], f'Plot_30min_{number:02d}.png') for number in numbers]
    images = []
    for loop, img in enumerate(imgs_paths):
        image = imageio.imread(img)
        images.append(image)
        print(f'image shape is {image.shape}')
        print(f'image type is {image.dtype}')
        print(f'loop is {loop} with {image.nbytes / 2**20}MB')
        print(f'total size is {np.array(images).nbytes / 2**20}MB')

    # imageio.mimsave(os.path.join(self.paths['GIFs'], f'GIF_Plot_{self.time_interval}1.gif'), images)
    imageio.imwrite(os.path.join(paths['GIFs'], f'GIF_Plot_30min1.gif'), images, fps=0.5*1000)

    print('Gif is done.')


if __name__ == '__main__':
    GIF_for_figures(fps=0.3, versions=['v0', 'v1', 'v2'], time_interval='')

    # making_gifs()

    # UniqGIFs(time_interval='1h21min', versions=['v0_test_5seconds'])

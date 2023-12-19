from Animation_3D_main import CustomDate, Data
import matplotlib.pyplot as plt
import numpy as np


class Looking(Data):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def Tests(self):

        print(f'self.cubes shape is {self.cubes.shape}')
        print(f'self.cubes_lineofsight_stereo shape is {self.cubes_lineofsight_STEREO.shape}')
        print(f'The other is {self.cubes_lineofsight_SDO.shape}')


    def Plotting(self):

        lineofsight_SDO = self.cubes_lineofsight_SDO[0]
        lineofsight_stereo = self.cubes_lineofsight_STEREO[0]


        for loop in range(3):

            imagesdo = np.any(lineofsight_SDO, axis=loop)
            imagestereo = np.any(lineofsight_stereo, axis=loop)

            imagesdo_name = f'sdo_{loop}.png'
            imagestereo_name=f'stereo{loop}.png'

            plt.figure(figsize=(8,8))
            plt.imshow(imagesdo, interpolation='none')
            plt.axis(True)
            plt.savefig(imagesdo_name, dpi=200)
            plt.close()

            plt.figure(figsize=(8,8))
            plt.imshow(imagestereo, interpolation='none')
            plt.axis(True)
            plt.savefig(imagestereo_name, dpi=200)
            plt.close()

    def Max_data(self):
        where_SDO = np.where(self.cubes_lineofsight_SDO[0]==1)
        where_stereo = np.where(self.cubes_lineofsight_STEREO[0]==1)

        print('For SDO:')
        print(f'max axis0 is {np.max(where_SDO[0])}')
        print(f'max axis1 is {np.max(where_SDO[1])}')
        print(f'max axis2 is {np.max(where_SDO[2])}')
        print('For stereo:')
        print(f'max axis0 is {np.max(where_stereo[0])}')
        print(f'max axis1 is {np.max(where_stereo[1])}')
        print(f'max axis2 is {np.max(where_stereo[2])}')


if __name__=='__main__':

    testing = Looking(all_data=True, line_of_sight=True)

    testing.Tests()
    testing.Max_data()

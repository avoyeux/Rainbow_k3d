import numpy as np
import glob
import os
import cv2
from mask2cube import Mask2cube
from mask2cube_sdo import Mask2cube_sdo
from astropy.io import fits
from scipy.ndimage import label

path = ''  # TODO: need to find the paths to the needed SDO data and the used masks. 
path_sdo = ''
list_png = glob.glob(os.path.join(path, '*.png'))
list_sdo = glob.glob(os.path.join(path, '*.fits.gz'))

nfiles  = len(list_png)  # TODO: not sure if this is right but it should be

solar_r = 6.96e5

# No clue yet what these are
img_x = 0 
img_y = 0
img_idx = 0

# 3D array boundaries (in a cartesian coordinate system I guess)
xt_min = -solar_r * 1.25 # don't know why we are using negative values 
yt_min = -solar_r * 0.5
zt_min = -solar_r * 0.32
xt_max = -solar_r * 0.95
yt_max = solar_r * 0  # why is this one null? 
zt_max = solar_r * 0.32
# TODO: I should try and draw the 3D shape

# Resolution 
dx = 0.003 * solar_r * 0.7068
dy = 0.003 * solar_r * 0.7068
dz = 0.003 * solar_r * 0.7068
# TODO: I should try and get the constant values straight from the fits header so that it makes more sense.


# In the Idl code, the way the next step is done is different. That being said, pretty sure what I am going to do gives 
# the same result on Python.
xt_vals = np.arange(xt_min, xt_max, dx, dtype='float64')
yt_vals = np.arange(yt_min, yt_max, dy, dtype='float64')
zt_vals = np.arange(zt_min, zt_max, dz, dtype='float64')

# 3D matrix shape
nx = len(xt_vals)
ny = len(yt_vals)
nz = len(zt_vals)

# The filled 3D matrices 
xt = np.tile(xt_vals[:, None, None], (1, ny, nz))
yt = np.tile(yt_vals[None, :, None], (nx, 1, nz))
zt = np.tile(zt_vals[None, None, :], (nx, ny, 1))

# Not sure what these are for but they are used later
cube3 = np.zeros((nx, ny, nz), dtype='float64')  #TODO: need to check if I can use np.empty later on
cube4 = np.zeros(np.shape(cube3))

bin_sdo = 8
bin_stereo = 2 # TODO: need to ask where these values come from

for i in range(nfiles):  # TODO: might be able to change this to a np.along_axis thingy
    png_path = list_png[i]
    image = cv2.imread(png_path)  #TODO: will need to do some further treatment as the given image is in RGB values uint8 
    mask = np.any(image < 190, axis=-1).astype('bool')
    # TODO: IMPORTANT: I need to say that I changed how I read the mask as I felt the last method took the png interpolation
    # too much into account 

    # Getting the image number
    base_name_w_extensions = os.path.basename(png_path)
    base_name = os.path.splitext(base_name_w_extensions)[0]
    number = base_name.lstrip('frame').lstrip('0')
    number = int(number) if number else 0

    cube1 = Mask2cube(mask, datainfos[int(number)], xt, yt, zt, latcen, latwidth, loncen, lonwidth, dlat, dlon, mid_date,
                                temp_img_x_stereo, temp_img_y_stereo)  # TODO: I don't understand at all what is happening in the code then
    
    img_x_stereo = np.zeros(np.shape(xt))
    img_x_stereo[img_idx_stereo] = temp_img_x_stereo
    img_y_stereo = np.zeros(np.shape(yt))
    img_y_stereo[img_idx_stereo] = temp_img_y_stereo

    sdo_name = os.path.join(path_sdo, f'AIA_fullhead_{number:03d}.fits.gz')

    hdul_sdo = fits.open(sdo_name)
    hdr = hdul_sdo[0].header  #TODO: need to check the format of the fits
    mask_sdo = hdul_sdo[0].data.astype('uint8')

    # TODO: ATTENTION in the idl code, there is also a fitshead2struct line but most likely not needed when using astropy
    cube2 = Mask2cube_sdo(mask_sdo, hdr, xt, yt, zt, temp_img_x_sdo, temp_img_y_sdo, img_idx_sdo)

    img_x_sdo = np.zeros(np.shape(xt))
    img_x_sdo[img_idx_sdo] = temp_img_x_sdo
    img_y_sdo = np.zeros(np.shape(yt))
    img_y_sdo[img_idx_sdo] = temp_img_y_sdo

    cube = (cube1 & cube2)  #TODO: why are we doing a bitwise operation? Aren't we loosing information?
    cube3 = cube3 + cube  #TODO: I am completely lost right now...

    loop = np.where(cube == 1)
    count = len(loop[0])

    if count > 0:
        regions, num_features = label(cube)
        regions = regions.astype('uint64')

        x = xt[loop]
        y = yt[loop]
        z = zt[loop]
        regions = regions[loop]  #TODO: not sure if this will work but we will see

        for probe in range(2):

            if probe == 0:
                img_x = img_x_sdo[loop]
                img_y = img_y_sdo[loop]
                naxis1 = hdr['NAXIS1']
                bins = bin_sdo
            else:
                img_x = img_x_stereo[loop]
                img_y = img_y_stereo[loop]
                naxis1 = np.round(lonwidth / dlon)
                bins = bin_stereo
            
            img_x = np.round(img_x / bins)
            img_y = np.round(img_y / bins)

            index = img_x.astype('int') + img_y.astype('int') * (naxis1 / bins).astype('int')
            uniq_values, uniq_indices = np.unique(index, return_index=True)
            uniq_idx = index[np.sort(uniq_indices)]

            for ipixel in range(len(uniq_idx)):
                idx_voxels = np.where(index == uniq_idx[ipixel])
                min_val = np.min(regions[idx_voxels])
                max_val = np.max(regions[idx_voxels])

                if min_val == max_val:
                    cube[(x[idx_voxels] - xt_min) / dx, (y[idx_voxels] - yt_min) / dy, (z[idx_voxels] - zt_min) / dz] \
                        += 2**(1 + probe) 

        cube4 += (cube >= 2)  # TODO: this surely doesn't do the same thing on python
    
    cube_filename = f'cube{number:03d}.npz'
    np.savez(cube_filename, cube=cube, xt_min=xt_min, xt_max=xt_max, yt_min=yt_min, yt_max=yt_max, zt_min=zt_min, zt_max=zt_max, 
             dx=dx, dy=dy, dz=dz)

cube = cube3 / nfiles
cubesum_filename = 'cube_sum.npz'
np.savez(cubesum_filename, cube=cube, xt_min=xt_min, xt_max=xt_max, yt_min=yt_min, yt_max=yt_max, zt_min=zt_min, zt_max=zt_max,
          dx=dx, dy=dy, dz=dz)
cube = cube4 / nfiles
cubenam_filename = 'cube_nam.npz'
np.savez(cubenam_filename, cube=cube, xt_min=xt_min, xt_max=xt_max, yt_min=yt_min, yt_max=yt_max, zt_min=zt_min, zt_max=zt_max,
          dx=dx, dy=dy, dz=dz)
# RAINBOW event animation using the K3D library.
This github repository is made up of codes related to the 3D visualisation of a solar protuberance using acquisitions made by STEREO and SDO from the 23.07.2012 till 25.07.2012. This was done to possibly study the periodic properties of coronal rain. The K3D library is a Jupyter library specialised in 3D visualisations.

The codes in said repository are as follows:

- `Animation_3D_main.py`: the Python code where all the data preprocessing and the implementation to the k3d library is done. This code is then used in the `new_3D_animation.ipynb` Jupyter notebook code.

- `new_3D_animation.ipynb`: Jupyter notebook code used to run the aforementioned Python code. Inside said notebook, in depth explanations on the possible arguments is also given.

- `k3d_voxel_stats.py`: Python code used to save some possibly useful statistics on the protuberance positions. The results are saved in a .csv file.

- `stats_to_plot.py`: Python code to plot the data from the .csv file created with `k3d_voxel_stats.py`.

- `mask_to_white.py`: Python code to change the initial un processed masks to uint8 greyscale masks where initial null values are seen as white (i.e. 255).

- `figures_sdostereo.py`: Python code using the k3d screenshots with the stereo and sdo acquisitions to create 5 image plots to then be used in a gif. The GIF showcases the time evolution of the protuberance following the point of view of SDO and STEREO.

- `common_alf.py`: Python code with some of my most 'polyvalent' functions. It is still a work in progress.

- `Figure_making.py`: Python code for plotting 3 images figures with the mask contours and the latitude and longitude gridlines. A corresponding GIF is also created.

- `GIF_maker.py`: Older Python code to create GIFs. This code might be deprecated but kept for now. Will most likely be deleted in the final version of this repository.

- `GIF_maker2.py`: Python code used to cut in 2 the images gotten from the mp4 video in Sir Auchere's presentation. 
import numpy as np

def Mask2cube(mask, datainfos, x, y, z, latcen, latwidth, loncen, lonwidth, dlat, dlon, mid_date, lon, lat):

    solar_r = 6.96e5 * 1.01  # I have no clue why the value for solar_r is different here 


    a = 14.51 # TODO: need to ask what these are and if I can just get them from the headers
    b = -3.12
    c = 0.34 

    # Boundaries for the longitude and latitude 
    lonmin = loncen - lonwidth / 2 
    lonmax = loncen + lonwidth / 2
    latmin = latcen - latwidth / 2
    latmax = latcen + latwidth / 2

    # Shape for the longitude and latitude
    npx = np.round(lonwidth / dlon)
    npy = np.round(latwidth / dlat)

    r = np.sqrt(x**2 + y**2 + z**2)
    corona = np.where(r >= solar_r)
    count = len(corona[0])

    if count > 0:
        xt = x[corona]  # TODO: this should give a 1D array so be careful
        yt = y[corona] # TODO: I got no clue how this should work. Why do we use a 3D array to make a 1D array
        zt = z[corona]

        # Datainfo stats
        longitude_rad = np.deg2rad(datainfos.lon)
        latitude_rad = np.deg2rad(datainfos.lat)
        distance = datainfos.dist
        date = datainfos.date

        dum = xt * np.cos(longitude_rad) + yt * np.sin(longitude_rad)  # TODO: need to check here for the different shapes
        yt = -xt * np.sin(longitude_rad) + yt * np.cos(longitude_rad)
        xt = dum

        dum = xt * np.cos(latitude_rad) + zt * np.sin(latitude_rad)
        zt = -xt * np.sin(latitude_rad) + zt * np.cos(latitude_rad)
        xt = dum

        omega = np.arctan2(solar_r, distance)
        gamma = np.arctan2(np.sqrt(yt**2 + zt**2), distance - xt)

        disk = np.where((gamma <= omega) & (xt >= (solar_r * np.sin(omega))))
        count = len(disk[0])

        if count > 0:
            yt = yt[disk]  # TODO: this might not work but not sure
            zt = zt[disk]

            alpha = np.arctan2(yt, distance - xt)
            beta = np.arctan2(zt, distance - xt)

            lcos_g = (2 * distance - np.sqrt(4 * distance**2 - 4 * (1 + np.tan(alpha)**2 + np.tan(beta)**2) \
                * (distance**2 -solar_r**2))) / (2 * (1 + np.tan(alpha)**2 + np.tan(beta)**2))
            
            xt = distance - lcos_g
            yt = lcos_g * np.tan(alpha)
            zt = lcos_g * np.tan(beta)

            ys = yt
            xs = xt * np.cos(latitude_rad) - zt * np.sin(latitude_rad)
            zs = zt * np.cos(latitude_rad) + xt * np.sin(latitude_rad)

            lat = np.arcsin(zs / solar_r)
            lon = np.rad2deg(np.arctan2(ys, xs) + longitude_rad)

            delta_lon = -(date - mid_date) * (a + b * np.sin(lat)**2 + c * np.sin(lat)**4 - 14.18)
            lon = (lon + delta_lon + 360) % 360  #TODO: The +360 is completly useless no?

            lon = np.round(npx * (lon - lonmin) / lonwidth)
            lat = np.round(npy * (np.rad2deg(lat) - latmin) / latwidth)
            cube = np.zeros(np.shape(x))
            
            cube[corona[disk]] = mask[lon, lat]  #TODO: this would probably not work

            img_idx = corona[disk]  #img_idx that is an argument for the function is completely useless then, and this is useless too

            return cube
        else:
            return -1
    else:
        return -1 









import numpy as np

def Mask2cube_sdo(mask, hdr, x, y, z, img_x, img_y, corona):
    
    solar_r = 6.96e5 * 1.01 # TODO: again, the length is different than for the main code

    r = np.sqrt(x**2 + y**2 + z**2)
    corona = np.where(r >= solar_r)
    count = len(corona[0])

    if count > 0:
        xt = x[corona]
        yt = y[corona]
        zt = z[corona]

        # Important stats
        crln_obs_rad = np.deg2rad(hdr['crln_obs'])
        crlt_obs_rad = np.deg2rad(hdr['crlt_obs'])
        crota2_rad = np.deg2rad(hdr['crota2'])
        dsun_obs = hdr['dsun_obs']

        dum = xt * np.cos(crln_obs_rad) + yt * np.sin(crln_obs_rad)
        yt = -xt * np.sin(crln_obs_rad) + yt * np.cos(crln_obs_rad)
        xt = dum

        dum = xt * np.cos(crlt_obs_rad) + zt * np.sin(crlt_obs_rad)
        zt = -xt * np.sin(crlt_obs_rad) + zt * np.cos(crlt_obs_rad)
        xt = dum

        dum = yt * np.cos(crota2_rad) - zt * np.sin(crota2_rad)
        zt = zt * np.cos(crota2_rad) + yt * np.sin(crota2_rad)
        yt = dum

        alpha = np.rad2deg(np.arctan2(yt, dsun_obs / 1e3 - xt)) * 3600
        beta = np.rad2deg(np.arctan2(zt, dsun_obs / 1e3 - xt)) * 3600

        img_x = np.round(alpha / hdr['cdelt1'] + hdr['crpix1'])
        img_y = np.round(beta / hdr['cdelt2'] + hdr['crpix2'])

        cube_sdo = np.zeros(np.shape(x))
        cube_sdo[corona] = mask[img_x, img_y]

        return cube_sdo
    else:
        return -1
    
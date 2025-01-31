; To create fake .save data from the HDF5 file with the fake data.


PRO convert_h5_to_save

  ; SETUP filepaths
  input_dir = '/home/avoyeux/old_project/avoyeux/python_codes/Data/fake_data/h5/'
  output_dir = '/home/avoyeux/old_project/avoyeux/python_codes/Data/fake_data/save/'
  file_list = FILE_SEARCH(input_dir + '*.h5')
  
  ; CHECK found files
  IF N_ELEMENTS(file_list) EQ 0 THEN BEGIN
    PRINT, 'No .h5 files found in ' + input_dir
    RETURN
  ENDIF

  FOR i=0, N_ELEMENTS(file_list)-1 DO BEGIN

    ; DATA setup
    filename_h5 = file_list[i]
    parts = FILE_BASENAME(filename_h5, '.h5')
    out_save = output_dir + parts + '.save'
    
    ; DATA open
    PRINT, 'Reading ', filename_h5
    file_id = H5F_OPEN(filename_h5)

    ; DATASETs read
    dset_id = H5D_OPEN(file_id, 'cube')
    cube    = H5D_READ(dset_id)
    H5D_CLOSE, dset_id

    dset_id = H5D_OPEN(file_id, 'dx')
    dx      = H5D_READ(dset_id)
    H5D_CLOSE, dset_id

    dset_id = H5D_OPEN(file_id, 'xt_min')
    xt_min  = H5D_READ(dset_id)
    H5D_CLOSE, dset_id

    dset_id = H5D_OPEN(file_id, 'yt_min')
    yt_min  = H5D_READ(dset_id)
    H5D_CLOSE, dset_id

    dset_id = H5D_OPEN(file_id, 'zt_min')
    zt_min  = H5D_READ(dset_id)
    H5D_CLOSE, dset_id

    dset_id = H5D_OPEN(file_id, 'xt_max')
    xt_max  = H5D_READ(dset_id)
    H5D_CLOSE, dset_id

    dset_id = H5D_OPEN(file_id, 'yt_max')
    yt_max  = H5D_READ(dset_id)
    H5D_CLOSE, dset_id

    dset_id = H5D_OPEN(file_id, 'zt_max')
    zt_max  = H5D_READ(dset_id)
    H5D_CLOSE, dset_id

    ; DATA close
    H5F_CLOSE, file_id
    
    ; DATA save
    SAVE, cube, dx, xt_min, yt_min, zt_min, xt_max, yt_max, zt_max, $
          FILENAME=out_save
    PRINT, 'Wrote: ', out_save
  ENDFOR
END

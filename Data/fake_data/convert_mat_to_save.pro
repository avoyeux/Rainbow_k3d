PRO convert_h5_to_save

  input_dir = '/home/avoyeux/python_codes/Data/fake_data/h5'
  output_dir = '/home/avoyeux/python_codes/Data/fake_data/save'
  ; Make sure the input_dir has trailing slash if needed
  file_list = FILE_SEARCH(input_dir + '*.h5')
  
  IF N_ELEMENTS(file_list) EQ 0 THEN BEGIN
    PRINT, 'No .h5 files found in ' + input_dir
    RETURN
  ENDIF

  ; Create output directory if needed
  FILE_MKDIR, output_dir, /ALLOW_EXISTING

  FOR i=0, N_ELEMENTS(file_list)-1 DO BEGIN
    filename_h5 = file_list[i]
    parts = FILE_BASENAME(filename_h5, '.h5')
    out_save = output_dir + parts + '.save'
    
    PRINT, 'Reading ', filename_h5
    h5_id = H5_OPEN(filename_h5)
    
    ; Read the datasets from HDF5
    cube   = H5_READ(h5_id, 'cube')
    dx     = H5_READ(h5_id, 'dx')
    xt_min = H5_READ(h5_id, 'xt_min')
    yt_min = H5_READ(h5_id, 'yt_min')
    zt_min = H5_READ(h5_id, 'zt_min')
    xt_max = H5_READ(h5_id, 'xt_max')
    yt_max = H5_READ(h5_id, 'yt_max')
    zt_max = H5_READ(h5_id, 'zt_max')

    ; Close the file
    H5_CLOSE, h5_id
    
    ; Now save them in a .save file
    SAVE, cube, dx, xt_min, yt_min, zt_min, xt_max, yt_max, zt_max, $
          FILENAME=out_save

    PRINT, 'Wrote: ', out_save
  ENDFOR
END
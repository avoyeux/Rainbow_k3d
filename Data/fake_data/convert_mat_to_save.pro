; To create the fake .save files from my .mat files

PRO ConvertMatToSave
    ; Define the input and output directories
    input_dir = '/home/avoyeux/old_project/avoyeux/Data/fake_data/mat'
    output_dir = '/home/avoyeux/old_project/avoyeux/Data/fake_data/save'

    ; Get the list of .mat files in the input directory
    file_list = FILE_SEARCH(input_dir + '*.mat')

    ; Loop over each .mat file and convert it to a .save file
    FOR i = 0, N_ELEMENTS(file_list) - 1 DO BEGIN
        ; Get the input file path
        input_mat_file = file_list[i]

        ; Extract the file name without extension
        file_name = FILE_BASENAME(input_mat_file, '.mat')

        ; Define the output file path
        output_save_file = output_dir + file_name + '.save'

        ; Read the .mat file
        data = READ_MAT(input_mat_file)

        ; Extract the variables
        cube = data.cube
        dx = data.dx
        xt_min = data.xt_min
        yt_min = data.yt_min
        zt_min = data.zt_min
        xt_max = data.xt_max
        yt_max = data.yt_max
        zt_max = data.zt_max

        ; Save the variables to a .save file
        SAVE, cube, dx, xt_min, yt_min, zt_min, xt_max, yt_max, zt_max, FILENAME=output_save_file
    ENDFOR
END
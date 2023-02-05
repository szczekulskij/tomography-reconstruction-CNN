import numpy as np

def read_txt(file_path):
    output_array = []
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        nr = line.replace("\n", "")
        nr = int(nr)
        output_array.append(nr)
    return np.array(output_array)

def read_matrix(file_name):
    # return np.fromfile(file_name, dtype='double')
    return np.fromfile(file_name, dtype='double')
    # return np.fromfile(file_name, dtype='float')
    
def get_data(read_from = "txt"):
    ''' 
    Reads the .mcr file and output of .m file to get nanowire's data(remember, single nanowire!)
    read_from: (str)
        either "txt" or "binary"
    '''
    if read_from not in ["txt", "binary"]: raise Exception

    DIRECTORY = "/Users/szczekulskij/side_projects/tomography-reconstruction-CNN/data/created_data/"
    NR_SLICES = 512
    NR_PROJECTIONS = 140
    output_sinograms = []

    for i in range(1,NR_SLICES + 1):
        folder_w_projections = DIRECTORY + "Joost_Slice" + str(i)
        sinogram = []
        for j in range(1, NR_PROJECTIONS + 1):
            if read_from == "binary":
                projection_file = folder_w_projections + "/" + "projection_" + str(j)
                projection = read_matrix(projection_file)
            elif read_from == "txt":
                projection_file = folder_w_projections + "/" + "projection_" + str(j) + ".txt"
                projection = read_txt(projection_file)
            sinogram.append(projection)
        sinogram = np.array(sinogram).T
        output_sinograms.append(sinogram)
    return output_sinograms
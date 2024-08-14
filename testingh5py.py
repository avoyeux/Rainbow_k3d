import h5py
import numpy as np

from datetime import datetime

h5File = h5py.File('example.h5', 'w')

h5File.attrs['author'] = 'Voyeux Alfred'
h5File.attrs['description'] = 'This is the main file descrition.'
h5File.attrs['creationDate'] = datetime.now().isoformat()
h5File.attrs['filename'] = 'filename.h5'




import h5py
import numpy as np

# Data for the datasets
data_root = np.random.rand(10)  # Random data for the root dataset
data_group = np.random.rand(5)  # Random data for the dataset in the group

# Create a new HDF5 file
with h5py.File('example.h5', 'w') as file:
    # Add attributes to the root of the file
    file.attrs['description'] = 'This is the main file description.'
    file.attrs['author'] = 'Author Name'
    
    # Create a dataset at the root level
    dset_root = file.create_dataset('root_dataset', data=data_root)
    dset_root.attrs['description'] = 'Dataset at the root level.'
    
    # Create a group
    group = file.create_group('group1')
    group.attrs['description'] = 'This is a group containing a dataset.'
    
    # Create a dataset within the group
    dset_group = group.create_dataset('dataset_in_group', data=data_group)
    dset_group.attrs['description'] = 'Dataset within a group.'
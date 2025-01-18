"""
To store the parent (base) class storing methods to be able to create an HDF5 file with a default
formatting.
"""

# IMPORTs
import os
import h5py

# IMPORTs alias
import numpy as np



class BaseHdf5Creator:
    """
    To store base methods for creating HDF5 files.
    """

    def __init__(self) -> None: pass

    def add_group(
            self,
            parent_group: h5py.File | h5py.Group,
            info: dict[str, str | int | float | np.generic | np.ndarray | dict],
            name: str,
        ) -> None:
        """
        Adds group(s) with dataset(s) and attribute(s) to a HDF5 group.
        Group created if dict[str, str | dict]
        Dataset created if not dict[str, dict] and at least one item not being a str.
        If an item is a dict, then .add_group() method is called on the item.
        Any str item becomes an attribute.

        Args:
            parent_group (h5py.File | h5py.Group): the HDF5 file or group where to add the new
                group.
            info (dict[str, str | int | float | np.generic | np.ndarray | dict]):
                the information and data to add in the group.
            name (str): the name of the group to add.
        """
            
        # CHECK name
        if name == '': return 

        if any(isinstance(value, dict) for value in info.values()):
            # GROUP create
            group = parent_group.require_group(name)

            for key, value in info.items():
                if isinstance(value, dict):
                    # GROUP add
                    self.add_group(group, value, key)
                else: 
                    # ATTRIBUTES add
                    group.attrs[key] = value

        elif all(isinstance(value, str) for value in info.values()):
            # GROUP with attributes only
            group = parent_group.require_group(name)
            for key, value in info.items(): group.attrs[key] = value

        else:
            # DATASET as values but no dict
            self.add_dataset(parent_group, info, name) #type: ignore

    def add_dataset(
            self,
            parent_group: h5py.File | h5py.Group,
            info: dict[str, str | int | float | np.generic | np.ndarray],
            name: str = '',
        ) -> None:
        """
        Adds a dataset and attributes to a HDF5 group like object.

        Args:
            parent_group (h5py.File | h5py.Group): the HDF5 file or group where to add the dataset.
            info (dict[str, str | int | float | np.generic | np.ndarray]): the information to add
                in the DataSet.
            name (str, optional): the name of the DataSet to add. Defaults to '' (then the
                attributes are directly added to the group).
        """
        
        # CHECK empty
        if len(info) == 0: return
        key: str = ''  # no need but for the type checker

        # DATASET key for ndarray
        stopped = False
        for key, item in info.items(): 
            if not isinstance(item, str): stopped = True; break
        if not stopped: key = ''  # no dataset. Add attributes to group.

        # DATASET create
        if (name != '') and (key != ''):
            dataset = parent_group.create_dataset(name, data=info[key])
        else:
            dataset = parent_group

        # ATTRIBUTEs add
        for k, value in info.items():
            if k == key: continue
            dataset.attrs[key] = value

"""
To test if all_data is really the intersection of the line of sight data.
"""

# IMPORTs
import os
import h5py
import yaml
import sparse
import unittest

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
from numba import njit
from typing import Any
from dataclasses import dataclass

# IMPORTs personal
from common import root_path, Decorators, DictToObject

# CONFIGURATION load
with open(os.path.join(root_path, 'config.yml'), 'r') as conf:
    config = DictToObject(yaml.safe_load(conf))

# PLACEHOLDERs type annotation
LockProxy = Any
ValueProxy = Any
QueueProxy = Any



@dataclass(slots=True, frozen=True, repr=False, eq=False)
class DataCube:
    """
    To store and get the coordinates of the cubes.
    """

    coords: h5py.Dataset

    def __getitem__(self, index: int) -> np.ndarray:

        # FILTER
        cube_filter = self.coords[0] == index
        cube_coords = self.coords[1:, cube_filter]
        return cube_coords


class CompareCubes:
    """
    To compare the intersection of the line of sight data with the all data.
    """

    def __init__(
            self,
            filepath: str,
            processes: int = 4,
            verbose: int = 1,
            flush: bool = False,
        ) -> None:
        """
        To initialize the CompareCubes class.

        Args:
            filepath (str): the filepath of the HDF5 to be tested.
            processes (int, optional): the number of processes to be used. Defaults to 4.
            verbose (int, optional): the verbosity level. Defaults to 1.
            flush (bool, optional): to flush the print. Defaults to False.
        """

        self.filepath = filepath
        self.processes = processes
        self.verbose = verbose
        self.flush = flush

    @Decorators.running_time
    def run_checks(self) -> list[bool]:
        """
        To run the checks on the intersection of the line of sight data.

        Returns:
            list[bool]: the results of the checks.
        """

        with h5py.File(self.filepath, 'r') as HDF5File:

            los_sdo_path = 'Filtered/SDO line of sight'
            sdo_data = self.get_data(HDF5File, los_sdo_path)

            los_stereo_path = 'Filtered/STEREO line of sight'
            stereo_data = self.get_data(HDF5File, los_stereo_path)

            all_data_path = 'Filtered/All data'
            all_data = self.get_data(HDF5File, all_data_path)

            # TEST
            return self.multiprocessing_test(sdo_data, stereo_data, all_data)

    def multiprocessing_test(
            self,
            sdo_data: DataCube,
            stereo_data: DataCube,
            all_data: DataCube,
        ) -> list[bool]:
        """
        To run the checks on the intersection of the line of sight data using multiprocessing.

        Args:
            sdo_data (DataCube): the SDO line of sight data.
            stereo_data (DataCube): the STEREO line of sight data.
            all_data (DataCube): the all data.

        Returns:
            list[bool]: the results of the checks.
        """

        # SETUP
        max_index = int(sdo_data.coords[0].max())
        process_nb = min(self.processes, max_index)
        manager = mp.Manager()
        lock = manager.Lock()
        value = manager.Value('i', max_index + 1)
        output_queue = manager.Queue()

        # RUN
        processes: [mp.Process] = [None] * process_nb
        for i in range(process_nb):
            p = mp.Process(
                target=self.check_intersection,
                args=(lock, value, sdo_data, stereo_data, all_data, output_queue),
            )
            p.start()
            processes[i] = p
        for p in processes: p.join()

        # RESULTs
        results: list[bool] = [None] * (max_index + 1)
        while not output_queue.empty():
            identifier, result = output_queue.get()
            results[identifier] = result
        return results

    def get_data(self, HDF5File: h5py.File, group_path: str) -> DataCube:
        """
        To get the data from the HDF5 file.

        Args:
            HDF5File (h5py.File): the HDF5 file.
            group_path (str): the path of the group in the HDF5 file to fetch.

        Returns:
            DataCube: the corresponding data cubes.
        """

        try:
            # DATA get
            data_coords: h5py.Dataset = HDF5File[group_path + '/coords']
        except KeyError:
            print(f"Group {group_path} not found in the HDF5 file.")

        # DATA formatting
        return DataCube(data_coords)

    def check_intersection(
            self,
            lock: LockProxy,
            value: ValueProxy,
            sdo_data: DataCube,
            stereo_data: DataCube,
            all_data: DataCube,
            output_queue: QueueProxy,
        ) -> None:
        """
        To check the intersection of the line of sight data.

        Args:
            lock (LockProxy): a multiprocessing lock.
            value (ValueProxy): a multiprocessing value.
            sdo_data (DataCube): the SDO line of sight data.
            stereo_data (DataCube): the STEREO line of sight data.
            all_data (DataCube): the all data.
            output_queue (QueueProxy): a multiprocessing queue to save the results.
        """

        while True:
            with lock:
                index = int(value.value) - 1
                if index < 0: break
                value.value -= 1

            # INTERSECTION
            intersection = self.intersect_2d_arrays(sdo_data[index], stereo_data[index])

            # CHECK empty
            all_data_cube = all_data[index]
            if intersection.size == 0 and all_data_cube.size == 0:
                print(f"Cube {index:03d} is empty but test passed.", flush=self.flush)
                output_queue.put((index, True))
                continue
            elif (
                intersection.size == 0 and all_data_cube.size != 0
                ) or (
                intersection.size != 0 and all_data_cube.size == 0
                ):
                print(f"Cube {index:03d} intersection is empty and test failed.", flush=self.flush)
                output_queue.put((index, False))
                continue

            # CHECK intersection
            test_result = np.array_equal(
                np.sort(intersection, axis=1),
                np.sort(all_data_cube, axis=1),
            )
            output_queue.put((index, test_result))

            if self.verbose > 0:
                if test_result:
                    print(f"Cube {index:03d} intersection test passed.", flush=self.flush)
                else:
                    print(f"Cube {index:03d} intersection test failed.", flush=self.flush)

    def intersect_2d_arrays(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        """
        To intersect two 2D arrays by doing a bitwise operation on the hashed coordinates.

        Args:
            array1 (np.ndarray): the first array.
            array2 (np.ndarray): the second array.

        Returns:
            np.ndarray: the intersected array.
        """

        set1 = set(map(tuple, array1.T))
        set2 = set(map(tuple, array2.T))
        intersection = set1 & set2
        return np.array(list(intersection)).T
    
    @staticmethod
    @njit(parallel=True)
    def old2_intersect_2d_arrays(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        """
        Intersects two 2D arrays by comparing columns.

        Args:
            array1 (np.ndarray): The first array.
            array2 (np.ndarray): The second array.

        Returns:
            np.ndarray: A 2D array of intersected columns.
        """
        
        # Pre-allocate a boolean array to track matches
        matches = np.zeros(array1.shape[1], dtype=np.bool_)
        print(f'len(matches): {len(matches)}')
        for c_1 in range(array1.shape[1]):
            col1 = array1[:, c_1]
                
            for c_2 in range(array2.shape[1]):
                col2 = array2[:, c_2]

                if np.array_equal(col1, col2):
                    matches[c_1] = True
                    break
        
        # Extract matched columns
        intersected_columns = array1[:, matches]
        return intersected_columns

    def old_intersect_2d_arrays(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        """
        Intersects two 2D arrays by comparing columns.

        Args:
            array1 (np.ndarray): The first array.
            array2 (np.ndarray): The second array.

        Returns:
            np.ndarray: A 2D array of intersected columns.
        """

        max1 = np.max(array1, axis=1)
        max2 = np.max(array2, axis=1)
        shape = (
            max(max1[0] + 1, max2[0] + 1),
            max(max1[1] + 1, max2[1] + 1),
            max(max1[2] + 1, max2[2] + 1),
        )
        array1_dense = sparse.COO(
            coords=array1.astype('uint16'),
            shape=shape,
            data=1,
        ).todense().astype(bool)
        array2_dense = sparse.COO(
            coords=array2.astype('uint16'),
            shape=shape,
            data=1,
        ).todense().astype(bool)

        intersection = (array1_dense & array2_dense).astype('uint8')
        return sparse.COO(intersection).coords


class TestCompareCubes(unittest.TestCase):
    """
    To test the CompareCubes class.
    """

    def setUp(self) -> None:
        """
        To set up the test.
        """

        self.compare_cubes = CompareCubes(
            filepath=os.path.join(root_path, *config.paths.data.real.split('/')),
            processes=config.debug.processes,
            verbose=config.debug.verbose,
            flush=config.debug.flush,
        )

    def test_check_intersection(self) -> None:
        """
        To test the check_intersection method.
        """

        results = self.compare_cubes.run_checks()
        self.assertTrue(all(results), "Not all intersection are correct")



if __name__ == '__main__':
    unittest.main()

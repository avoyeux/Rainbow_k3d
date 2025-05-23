"""
To test if all_data is really the intersection of the line of sight data.
"""

# IMPORTs
import h5py
import unittest

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
from typing import Any
from dataclasses import dataclass

# IMPORTs personal
from common import config, Decorators

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
            filepath: str | None = None,
            processes: int | None = None,
            verbose: int | None = None,
            flush: bool | None = None,
        ) -> None:
        """
        To initialize the CompareCubes class.

        Args:
            filepath (str | None, optional): the filepath of the HDF5 to be tested. When None, it
                uses the config file value. Defaults to None.
            processes (int | None, optional): the number of processes to be used. When None, it
                uses the config file value. Defaults to None.
            verbose (int | None, optional): the verbosity level. When None, it uses the config file
                value. Defaults to None.
            flush (bool | None, optional): to flush the print. When None, it uses the config file
                value. Defaults to None.
        """

        # CONFIGURATION values
        self.filepath = config.path.data.real if filepath is None else filepath  #type:ignore
        if processes is None:
            self.processes: int = config.run.debug.processes  #type:ignore
        else:
            self.processes = processes if processes > 1 else 1
        self.verbose = config.run.debug.verbose if verbose is None else verbose  #type:ignore
        self.flush = config.run.debug.flush if flush is None else flush  #type:ignore

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
        processes: [mp.Process] = [None] * process_nb  #type:ignore
        for i in range(process_nb):
            p = mp.Process(
                target=self.check_intersection,
                args=(lock, value, sdo_data, stereo_data, all_data, output_queue),
            )
            p.start()
            processes[i] = p
        for p in processes: p.join()

        # RESULTs
        results: list[bool] = [None] * (max_index + 1)  #type:ignore
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
            data_coords: h5py.Dataset = HDF5File[group_path + '/coords']  #type:ignore
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
        print(f'array1 and array2 shapes are {array1.shape}, {array2.shape}')
        set1 = set(map(tuple, array1.T))
        set2 = set(map(tuple, array2.T))
        intersection = set1 & set2
        return np.array(list(intersection)).T
    

class TestCompareCubes(unittest.TestCase):
    """
    To test the CompareCubes class.
    """

    def setUp(self) -> None:
        """
        To set up the test.
        """

        self.compare_cubes = CompareCubes()

    def test_check_intersection(self) -> None:
        """
        To test the check_intersection method.
        """

        results = self.compare_cubes.run_checks()
        self.assertTrue(all(results), "Not all intersection are correct")



if __name__ == '__main__': unittest.main()

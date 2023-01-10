import array
import os
import numpy as np
import tqdm as tqdm

from dataset.laserscan import SemLaserScan, LaserScan
from enum import Enum


class DataSource(Enum):
    KITTI = 1
    Custom = 2


class ScanIO:
    def __init__(self):
        pass

    @staticmethod
    def get_sequence_path_velodyne(root: str, sequence: int):
        seq = '{0:02d}'.format(int(sequence))
        scans_path = os.path.join(root, "sequences", seq, "velodyne")
        return scans_path

    @staticmethod
    def get_sequence_path_labels(root: str, sequence: int):
        seq = '{0:02d}'.format(int(sequence))
        labels_path = os.path.join(root, "sequences", seq, "labels")
        return labels_path

    @staticmethod
    def load_scan(filename: str, source: DataSource = DataSource.KITTI, labels_filename: str = None,
                  color_dict: dict = None):
        if labels_filename is None or color_dict is None:
            scan = LaserScan(project=True)
        else:
            scan = SemLaserScan(color_dict, project=True)

        if source is DataSource.KITTI:
            scan.open_scan(filename)
        elif source is DataSource.Custom:
            pcd = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            scan.set_points(pcd[:, :3], pcd[:, 3])

        if labels_filename is not None:
            scan.open_label(labels_filename)
            scan.do_label_projection()
            scan.colorize()

        return scan

    @staticmethod
    def get_scans_filenames(folder: str, extension: str = '.bin', recursive: bool = False):
        scan_names = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(extension):
                    scan_names.append(os.path.join(root, file))

            if not recursive:
                break
        scan_names.sort()
        return scan_names

    def load_scans(self, folder: str, labels_folder: str = None, source: DataSource = DataSource.KITTI):
        # Walk on folder
        # for(load_scan)...
        pass

    def save_scan(self, filename: str, scan: SemLaserScan | LaserScan, labels_filename: str = None,
                  source: DataSource = DataSource.KITTI):
        pass

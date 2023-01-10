import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np

from dataset.laserscan import SemLaserScan, LaserScan
from dataset.ioscan import ScanIO
from dataset.transforms import Transforms, DataAnalyzer

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class CustomDataset(Dataset):
    def __init__(self, root,  # directory where scans are
                 labels_dir,  # directory where labels are
                 labels,  # label dict: (e.g 10: "car")
                 color_map,  # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,  # inverse of previous (recover labels)
                 sensor,  # sensor to parse scans from
                 max_points=150000,  # max number of points present in dataset
                 gt=True,
                 transform=False,
                 load_single_scan: bool = False,
                 single_scan_index=0,
                 darkening_factor=1.0,
                 compression_threshold=50.0,
                 compression_ratio=0.1
                 ):
        self.compression_ratio = compression_ratio
        self.compression_threshold = compression_threshold
        self.darkening_factor = darkening_factor
        self.labels_dir = labels_dir
        self.root = root
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.transform = transform
        self.load_single_scan = load_single_scan
        self.single_scan_index = single_scan_index

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.n_classes = len(self.learning_map_inv)

        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure labels is a dict
        assert (isinstance(self.labels, dict))

        # make sure color_map is a dict
        assert (isinstance(self.color_map, dict))

        # make sure learning_map is a dict
        assert (isinstance(self.learning_map, dict))

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []

        scan_names = ScanIO.get_scans_filenames(self.root, EXTENSIONS_SCAN[0])
        self.scan_files.extend(scan_names)
        self.scan_files.sort()

        if self.labels_dir is not None:
            label_names = ScanIO.get_scans_filenames(self.root, EXTENSIONS_LABEL[0])
            self.label_files.extend(label_names)
            # sort for correspondence
            self.label_files.sort()

            assert (len(self.scan_files) == len(self.label_files))

        if len(self.scan_files) == 0:
            raise f"Scans in directory {self.root} with extension {EXTENSIONS_SCAN[0]} were not found!"

        if load_single_scan:
            if self.single_scan_index >= len(self.scan_files):
                raise f"Selected index {self.single_scan_index} is out of bounds for selected folder (contains {len(self.scan_files)} scans)"
            single_scan_filename = self.scan_files[self.single_scan_index]
            self.scan_files.clear()
            self.scan_files.append(single_scan_filename)

            if self.gt:
                single_label_filename = self.label_files[self.single_scan_index]
                self.label_files.clear()
                self.label_files.append(single_label_filename)

        print(f"Using {len(self.scan_files)} scans from folder {self.root}. Labels loaded")

    def __getitem__(self, index):
        # get item in tensor shape
        scan_file = self.scan_files[index]
        if self.gt:
            label_file = self.label_files[index]

        # open a semantic laserscan
        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        if self.transform:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True
                drop_points = random.uniform(0, 0.5)

        if self.gt:
            scan = SemLaserScan(self.color_map,
                                project=True,
                                H=self.sensor_img_H,
                                W=self.sensor_img_W,
                                fov_up=self.sensor_fov_up,
                                fov_down=self.sensor_fov_down,
                                DA=DA,
                                flip_sign=flip_sign,
                                drop_points=drop_points)
        else:
            scan = LaserScan(project=True,
                             H=self.sensor_img_H,
                             W=self.sensor_img_W,
                             fov_up=self.sensor_fov_up,
                             fov_down=self.sensor_fov_down,
                             DA=DA,
                             rot=rot,
                             flip_sign=flip_sign,
                             drop_points=drop_points)

        # open and obtain scan
        scan.open_scan(scan_file,
                       normalize_remissions=True,
                       rotate_scan=True,
                       remissions_factor=self.darkening_factor,
                       compression_threshold=self.compression_threshold,
                       compression_ratio=self.compression_ratio)

        if self.gt:
            scan.open_label(label_file)
            # map unused classes to used classes (also for projection)
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []

        if self.gt:
            scan.colorize()
            proj_sem_color = torch.from_numpy(scan.proj_sem_color).clone()
        else:
            proj_sem_color = []

        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        # return
        return proj, proj_mask, proj_labels, unproj_labels, proj_sem_color, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]


class Parser:
    def __init__(self, root: str, labels_dir: str | None, labels: dict, color_map, learning_map, learning_map_inv,
                 sensor, max_points,
                 batch_size, workers, darkening_factor, compression_threshold, compression_ratio):
        self.labels_dir = labels_dir
        self.loader = None
        self.dataset = None
        self.workers = workers
        self.batch_size = batch_size
        self.max_points = max_points
        self.sensor = sensor
        self.learning_map_inv = learning_map_inv
        self.learning_map = learning_map
        self.color_map = color_map
        self.labels = labels
        self.root = root
        self.darkening_factor = darkening_factor
        self.compression_threshold = compression_threshold
        self.compression_ratio = compression_ratio

    def get_dataset(self,
                    use_semantics=True,
                    load_single_scan: bool = False,
                    single_scan_index=0):
        self.dataset = CustomDataset(root=self.root,
                                     labels_dir=self.labels_dir,
                                     labels=self.labels,
                                     color_map=self.color_map,
                                     learning_map=self.learning_map,
                                     learning_map_inv=self.learning_map_inv,
                                     sensor=self.sensor,
                                     max_points=self.max_points,
                                     gt=use_semantics,
                                     load_single_scan=load_single_scan,
                                     single_scan_index=single_scan_index,
                                     darkening_factor=self.darkening_factor,
                                     compression_threshold=self.compression_threshold,
                                     compression_ratio=self.compression_ratio)
        return self.dataset

    def get_loader(self,
                   use_semantics=True,
                   load_single_scan: bool = False,
                   single_scan_index=0
                   ):
        if self.dataset is None:
            self.get_dataset(use_semantics=use_semantics,
                             load_single_scan=load_single_scan,
                             single_scan_index=single_scan_index)

        self.loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.workers,
                                                  drop_last=True)

        return self.loader

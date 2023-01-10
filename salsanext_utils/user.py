import os

import torch
from scipy.stats import norm
from visualization import img_vis

from dataset.laserscan import SemLaserScan
from dataset.transforms import DataAnalyzer
from salsanext_utils.inferrer import Inferrer
from dataset.ioscan import ScanIO as IO
from dataset.kitti_dataset import Parser as KittiParser
from dataset.custom_dataset import Parser as CustomParser

import numpy as np

from visualization import img_vis
from visualization.ouster_viz import OusterViz


class User:
    def __init__(self, flags, arch, data):
        self.max_scans = 0
        self.accum_scans = None
        self.loader = None
        self.flags = flags
        self.arch = arch
        self.data = data

        if self.flags.data_source == 'kitti':
            sequence = [0]
            self.scans_paths = IO.get_scans_filenames(IO.get_sequence_path_velodyne(self.flags.dataset_lf, sequence[0]))
            self.random_index = int(np.random.randint(low=0, high=int(len(self.scans_paths)), size=1))
            self.parser = KittiParser(root=self.flags.dataset_lf,
                                      sequences=sequence,
                                      labels=self.data['labels'],
                                      color_map=self.data['color_map'],
                                      learning_map=self.data['learning_map'],
                                      learning_map_inv=self.data['learning_map_inv'],
                                      sensor=self.arch["dataset"]["sensor"],
                                      max_points=self.arch["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.arch["train"]["workers"])
        elif self.flags.data_source == 'custom':
            self.scans_paths = IO.get_scans_filenames(self.flags.dataset_lf)
            self.random_index = int(np.random.randint(low=0, high=int(len(self.scans_paths)), size=1))
            self.parser = CustomParser(root=self.flags.dataset_lf,
                                       labels_dir=None,
                                       labels=self.data['labels'],
                                       color_map=self.data['color_map'],
                                       learning_map=self.data['learning_map'],
                                       learning_map_inv=self.data['learning_map_inv'],
                                       sensor=self.arch["dataset"]["sensor"],
                                       max_points=self.arch["dataset"]["max_points"],
                                       batch_size=1,
                                       workers=self.arch["train"]["workers"],
                                       darkening_factor=self.flags.darkening_factor,
                                       compression_threshold=self.flags.compression_threshold,
                                       compression_ratio=self.flags.compression_ratio)
        else:
            print(f"Selected self.data source '{self.flags.data_source}' is not supported. Select 'kitti' or 'custom'")
            quit()

        self.inferrer = Inferrer(self.flags, self.arch, self.data)

    def infer_scan(self, index: int | None = None):
        if index is None:
            index = self.random_index

        if index >= len(self.scans_paths):
            raise Exception("Index provided ({}) is greater than number of scans in dataset ({})".format(index,
                                                                                                         len(self.scans_paths)))

        self.loader = self.parser.get_loader(use_semantics=self.flags.use_semantics,
                                             load_single_scan=True,
                                             single_scan_index=index)

        self.accum_scans = []
        self.__infer_dataset()
        if self.flags.use_visualizer:
            self.visualize_accum()

    def infer_dataset(self, max_scans=0):
        if self.flags.data_source == "custom":
            self.loader = self.parser.get_loader(use_semantics=self.flags.use_semantics,
                                                 load_single_scan=False,
                                                 single_scan_index=0)
        elif self.flags.data_source == "kitti":
            self.loader = self.parser.get_loader()

        self.max_scans = max_scans
        self.accum_scans = []
        self.__infer_dataset()
        if self.flags.use_visualizer:
            self.visualize_accum()

    def __infer_dataset(self):
        # proj, proj_mask, proj_labels, unproj_labels, proj_sem_color, path_seq, path_name, proj_x, proj_y,
        # proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points
        for i, (
                proj_in, proj_mask, proj_labels, _, proj_sem_color, path_seq, path_name, p_x, p_y, proj_range,
                unproj_range,
                _, unproj_xyz,
                proj_remissions, unproj_remissions,
                n_points) in enumerate(self.loader):
            scan_filename = path_name[0].replace("_lf", "").replace(".label", ".bin")
            print(f"Processing scan LF {scan_filename}...")

            # -d "/Volumes/TOSHIBA EXT/log_2911/lidar_frame/scan_0000000000_lf-scan_0000000249_lf" -w
            # "/Volumes/TOSHIBA EXT/log_2911/world_frame/scan_0000000000-scan_0000000249" -o "/Volumes/My
            # Passport/results_2911/scan_0000000000-scan_0000000249/predictions" -r "/Volumes/My
            # Passport/results_2911/scan_0000000000-scan_0000000249/ranges" -m ./pretrained -s custom

            # Infer scan
            predictions, scan = self.inferrer.infer(proj_in, p_x, p_y, proj_range, unproj_range, unproj_xyz,
                                                    unproj_remissions,
                                                    n_points)

            # Trim predictions to match scan size
            indices = np.where((unproj_xyz[0].numpy() == -1.0).sum(axis=1) == 3)[0]
            predictions_trimmed = np.delete(predictions, indices, axis=0)

            # Trim also ranges
            ranges_trimmed = np.copy(scan.unproj_range)
            ranges_trimmed = np.delete(ranges_trimmed, indices, axis=0)

            # Save predictions
            if self.flags.output_folder is not None:
                output_file = os.path.join(self.flags.output_folder, path_name[0].replace("_lf", ""))
                print(f"Saving predictions to file {output_file}")
                predictions_trimmed.tofile(output_file)

            if self.flags.save_ranges is not None:
                save_ranges = os.path.join(self.flags.save_ranges,
                                           path_name[0].replace("_lf", "").replace("label", "range"))
                print(f"Saving predictions to file {save_ranges}")
                ranges_trimmed.tofile(save_ranges)

            # Visualize inferred scan
            if self.flags.use_visualizer:

                # Once predicted, assign labels to WF scan
                scan_wf_filename = path_name[0].replace("_lf", "").replace(".label", ".bin")
                scan_wf_path = os.path.join(self.flags.dataset_wf, scan_wf_filename)
                # Load WF scan
                scan_wf = SemLaserScan(self.data['color_map'], project=True)
                scan_wf.open_scan(scan_wf_path,
                                  normalize_remissions=False,
                                  rotate_scan=False,
                                  remissions_factor=1.0)
                scan_wf.set_label(predictions_trimmed)
                scan_wf.do_range_projection()
                scan_wf.do_label_projection()
                scan_wf.colorize()

                if self.accum_scans is not None:
                    self.accum_scans.append(scan_wf)

                    if self.max_scans == 0:
                        img_vis.visualize_proj(scan.proj_remission, title="Remission")
                        if len(scan.proj_sem_color):
                            img_vis.visualize_semantics(scan.proj_sem_color, title="Semantics")

                    if not self.flags.filter_labels:
                        filter_labels = None
                    else:
                        filter_labels = self.flags.filter_labels

            if self.max_scans != 0:
                if i >= self.max_scans - 1:
                    return

    def visualize_accum(self):
        print("Visualize accumulated point cloud")
        scan_accum = SemLaserScan(self.data["color_map"], project=True)
        points_accum = np.empty((0, 3))
        remissions_accum = np.empty(0)
        sem_labels_accum = np.empty(0, dtype=np.uint8)
        for i, scan_wf in enumerate(self.accum_scans):
            points_accum = np.concatenate((points_accum, scan_wf.points), axis=0)
            remissions_accum = np.concatenate((remissions_accum, scan_wf.remissions))
            sem_labels_accum = np.concatenate((sem_labels_accum, scan_wf.sem_label))

        scan_accum.set_points(points_accum, remissions_accum)
        scan_accum.set_label(sem_labels_accum)
        scan_accum.do_range_projection()
        scan_accum.do_label_projection()
        scan_accum.colorize()

        viz = OusterViz()
        viz.add_scan(scan_accum, use_labels=True, center=True)
        viz.update()
        viz.run()

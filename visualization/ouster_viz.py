import copy
import numpy as np

from ouster.sdk import viz


class OusterViz:
    def __init__(self, name="Scan viewer"):
        self.point_viz = viz.PointViz(name)
        viz.add_default_controls(self.point_viz)

    def update(self):
        self.point_viz.update()
        self.point_viz.camera.reset()

    def add_scan(self, scan, use_labels=False, center=False, filter_labels: list | None = None):
        # Deep copy scan for safety
        l_scan = copy.deepcopy(scan)

        if center:
            # Bring l_scan to (0,0,0), so it can be visualized
            l_scan.points[:, 0] -= np.average(l_scan.points[:, 0])
            l_scan.points[:, 1] -= np.average(l_scan.points[:, 1])
            l_scan.points[:, 2] -= np.average(l_scan.points[:, 2])

        # Filter by label
        if filter_labels is not None:
            filter_labels = list(map(int, filter_labels))  # map filter_labels list to type int
            cond = np.where(np.isin(l_scan.sem_label, filter_labels))
            l_scan.points = l_scan.points[cond]
            l_scan.remissions = l_scan.remissions[cond]
            l_scan.sem_label_color = l_scan.sem_label_color[cond]

        num_points = l_scan.points.shape[0]
        point_set = np.concatenate((l_scan.points[:, 0], l_scan.points[:, 1], l_scan.points[:, 2]), axis=None)

        cloud_xyz = viz.Cloud(num_points)
        cloud_xyz.set_xyz(point_set)

        if use_labels:
            sem_label_color = np.copy(l_scan.sem_label_color)
            color_set = np.vstack((sem_label_color.transpose(), np.ones(num_points)))
            cloud_xyz.set_mask(color_set.transpose())
        else:
            rem = np.copy(l_scan.remissions)
            cloud_xyz.set_key(rem / np.amax(rem))

        self.point_viz.add(cloud_xyz)

    def run(self):
        self.point_viz.update()
        self.point_viz.run()

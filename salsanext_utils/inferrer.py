import dataset.transforms
from dataset.laserscan import LaserScan, SemLaserScan
import torch
import torch.nn as nn
import numpy as np
import time

from dataset.transforms import DataAnalyzer
from salsanext.modules.SalsaNext import SalsaNext
from salsanext.modules.SalsaNextAdf import SalsaNextUncertainty
from salsanext.postproc.KNN import KNN


class Inferrer:
    def __init__(self, flags, arch: dict, data: dict):
        self.data = data
        self.reset(model_dir=flags.model,
                   sensor=arch["dataset"]["sensor"],
                   uncertainty=False,
                   montecarlo=flags.monte_carlo,
                   learning_map_inv=data["learning_map_inv"])

        if arch["post"]["KNN"]["use"]:
            self.load_post(arch["post"]["KNN"]["params"])

    def reset(self, model_dir: str, sensor: dict, uncertainty: bool, montecarlo: int, gpu: bool = False,
              learning_map_inv: dict = None):
        # Init model and post
        self.model = None
        self.post = None

        self.mc = montecarlo
        self.uncertainty = uncertainty
        self.gpu = gpu
        self.model_dir = model_dir

        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)

        self.learning_map_inv = learning_map_inv
        self.to_orig_fn = self.__to_original

        self.n_classes = len(self.learning_map_inv)
        self.load_model(self.model_dir)  # --> self.model (torch model)

    def load_model(self, model_dir: str):
        # concatenate the encoder and the head
        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.uncertainty:
                self.model = SalsaNextUncertainty(self.n_classes)
                self.model = nn.DataParallel(self.model)
                w_dict = torch.load(model_dir + "/SalsaNext",
                                    map_location=lambda storage, loc: storage)
                self.model.load_state_dict(w_dict['state_dict'], strict=True)
            else:
                self.model = SalsaNext(self.n_classes)
                self.model = nn.DataParallel(self.model)
                w_dict = torch.load(model_dir + "/SalsaNext",
                                    map_location=lambda storage, loc: storage)
                self.model.load_state_dict(w_dict['state_dict'], strict=True)

    def load_post(self, knn_params):
        # use knn post-processing?
        self.post = KNN(knn_params, self.n_classes)

    @staticmethod
    def __map(label, mapdict):
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

    def __to_original(self, label):
        # put label in original values
        return self.__map(label, self.learning_map_inv)

    def infer(self, proj_in, p_x, p_y, proj_range,
              unproj_range, unproj_xyz, unproj_remissions,
              n_points):
        # first cut to rela size (batch size one allows it)
        # assert(isinstance(scan, LaserScan), "scan must be of type LaserScan")

        cnn = []
        knn = []
        end = time.time()

        if not self.uncertainty:
            self.model.eval()
        total_time = 0
        total_frames = 0

        if self.gpu:
            proj_in = proj_in.cuda()
            p_x = p_x.cuda()
            p_y = p_y.cuda()
            if self.post:
                proj_range = proj_range.cuda()
                unproj_range = unproj_range.cuda()

        # compute output
        if self.uncertainty:
            proj_output_r, log_var_r = self.model(proj_in)
            for i in range(self.mc):
                log_var, proj_output = self.model(proj_in)
                log_var_r = torch.cat((log_var, log_var_r))
                proj_output_r = torch.cat((proj_output, proj_output_r))

            proj_output2, log_var2 = self.model(proj_in)
            proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1)
            log_var2 = log_var_r.mean(dim=0, keepdim=True).mean(dim=1)
            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - end
            total_time += frame_time
            total_frames += 1
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # log_var2 = log_var2[0][p_y, p_x]
            # log_var2 = log_var2.cpu().numpy()
            # log_var2 = log_var2.reshape((-1)).astype(np.float32)

            log_var2 = log_var2[0][p_y, p_x]
            log_var2 = log_var2.cpu().numpy()
            log_var2 = log_var2.reshape((-1)).astype(np.float32)
            # assert proj_output.reshape((-1)).shape == log_var2.reshape((-1)).shape == pred_np.reshape((-1)).shape

            # map to original label
            pred_np = self.to_orig_fn(pred_np)

            print(total_time / total_frames)
        else:
            proj_output = self.model(proj_in)
            proj_argmax = proj_output[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("CNN Inferred scan in", res, "sec")
            end = time.time()
            cnn.append(res)

            if self.post:
                # knn post-proc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original point-cloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("KNN Inferred scan in", res, "sec")
            knn.append(res)
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # map to original label
            pred_np = self.to_orig_fn(pred_np)

            scan = SemLaserScan(self.data['color_map'], project=True)
            scan.set_points(unproj_xyz[0].numpy(), unproj_remissions[0].numpy())
            scan.set_label(pred_np)
            scan.do_range_projection()
            scan.do_label_projection()
            scan.colorize()

            return pred_np, scan

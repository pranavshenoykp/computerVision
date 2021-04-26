#!/usr/bin/python3

import logging
import os
from pathlib import Path
import pdb

from typing import List, Mapping, Tuple

import cv2
import imageio
import numpy as np
import torch

from proj6_code.avg_meter import AverageMeter, SegmentationAverageMeter
from proj6_code.utils import get_logger, get_dataloader_id_to_classname_map, cv2_imread_rgb
from proj6_code.mask_utils import write_six_img_grid_w_embedded_names

# from mseg.utils.mask_utils import (
#     save_pred_vs_label_7tuple,save_pred_vs_label_4tuple,
#     
# )
# from mseg.utils.mask_utils_detectron2 import Visualizer
# from mseg_semantic.utils.confusion_matrix_renderer import ConfusionMatrixRenderer

"""
Given a set of inference results (inferred label maps saved as grayscale images),
compute the accuracy vs. ground truth label maps.

Expects inference results to be saved as {save_folder}/gray/*.png, exactly as our
test scripts spit out.
"""


logger = get_logger()


class AccuracyCalculator:
    def __init__(
        self,
        args,
        data_list: List[Tuple[str, str]],
        dataset_name: str,
        class_names: List[str],
        save_folder: str,
        eval_taxonomy: str,
        num_eval_classes: int,
        excluded_ids: int,
        render_confusion_matrix: bool = False,
    ) -> None:
        """
        Args:
            args,
            data_list
            dataset_name:
            class_names:
            save_folder:
            num_eval_classes:
            render_confusion_matrix:
        """
        assert isinstance(eval_taxonomy, str)
        self.args = args
        self.data_list = data_list
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.save_folder = save_folder
        self.eval_taxonomy = eval_taxonomy
        self.num_eval_classes = num_eval_classes
        self.excluded_ids = excluded_ids
        self.gray_folder = os.path.join(save_folder, "gray")
        self.render_confusion_matrix = render_confusion_matrix

        if self.render_confusion_matrix:
            self.cmr = ConfusionMatrixRenderer(self.save_folder, class_names, self.dataset_name)
        self.sam = SegmentationAverageMeter()

        # can handle the `universal` taxonomy scenario just fine, since we pass in the classes manually
        self.id_to_class_name_map = get_dataloader_id_to_classname_map(
            self.dataset_name, class_names, include_ignore_idx_cls=True
        )

        assert isinstance(args.vis_freq, int)
        assert isinstance(args.model_path, str)

    def compute_metrics(self, save_vis: bool = True) -> None:
        """
        Args:
        -   save_vis: whether to save visualize examplars
        """
        self.evaluate_predictions(save_vis)
        self.print_results()
        self.dump_acc_results_to_file()

    def evaluate_predictions(self, save_vis: bool = True) -> None:
        """Calculate accuracy.

        Args:
            save_vis: whether to save visualizations on predictions vs. ground truth
        """
        pred_folder = self.gray_folder
        for i, (image_path, target_path) in enumerate(self.data_list):

            image_name = Path(image_path).stem
            pred = cv2.imread(os.path.join(pred_folder, image_name + ".png"), cv2.IMREAD_GRAYSCALE)

            target_img = imageio.imread(target_path)
            target_img = target_img.astype(np.int64)

            self.sam.update_metrics_cpu(pred, target_img, self.num_eval_classes)

            if (i + 1) % self.args.vis_freq == 0:
                print_str = (
                    f'Evaluating {i + 1}/{len(self.data_list)} on image {image_name+".png"},'
                    + f" accuracy {self.sam.accuracy:.4f}."
                )
                logger.info(print_str)

            if save_vis and ((i + 1) % self.args.vis_freq == 0):
                save_prediction_visualization(
                    pred_folder, image_path, image_name, pred, target_img, self.id_to_class_name_map
                )

    def print_results(self) -> None:
        """
        Dump per-class IoUs and mIoU to stdout.
        """
        iou_class, accuracy_class, mIoU, mAcc, allAcc = self.sam.get_metrics()

        if self.render_confusion_matrix:
            self.cmr.render()
        logger.info(self.dataset_name + " " + self.args.model_path)
        logger.info("Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(mIoU, mAcc, allAcc))

        for i in range(self.num_eval_classes):
            excluded_u_class = (self.eval_taxonomy == "universal") and (i in self.excluded_ids)
            if self.eval_taxonomy != "universal" or not excluded_u_class:
                logger.info(
                    "Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.".format(
                        f"{i:02}", iou_class[i], accuracy_class[i], self.class_names[i]
                    )
                )

    def dump_acc_results_to_file(self) -> None:
        """
        Save per-class IoUs and mIoU to a .txt file.

        When evaluating a model trained within the universal taxonomy, on the val
        split of a training dataset, we must exclude certain classes -- ie those
        classes present in the universal/unified taxonomy, but absent in the
        training dataset taxonomy. Otherwise our evaluation will be unfair.

        TODO: make note about comparing rows vs. columns in the main paper
        """
        result_file = f"{self.save_folder}/results.txt"
        if self.eval_taxonomy == "universal":
            iou_class, accuracy_class, mIoU, mAcc, allAcc = self.sam.get_metrics(
                exclude=True, exclude_ids=self.excluded_ids
            )
        else:
            iou_class, accuracy_class, mIoU, mAcc, allAcc = self.sam.get_metrics()
        result = open(result_file, "w")
        result.write("Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.\n".format(mIoU, mAcc, allAcc))

        for i in range(self.num_eval_classes):
            excluded_u_class = (self.eval_taxonomy == "universal") and (i in self.excluded_ids)
            if (self.eval_taxonomy != "universal") or not excluded_u_class:
                result.write(
                    "Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.\n".format(
                        f"{i:02}", iou_class[i], accuracy_class[i], self.class_names[i]
                    )
                )
        result.close()


def save_prediction_visualization(
    pred_folder: str,
    image_path: str,
    image_name: str,
    pred: np.ndarray,
    target_img: np.ndarray,
    id_to_class_name_map: Mapping[int, str],
) -> None:
    """
    Args:
        pred_folder
        image_path
        pred
        target_img
        id_to_class_name_map
    """
    image_name = Path(image_name).stem
    mask_save_dir = pred_folder.replace("gray", "rgb_mask_predictions")
    grid_save_fpath = f"{mask_save_dir}/{image_name}.jpg"
    rgb_img = cv2_imread_rgb(image_path)
    # save_pred_vs_label_7tuple(rgb_img, pred, target_img, self.id_to_class_name_map, grid_save_fpath)
    write_six_img_grid_w_embedded_names(rgb_img, pred, target_img, id_to_class_name_map, grid_save_fpath)

    # overlaid_save_fpath = f'{mask_save_dir}_overlaid/{image_name}.jpg'
    # create_leading_fpath_dirs(overlaid_save_fpath)
    # frame_visualizer = Visualizer(rgb_img, metadata=None)
    # overlaid_img = frame_visualizer.overlay_instances(
    #     label_map=pred,
    #     id_to_class_name_map=id_to_class_name_map
    # )
    # imageio.imwrite(overlaid_save_fpath, overlaid_img)

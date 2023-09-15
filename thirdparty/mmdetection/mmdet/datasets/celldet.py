# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
import os
import scipy.io as sio
import scipy
from scipy.optimize import linear_sum_assignment

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CellDetDataset(CustomDataset):
    CLASSES = ("lymph", "tumor", "stromal",)

    PALETTE = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='points',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['points']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        paired_all = []
        unpaired_true_all = []
        unpaired_pred_all = []
        true_inst_type_all = []
        pred_inst_type_all = []

        for idx, result in enumerate(results):
            img_id = self.img_ids[idx]
            img_info = self.coco.load_imgs(img_id)
            img_name = img_info[0]['filename']
            img_path = os.path.join(self.data_root, "test/gt_mat", img_name[:-4] + ".mat")
            true_info = sio.loadmat(img_path)
            true_centroid = (true_info["inst_centroid"]).astype("float32")
            true_inst_type = (true_info["inst_type"]).astype("int32")

            if true_centroid.shape[0] != 0:
                true_inst_type = true_inst_type[:, 0]
            else:  # no instance at all
                true_centroid = np.array([[0, 0]])
                true_inst_type = np.array([0])

            # MCS classes
            true_inst_type[(true_inst_type == 2)] = 9
            true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 2
            true_inst_type[
                (true_inst_type == 1) | (true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 3
            true_inst_type[(true_inst_type == 9)] = 1

            # pred info
            pred_centroid = []
            pred_inst_type = []

            for i in range(len(self.cat_ids)):
                classes = result[i][np.where(result[i][:, 2] > 0.6)]
                classes = classes[:, :2]
                pred_centroid.extend(classes)
                pred_type = np.full((classes.shape[0], 1), i + 1)
                pred_inst_type.extend(pred_type)

            pred_centroid = np.asarray(pred_centroid).astype("float32")
            pred_inst_type = np.asarray(pred_inst_type).astype("int32")

            if pred_centroid.shape[0] != 0:
                pred_inst_type = pred_inst_type[:, 0]
            else:  # no instance at all
                pred_centroid = np.array([[0, 0]])
                pred_inst_type = np.array([0])

            paired, unpaired_true, unpaired_pred = self.pair_coordinates(
                true_centroid, pred_centroid, 6
            )

            true_idx_offset = (
                true_idx_offset + true_inst_type_all[-1].shape[0] if idx != 0 else 0
            )
            pred_idx_offset = (
                pred_idx_offset + pred_inst_type_all[-1].shape[0] if idx != 0 else 0
            )
            true_inst_type_all.append(true_inst_type)
            pred_inst_type_all.append(pred_inst_type)

            if paired.shape[0] != 0:  # ! sanity
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)

        paired_all = np.concatenate(paired_all, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
            type_samples = (paired_true == type_id) | (paired_pred == type_id)

            paired_true = paired_true[type_samples]
            paired_pred = paired_pred[type_samples]

            tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
            tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
            fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
            fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

            fp_d = (unpaired_pred == type_id).sum()
            fn_d = (unpaired_true == type_id).sum()

            f1_type = (2 * (tp_dt + tn_dt)) / (
                    2 * (tp_dt + tn_dt)
                    + w[0] * fp_dt
                    + w[1] * fn_dt
                    + w[2] * fp_d
                    + w[3] * fn_d
            )
            return f1_type

        w = [1, 1]
        tp_d = paired_pred_type.shape[0]
        fp_d = unpaired_pred_type.shape[0]
        fn_d = unpaired_true_type.shape[0]

        tp_tn_dt = (paired_pred_type == paired_true_type).sum()
        fp_fn_dt = (paired_pred_type != paired_true_type).sum()

        acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
        f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

        w = [2, 2, 1, 1]

        # if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

        results_list = [f1_d, acc_type]
        for type_uid in type_uid_list:
            f1_type = _f1_type(
                paired_true_type,
                paired_pred_type,
                unpaired_true_type,
                unpaired_pred_type,
                type_uid,
                w,
            )
            results_list.append(f1_type)

        results_dict = {}
        print_log("\n===========================Evaluation=======================", logger=logger)
        print_log("F1 Detection:{}".format(results_list[0]), logger=logger)
        results_dict["F1d"] = results_list[0]
        for i in range(len(self.cat_ids)):
            print_log("F1 Type{}:{}".format(self.cat_ids[i], results_list[i + 2]), logger=logger)
            results_dict["F1_Type{}".format(self.cat_ids[i])] = results_list[i + 2]
        print_log("===========================Evaluation=======================\n", logger=logger)

        return results_dict

    def pair_coordinates(self, setA, setB, radius):
        """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
        unique pairing (largest possible match) when pairing points in set B
        against points in set A, using distance as cost function.

        Args:
            setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                        of N different points
            radius: valid area around a point in setA to consider
                    a given coordinate in setB a candidate for match
        Return:
            pairing: pairing is an array of indices
            where point at index pairing[0] in set A paired with point
            in set B at index pairing[1]
            unparedA, unpairedB: remaining poitn in set A and set B unpaired

        """
        # * Euclidean distance as the cost matrix
        pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

        # * Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensured
        indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

        # extract the paired cost and remove instances
        # outside of designated radius
        pair_cost = pair_distance[indicesA, paired_indicesB]

        pairedA = indicesA[pair_cost <= radius]
        pairedB = paired_indicesB[pair_cost <= radius]

        pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
        unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
        unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
        return pairing, unpairedA, unpairedB

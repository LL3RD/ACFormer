import asyncio
from argparse import ArgumentParser

from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import scipy
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict, get_sliced_prediction_cell

from pycocotools import mask as maskUtils

from stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates
)
import warnings

warnings.filterwarnings('ignore')


def annToRLE(segm, img_size):
    h, w = img_size
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    return rle


def annToMask(segm, img_size):
    rle = annToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_file', default="/data2/huangjunjia/coco/CoNSeP/HoverNet_CoNSeP_1000x1000/Test/Images/",
                        help='Image file')
    parser.add_argument('--config',
                        default="/data2/huangjunjia/coco/CoNSeP/E2EC/E2EC_Swin/e2ec_swin.py",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default="/data2/huangjunjia/coco/CoNSeP/E2EC/E2EC_Swin/best_bbox_mAP_iter_1600.pth",
                        help='Checkpoint file')
    parser.add_argument('--gt-file',
                        default="/data2/huangjunjia/coco/CoNSeP/HoverNet_CoNSeP_1000x1000/Test/Labels/",
                        help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:3', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.2, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    num_classes = 3
    model = AutoDetectionModel.from_pretrained(
        model_type='mmdet',
        model_path=args.checkpoint,
        config_path=args.config,
        confidence_threshold=0.5,
        image_size=512,
        device=args.device
    )
    result = get_prediction("/data2/huangjunjia/coco/CoNSeP/HoverNet_SAHI_250_8/Test/test_14_0_0_250_250.jpg", model)
    # result.export_visuals(export_dir="/data2/huangjunjia/coco/CoNSeP/debug/debug")

    img_size = (1000, 1000)
    inst_map = np.zeros(img_size)
    result_coco = result.to_coco_annotations()
    for idx, coco in enumerate(result_coco):
        segm = coco['segmentation']
        for seg in segm:
            if len(seg) == 0:
                continue
            bin_segm = annToMask([seg], img_size)
            inst_map[np.where(bin_segm == 1)] = idx + 1

    plt.imshow(inst_map)
    plt.show()

    result = get_sliced_prediction(
        "/data2/huangjunjia/coco/CoNSeP/HoverNet_CoNSeP_1000x1000/Test/Images/test_14.png", model,
        slice_height=250,
        slice_width=250,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_threshold=0.5,
    )
    # #
    img_size = (1000, 1000)
    inst_map = np.zeros(img_size)
    result_coco = result.to_coco_annotations()
    for idx, coco in enumerate(result_coco):
        segm = coco['segmentation']
        for seg in segm:
            if len(seg) == 0:
                continue
            bin_segm = annToMask([seg], img_size)
            inst_map[np.where(bin_segm == 1)] = idx + 1

    plt.imshow(inst_map)
    plt.show()

    # return

    mat = sio.loadmat("/data2/huangjunjia/coco/CoNSeP/HoverNet_Patch/consep_1000x1000/seg_01_49/mat/test_14.mat")
    plt.imshow(mat["inst_map"])
    plt.show()

    mat = sio.loadmat("/data2/huangjunjia/coco/CoNSeP/HoverNet_CoNSeP_1000x1000/Test/Labels/test_14.mat")
    plt.imshow(mat["inst_map"])
    plt.show()

    # result.export_visuals(export_dir="/data2/huangjunjia/coco/CoNSeP/debug/debug_slice")

    plt.imshow(cv2.imread("/data2/huangjunjia/coco/CoNSeP/debug/debug_slice/prediction_visual.png")[:, :, ::-1])
    plt.show()

    metrics_name = ['DICE', 'AJI', 'DQ', 'SQ', 'PQ', 'AJI_PLUS']
    metrics = [[], [], [], [], [], []]

    for img_indx, img_name in enumerate(os.listdir(args.img_file)):
        gt_path = os.path.join(args.gt_file, img_name[:-4] + '.mat')
        true_info = sio.loadmat(gt_path)
        true = (true_info["inst_map"]).astype("int32")

        pred = np.zeros(img_size)
        pred = pred.astype("int32")

        result = get_sliced_prediction(
            os.path.join(args.img_file, img_name),
            model,
            slice_height=250,
            slice_width=250,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        result_coco = result.to_coco_annotations()
        for idx, coco in enumerate(result_coco):
            segm = coco['segmentation']
            for seg in segm:
                if len(seg) == 0:
                    continue
                bin_segm = annToMask([seg], img_size)
                pred[np.where(bin_segm == 1)] = idx + 1

        true = remap_label(true, by_size=False)
        pred = remap_label(pred, by_size=False)

        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        dice = get_dice_1(true, pred)
        fast_aji = get_fast_aji(true, pred)
        fast_aji_plus = get_fast_aji_plus(true, pred)
        metrics[0].append(dice)
        metrics[1].append(fast_aji)
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(fast_aji_plus)

    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print("===========================Evaluation=======================")
    for i, name in enumerate(metrics_name):
        print("{}: {}".format(name, metrics_avg[i]))
    print("===========================Evaluation=======================")

if __name__ == '__main__':
    args = parse_args()
    main(args)

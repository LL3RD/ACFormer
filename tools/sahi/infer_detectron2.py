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
                        default="/data2/huangjunjia/coco/CoNSeP/MaskDINO_SAHI/Mask2Former/config.yaml",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default="/data2/huangjunjia/coco/CoNSeP/MaskDINO_SAHI/Mask2Former/model_0049999.pth",
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
    # build the model from a config file and a checkpoint file

    # model = init_detector(args.config, args.checkpoint, device=args.device)
    # result = inference_detector(model, "/data2/huangjunjia/coco/CoNSeP/HoverNet_SAHI/Test/test_14_600_400_850_650.jpg")

    model = AutoDetectionModel.from_pretrained(
        model_type='detectron2',
        model_path=args.checkpoint,
        config_path=args.config,
        confidence_threshold=0.5,
        image_size=1024,
        device=args.device
    )
    #

    # result = get_prediction("/data2/huangjunjia/coco/CoNSeP/HoverNet_CoNSeP_1000x1000/Test/Images/test_14.png", model)
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

    # plt.imshow(cv2.imread("/data2/huangjunjia/coco/CoNSeP/debug/debug/prediction_visual.png"))
    # plt.show()

    # return
    result = get_sliced_prediction_cell(
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

    # return
    metrics_name = ['DICE', 'AJI', 'DQ', 'SQ', 'PQ', 'AJI_PLUS']
    metrics = [[], [], [], [], [], []]

    for img_indx, img_name in enumerate(os.listdir(args.img_file)):
        gt_path = os.path.join(args.gt_file, img_name[:-4] + '.mat')
        true_info = sio.loadmat(gt_path)
        true = (true_info["inst_map"]).astype("int32")

        pred = np.zeros(img_size)
        pred = pred.astype("int32")

        result = get_sliced_prediction_cell(
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


#     paired_all = []
#     unpaired_true_all = []
#     unpaired_pred_all = []
#     true_inst_type_all = []
#     pred_inst_type_all = []
#
#     for img_indx, img_name in enumerate(os.listdir(args.img_file)):
#         gt_path = os.path.join(args.gt_file, img_name[:-4] + '.mat')
#         true_info = sio.loadmat(gt_path)
#         true_centroid = (true_info["inst_centroid"]).astype("float32")
#         true_inst_type = (true_info["inst_type"]).astype("int32")
#
#         if true_centroid.shape[0] != 0:
#             true_inst_type = true_inst_type[:, 0]
#         else:  # no instance at all
#             true_centroid = np.array([[0, 0]])
#             true_inst_type = np.array([0])
#
#         # pred info
#         img_path = os.path.join(args.img_file, img_name)
#
#
#         # Save Results
#
#         # Visual
#         # import matplotlib.patches as patches
#         # image = mmcv.imread(img_path)
#         # w = image.shape[0]
#         # h = image.shape[1]
#         # dpi = 100
#         # fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
#         # axes = fig.add_axes([0, 0, 1, 1])
#         # axes.set_axis_off()
#         # axes.imshow(image)
#         # currentAxis = fig.gca()
#
#         result = inference_detector(model, img_path)
#
#         pred_centroid = []
#         pred_inst_type = []
#
#         # Visual
#         # color = {0:'r', 1:'g', 2:'b'}
#         for i in range(num_classes):
#             classes = result[i][np.where(result[i][:, 4] > 0.3)]
#             for class_ in classes:
#                 x, y, x2, y2 = class_[:4]
#         #         rect = patches.Rectangle((int(x), int(y)), int(x2-x), int(y2-y), linewidth=0.5, edgecolor=color[i], facecolor='none')
#         #         currentAxis.add_patch(rect)
#             classes[:, 0] = (classes[:, 0] + classes[:, 2]) / 2
#             classes[:, 1] = (classes[:, 1] + classes[:, 3]) / 2
#             classes = classes[:, :2]
#             pred_centroid.extend(classes)
#             pred_type = np.full((classes.shape[0], 1), i + 1)
#             pred_inst_type.extend(pred_type)
#         #
#         # plt.axis("off")
#         # if img_name=="test_12.png":
#         #     plt.savefig("/data2/huangjunjia/coco/Visual/Rect_Pred.png", bbox_inches='tight',pad_inches = 0)
#         # plt.show()
#
#         pred_centroid = np.asarray(pred_centroid).astype("float32")
#         pred_inst_type = np.asarray(pred_inst_type).astype("int32")
#
#         if pred_centroid.shape[0] != 0:
#             pred_inst_type = pred_inst_type[:, 0]
#         else:  # no instance at all
#             pred_centroid = np.array([[0, 0]])
#             pred_inst_type = np.array([0])
#
#         # Save Predict Result
#         pred_mat = {}
#         pred_mat["inst_centroids"] = pred_centroid
#         pred_mat["inst_type"] = pred_inst_type
#         # sio.savemat(os.path.join("/data2/huangjunjia/coco/Visual/QuantitiveC/CoNSeP/TOOD/", img_name[:-4] + ".mat"),
#         #             pred_mat)
#         print(np.unique(pred_inst_type))
#
#         paired, unpaired_true, unpaired_pred = pair_coordinates(
#             true_centroid, pred_centroid, 6
#         )
#
#         true_idx_offset = (
#             true_idx_offset + true_inst_type_all[-1].shape[0] if img_indx != 0 else 0
#         )
#         pred_idx_offset = (
#             pred_idx_offset + pred_inst_type_all[-1].shape[0] if img_indx != 0 else 0
#         )
#         true_inst_type_all.append(true_inst_type)
#         pred_inst_type_all.append(pred_inst_type)
#
#         if paired.shape[0] != 0:  # ! sanity
#             paired[:, 0] += true_idx_offset
#             paired[:, 1] += pred_idx_offset
#             paired_all.append(paired)
#
#         unpaired_true += true_idx_offset
#         unpaired_pred += pred_idx_offset
#         unpaired_true_all.append(unpaired_true)
#         unpaired_pred_all.append(unpaired_pred)
#
#     paired_all = np.concatenate(paired_all, axis=0)
#     unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
#     unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
#     true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
#     pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)
#
#     paired_true_type = true_inst_type_all[paired_all[:, 0]]
#     paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
#     unpaired_true_type = true_inst_type_all[unpaired_true_all]
#     unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]
#
#     def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
#         type_samples = (paired_true == type_id) | (paired_pred == type_id)
#
#         paired_true = paired_true[type_samples]
#         paired_pred = paired_pred[type_samples]
#
#         tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
#         tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
#         fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
#         fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()
#
#         fp_d = (unpaired_pred == type_id).sum()
#         fn_d = (unpaired_true == type_id).sum()
#
#         f1_type = (2 * (tp_dt + tn_dt)) / (
#                 2 * (tp_dt + tn_dt)
#                 + w[0] * fp_dt
#                 + w[1] * fn_dt
#                 + w[2] * fp_d
#                 + w[3] * fn_d
#         )
#         return f1_type
#
#     w = [1, 1]
#     tp_d = paired_pred_type.shape[0]
#     fp_d = unpaired_pred_type.shape[0]
#     fn_d = unpaired_true_type.shape[0]
#
#     tp_tn_dt = (paired_pred_type == paired_true_type).sum()
#     fp_fn_dt = (paired_pred_type != paired_true_type).sum()
#
#     acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
#     f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)
#
#     w = [2, 2, 1, 1]
#
#     # if type_uid_list is None:
#     type_uid_list = np.unique(true_inst_type_all).tolist()
#
#     results_list = [f1_d, acc_type]
#     for type_uid in type_uid_list:
#         f1_type = _f1_type(
#             paired_true_type,
#             paired_pred_type,
#             unpaired_true_type,
#             unpaired_pred_type,
#             type_uid,
#             w,
#         )
#         results_list.append(f1_type)
#
#     np.set_printoptions(formatter={"float": "{: 0.5f}".format})
#     results_list = np.array(results_list)
#
#     print("\n===========================Evaluation=======================")
#     print("F1 Detection:{}".format(results_list[0]))
#     for i in range(num_classes):
#         print("F1 Type{}:{}".format(i, results_list[i + 2]))
#     print("===========================Evaluation=======================\n")
#     return
#
#
# def pair_coordinates(setA, setB, radius):
#     """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
#     unique pairing (largest possible match) when pairing points in set B
#     against points in set A, using distance as cost function.
#
#     Args:
#         setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
#                     of N different points
#         radius: valid area around a point in setA to consider
#                 a given coordinate in setB a candidate for match
#     Return:
#         pairing: pairing is an array of indices
#         where point at index pairing[0] in set A paired with point
#         in set B at index pairing[1]
#         unparedA, unpairedB: remaining poitn in set A and set B unpaired
#
#     """
#     # * Euclidean distance as the cost matrix
#     pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')
#
#     # * Munkres pairing with scipy library
#     # the algorithm return (row indices, matched column indices)
#     # if there is multiple same cost in a row, index of first occurence
#     # is return, thus the unique pairing is ensured
#     indicesA, paired_indicesB = linear_sum_assignment(pair_distance)
#
#     # extract the paired cost and remove instances
#     # outside of designated radius
#     pair_cost = pair_distance[indicesA, paired_indicesB]
#
#     pairedA = indicesA[pair_cost <= radius]
#     pairedB = paired_indicesB[pair_cost <= radius]
#
#     pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
#     unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
#     unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
#     return pairing, unpairedA, unpairedB


if __name__ == '__main__':
    args = parse_args()
    main(args)

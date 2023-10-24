from argparse import ArgumentParser

# from mmdet.apis import (async_inference_detector, inference_detector,
#                         init_detector, show_result_pyplot)

from ssod.apis.inference import init_detector, inference_detector
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import scipy.io as sio
import os
import scipy
from scipy.optimize import linear_sum_assignment


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_file',
                        default="Path to BRCA/HoverNet_BRCA/Test/Images",
                        help='Image file')
    parser.add_argument('--img_file_256',
                        default="Path to BRCA/COCO_BRCA/BRCA_Test/",
                        help='Image file')
    parser.add_argument('--config',
                        default="Path to ACFormer_BRCA.py",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default="Path to ACFormer_BRCA.pth",
                        help='Checkpoint file')
    parser.add_argument('--gt-file',
                        default="Path to BRCA/HoverNet_BRCA/Test/Labels",
                        help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:7', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.4, help='bbox score threshold')
    parser.add_argument(
        '--mode', type=str, default="globals", help='inference on')
    parser.add_argument(
        '--dataset', type=str, default="BRCA", help='inference on')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    num_classes = 3
    model = init_detector(
        args.config, args.checkpoint, device=args.device
    )
    model.inference_on = args.mode
    thr = args.score_thr

    # for thrs in range(35, 45, 1):
    if thr:
        # thr = thrs / 100.
        img_list = os.listdir(args.img_file)

        paired_all = []
        unpaired_true_all = []
        unpaired_pred_all = []
        true_inst_type_all = []
        pred_inst_type_all = []

        for img_indx, img_name in enumerate(img_list):
            # gt info
            gt_path = os.path.join(args.gt_file, img_name[:-4] + '.mat')
            true_info = sio.loadmat(gt_path)
            true_centroid = (true_info["inst_centroid"]).astype("float32")
            true_inst_type = (true_info["inst_type"]).astype("int32")

            if true_centroid.shape[0] != 0:
                true_inst_type = true_inst_type[:, 0]
            else:  # no instance at all
                true_centroid = np.array([[0, 0]])
                true_inst_type = np.array([0])

            # MCS classes
            if args.dataset == "CoNSeP":
                true_inst_type[(true_inst_type == 2)] = 9
                true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 2
                true_inst_type[
                    (true_inst_type == 1) | (true_inst_type == 5) | (true_inst_type == 6) | (
                            true_inst_type == 7)] = 3
                true_inst_type[(true_inst_type == 9)] = 1

            elif args.dataset == "BRCA":
                true_inst_type[(true_inst_type == 2)] = 2
                true_inst_type[(true_inst_type == 1)] = 1
                true_inst_type[(true_inst_type == 3)] = 3

            pred_centroid = []
            pred_inst_type = []

            col_offset = 0
            raw_offset = 0

            for raw in range(2):
                for col in range(2):
                    img_path = os.path.join(args.img_file_256, img_name[:-4] + "_{}{}.png".format(raw, col))
                    result = inference_detector(model, img_path)

                    if raw == 0 and col == 0:
                        img_shape = mmcv.imread(img_path).shape
                        col_offset = img_shape[0]
                        raw_offset = img_shape[1]

                    for cls in range(num_classes):
                        classes = result[cls][np.where(result[cls][:, 2] > thr)]
                        classes = classes[:, :2]

                        classes[:, 0] += col * raw_offset
                        classes[:, 1] += raw * col_offset

                        pred_centroid.extend(classes)
                        pred_type = np.full((classes.shape[0], 1), cls + 1)
                        pred_inst_type.extend(pred_type)

            pred_centroid = np.asarray(pred_centroid).astype("float32")
            pred_inst_type = np.asarray(pred_inst_type).astype("int32")

            if pred_centroid.shape[0] != 0:
                pred_inst_type = pred_inst_type[:, 0]
            else:  # no instance at all
                pred_centroid = np.array([[0, 0]])
                pred_inst_type = np.array([0])

            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_centroid, pred_centroid, 6
            )

            true_idx_offset = (
                true_idx_offset + true_inst_type_all[-1].shape[0] if img_indx != 0 else 0
            )
            pred_idx_offset = (
                pred_idx_offset + pred_inst_type_all[-1].shape[0] if img_indx != 0 else 0
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

        np.set_printoptions(formatter={"float": "{: 0.5f}".format})
        results_list = np.array(results_list)

        print("\n===========================Evaluation=======================")
        print("Thr: {}".format(thr))
        print("F1 Detection:{}".format(results_list[0]))
        for i in range(num_classes):
            print("F1 Type{}:{}".format(i, results_list[i + 2]))
        print("===========================Evaluation=======================\n")
    return


def pair_coordinates(setA, setB, radius):
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


if __name__ == '__main__':
    args = parse_args()
    main(args)

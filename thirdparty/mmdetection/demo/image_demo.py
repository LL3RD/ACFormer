# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import scipy.io as sio


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',
                        default="/data/huangjunjia/StomachDataset/CoNSeP/MCSpatNet/datasets/COCO_STUFF/CoNSeP_Test/test_1.png",
                        help='Image file')
    parser.add_argument('--config', default="/data2/huangjunjia/coco/debug_cell/deformable_detr.py", help='Config file')
    parser.add_argument('--checkpoint', default="/data2/huangjunjia/coco/debug_cell/iter_112000.pth",
                        help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:6', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    image = mmcv.imread(args.img, channel_order='rgb')
    # test a single image
    result = inference_detector(model, args.img)
    class1 = result[1][np.where(result[1][:, 2] > 0.5)]
    plt.imshow(image)
    plt.scatter(class1[:, 0], class1[:, 1], s=1, color=(1, 0, 0))
    plt.show()

    mat = sio.loadmat(
        "/data/huangjunjia/StomachDataset/CoNSeP/MCSpatNet/datasets/debug/consep/test/gt_mat/test_1.mat")

    class1_gt = mat["inst_centroid"][np.where(mat["inst_type"] == 4)[0]]
    plt.imshow(image)
    plt.scatter(class1_gt[:, 0], class1_gt[:, 1], s=1, color=(1, 0, 0))

    plt.scatter(class1[:, 0], class1[:, 1], s=1, color=(0, 1, 0))
    plt.show()
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)

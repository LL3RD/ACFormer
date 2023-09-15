# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import torch
import numpy as np
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import time

def init_detector(config, checkpoint=None, device="cuda:0", cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.train_cfg = None

    if hasattr(config.model, "model"):
        config.model.model.pretrained = None
        config.model.model.train_cfg = None
    else:
        config.model.pretrained = None

    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        map_loc = "cpu" if device == "cpu" else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "Class names are not saved in the checkpoint's "
                "meta data, use COCO classes by default."
            )
            model.CLASSES = get_classes("coco")
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]


    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    if not is_batch:
        return results[0]
    else:
        return results


def save_result(model, img, result, score_thr=0.3, out_file="res.png"):
    """Save the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str): Specifies where to save the visualization result
    """
    if hasattr(model, "module"):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        out_file=out_file,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
    )


#
#
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import matplotlib.pyplot as plt
# from .utils import Transform2D, spatial_pyramid_pool, _get_trans_mat, _transform_points
#
# def denorm(img):
#     std=[58.395, 57.12, 57.375]
#     mean=[123.675, 116.28, 103.53]
#     denorm_img = img*std+mean
#     return denorm_img
#
# data_group = data.copy()
# data_group['img'] = data_group['img'][0]
# x, out_size = model.patch_embed(data_group["img"])
# absolute_pos_embed = F.interpolate(
#             model.absolute_pos_embed, size=out_size, mode="bicubic"
#         )
# x = x + absolute_pos_embed.flatten(2).transpose(1, 2)
# x = torch.cat((model.affine_token.expand(x.shape[0], -1, -1), x), dim=1)
# x = model.drop_after_pos(x)
# x = model.transformer(x)
# theta = model.fc_loc(x[:, 0])
# theta = theta.view(-1, model.affine_number, 2, 3)
# print(theta)
#
# for i, the in enumerate(theta):
#     img_shape = data_group["img_metas"][i][i]["img_shape"]
#     grid = F.affine_grid(the.unsqueeze(0), [1, img_shape[2], img_shape[0], img_shape[1]])
#     affine_img = F.grid_sample(data_group["img"][i].unsqueeze(0), grid)
#     plt.imshow(denorm(affine_img.cpu()[0].permute(1, 2, 0).detach().numpy())/255.)
#     plt.show()
#     plt.imshow(denorm(data_group["img"].cpu()[0].permute(1, 2, 0).detach().numpy())/255.)
#     plt.show()










#
#
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import matplotlib.pyplot as plt
# from .utils import Transform2D, spatial_pyramid_pool, _get_trans_mat, _transform_points
#
# def denorm(img):
#     std=[58.395, 57.12, 57.375]
#     mean=[123.675, 116.28, 103.53]
#     denorm_img = img*std+mean
#     return denorm_img
#
# data_group = data.copy()
# data_group['img'] = data_group['img'][0]
# xs = model.localization(data_group["img"])
# spp = spatial_pyramid_pool(xs, 1, [int(xs.shape[2]), int(xs.shape[3])], [4, 2, 1])
# theta = model.fc_loc(spp)
# theta = theta.view(-1, 2, 3)
# print(theta)
#
# for i, the in enumerate(theta):
#     img_shape = data_group["img_metas"][i][i]["img_shape"]
#     grid = F.affine_grid(the.unsqueeze(0), [1, img_shape[2], img_shape[0], img_shape[1]])
#     affine_img = F.grid_sample(data_group["img"][i].unsqueeze(0), grid)
#     plt.imshow(denorm(affine_img.cpu()[0].permute(1, 2, 0).detach().numpy())/255.)
#     plt.show()
#     plt.imshow(denorm(data_group["img"].cpu()[0].permute(1, 2, 0).detach().numpy())/255.)
#     plt.show()


#     gt_points[i][:, 0] -= img_shape[1] / 2
#     gt_points[i][:, 1] -= img_shape[0] / 2
#     trans_mat = the.detach()
#     trans_mat = torch.cat((trans_mat, torch.tensor([[0, 0, 1]]).to(trans_mat.device)))
#     M = _get_trans_mat([trans_mat], [ori_mat])
#
#     gt_points[i] = _transform_points([gt_points[i]], M)[0]
#     gt_points[i][:, 0] += img_shape[1] / 2
#     gt_points[i][:, 1] += img_shape[0] / 2
#
#     valid_inds = (gt_points[i][:, 0] >= 0) & (gt_points[i][:, 0] < img_shape[1]) & \
#                  (gt_points[i][:, 1] >= 0) & (gt_points[i][:, 1] < img_shape[0])
#     gt_points[i] = gt_points[i][valid_inds, :]
#     gt_labels.append(data_group["gt_labels"][i][valid_inds])



'''
RGB = denorm(data_groups["img"][0].detach().permute(1, 2, 0).cpu().numpy())/255.
w=RGB.shape[0]
h=RGB.shape[1]
dpi=400
fig=plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
axes=fig.add_axes([0,0,1,1])
axes.set_axis_off()
axes.imshow(RGB)
plt.savefig("/data2/huangjunjia/coco/Visual/global_img.png", bbox_inches='tight')
plt.show()
'''
# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .detr import DETR
from mmdet.core import point2result


@DETECTORS.register_module()
class DeformableDETR_CellDet(DETR):

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_points=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        result_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale
        )  # [[xy,score],cls]
        point_results = [
            point2result(det_points, det_labels, self.bbox_head.num_classes)
         for det_points, det_labels in result_list
        ]
        return point_results



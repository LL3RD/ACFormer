import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n
from collections.abc import Sequence
from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid, ParallelBlock
import torch.nn as nn
import torch.nn.functional as F
import math
from mmdet.models.utils.transformer import PatchEmbed
from mmcv.cnn.utils.weight_init import trunc_normal_


@DETECTORS.register_module()
class GlobalLocal_STN_Sequence(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None, stn_cfg=None):
        super(GlobalLocal_STN_Sequence, self).__init__(
            dict(globals=build_detector(model), locals=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("globals")
            self.global_weight = self.train_cfg.global_weight
        # STN Sequence
        self.embed = stn_cfg['embed']
        self.affine_number = 4

        self.patch_embed = PatchEmbed(in_channels=3, embed_dims=self.embed, stride=4, kernel_size=4,
                                      norm_cfg=dict(type='LN'))
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, self.embed, 250 // 4, 250 // 4)
        )
        trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=0.1)

        self.affine_token = nn.Parameter(
            torch.zeros(1, 1, self.embed)
        )
        nn.init.normal_(self.affine_token, std=1e-6)
        self.transformer = ParallelBlock(dim=self.embed, num_heads=12, qkv_bias=True, )

        self.fc_loc = nn.Sequential(
            nn.Linear(self.embed, self.embed),
            nn.ReLU(True),
            nn.Linear(self.embed, self.affine_number * 2 * 3)
        )

        self.transformer.apply(self.init_weight_)
        self.fc_loc.apply(self.init_weight_)

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.5, 0, -0.5, 0, 0.5, -0.5,
                                                     0.5, 0, -0.5, 0, 0.5, 0.5,
                                                     0.5, 0, 0.5, 0, 0.5, -0.5,
                                                     0.5, 0, 0.5, 0, 0.5, 0.5], dtype=torch.float))

        self.ori_mat = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()

        # init weight

    def init_weight_(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal(m.weight, mean=0, std=0.5)
        if type(m) == nn.Linear:
            torch.nn.init.uniform(m.weight, a=-0.1, b=0.1)
            m.bias.data.fill_(0.01)

    def stn(self, data_groups):
        x = data_groups["ori_img"]
        x, out_size = self.patch_embed(x)
        absolute_pos_embed = F.interpolate(
            self.absolute_pos_embed, size=out_size, mode="bicubic"
        )
        x = x + absolute_pos_embed.flatten(2).transpose(1, 2)
        x = torch.cat((self.affine_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.drop_after_pos(x)
        x = self.transformer(x)
        theta = self.fc_loc(x[:, 0])
        theta = theta.view(-1, self.affine_number, 2, 3)

        # clip
        theta = torch.clamp(theta, min=torch.tensor([[0.2, -1, -0.5], [-1, 0.2, -0.5]]).to(theta.device),
                            max=torch.tensor([[2, 1, 0.5], [1, 2, 0.5]]).to(theta.device))

        #
        # RandCrop
        # theta = torch.tensor([[[0.6, 0, float(torch.randint(low=-5, high=6, size=(1,))[0] / 10.)],
        #                        [0, 0.6, float(torch.randint(low=-5, high=6, size=(1,))[0] / 10.)]],
        #                       [[0.6, 0, float(torch.randint(low=-5, high=6, size=(1,))[0] / 10.)],
        #                        [0, 0.6, float(torch.randint(low=-5, high=6, size=(1,))[0] / 10.)]],
        #                       [[0.6, 0, float(torch.randint(low=-5, high=6, size=(1,))[0] / 10.)],
        #                        [0, 0.6, float(torch.randint(low=-5, high=6, size=(1,))[0] / 10.)]],
        #                       [[0.6, 0, float(torch.randint(low=-5, high=6, size=(1,))[0] / 10.)],
        #                        [0, 0.6, float(torch.randint(low=-5, high=6, size=(1,))[0] / 10.)]], ]).unsqueeze(0).float().to(data_groups["img"].device)

        #
        affine_results = []
        gt_labels = []
        gt_points = [data_groups['gt_points'][0].clone() for i in range(self.affine_number)]
        ori_mat = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float().to(data_groups["img"].device)

        for i, the_b in enumerate(theta):
            batch_input_shape = data_groups["img_metas"][i]["batch_input_shape"]
            img_shape = data_groups["img_metas"][i]["img_shape"]
            ori_img = data_groups["img"][i][:, :img_shape[0], :img_shape[1]]

            for j, the in enumerate(the_b):
                grid = F.affine_grid(the.unsqueeze(0), [1, img_shape[2], img_shape[0], img_shape[1]])

                # Trans Image
                affine_img = F.grid_sample(ori_img.unsqueeze(0), grid)
                affine_img = F.pad(affine_img,
                                   (0, batch_input_shape[0] - img_shape[0], 0, batch_input_shape[1] - img_shape[1]),
                                   "constant", 0)
                affine_results.append(affine_img)

                # Trans GT Points
                gt_points[j][:, 0] -= img_shape[1] / 2
                gt_points[j][:, 1] -= img_shape[0] / 2
                trans_mat = the.detach()
                trans_mat = torch.cat((trans_mat, torch.tensor([[0, 0, 1]]).to(trans_mat.device)))

                # Convert to Affine_mode
                # https://discuss.pytorch.org/t/how-to-convert-an-affine-transform-matrix-into-theta-to-use-torch-nn-functional-affine-grid/24315/4
                trans_mat[0, 1] = trans_mat[0, 1] * img_shape[1] / img_shape[0]
                trans_mat[0, 2] = (trans_mat[0, 2]) * img_shape[1] / 2
                trans_mat[1, 0] = trans_mat[1, 0] * img_shape[0] / img_shape[1]
                trans_mat[1, 2] = (trans_mat[1, 2]) * img_shape[0] / 2

                M = self._get_trans_mat([trans_mat], [ori_mat])

                gt_points[j] = self._transform_points([gt_points[j]], M)[0]
                gt_points[j][:, 0] += img_shape[1] / 2
                gt_points[j][:, 1] += img_shape[0] / 2

                valid_inds = (gt_points[j][:, 0] >= 0) & (gt_points[j][:, 0] < img_shape[1]) & \
                             (gt_points[j][:, 1] >= 0) & (gt_points[j][:, 1] < img_shape[0])
                gt_points[j] = gt_points[j][valid_inds, :]
                if len(gt_points[j]) == 0:
                    gt_labels.append(data_groups["gt_labels"][i][1:1])
                else:
                    gt_labels.append(data_groups["gt_labels"][i][valid_inds])

        affine_results = torch.vstack(affine_results)

        data_groups["trans_img"] = affine_results
        data_groups["trans_gt_points"] = gt_points
        data_groups["trans_gt_labels"] = gt_labels
        data_groups["trans_matrix"] = theta.detach()

        data_groups["img_metas"] = [data_groups["img_metas"][0] for i in range(self.affine_number)]

        return data_groups

    # import matplotlib.pyplot as plt
    # plt.imshow(affine_results[0][0].detach().permute(1, 2, 0).cpu())
    # plt.scatter(gt_points[0].cpu()[:, 0], gt_points[0].cpu()[:, 1], s=1, color=(1, 0, 0))
    # plt.show()
    def loss(self,
             all_cls_scores,
             all_point_preds,
             enc_cls_scores,
             enc_point_preds,
             gt_points_list,
             gt_labels_list,
             img_metas,
             mode='local',
             gt_bboxes_ignore=None):

        num_dec_layers = len(all_cls_scores)
        loss_single = eval('self.{}s.bbox_head.loss_single'.format(mode))
        if mode == 'local':
            all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
            all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
            all_gt_points_ignore_list = [
                gt_bboxes_ignore for _ in range(num_dec_layers)
            ]
            img_metas_list = [img_metas for _ in range(num_dec_layers)]

        else:
            all_gt_points_list = gt_points_list
            all_gt_labels_list = gt_labels_list
            all_gt_points_ignore_list = [
                gt_bboxes_ignore for _ in range(num_dec_layers)
            ]
            img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_points = multi_apply(
            loss_single, all_cls_scores, all_point_preds,
            all_gt_points_list, all_gt_labels_list, img_metas_list,
            all_gt_points_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i]) if mode == "local" else torch.zeros_like(gt_labels_list[-1][i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_point = \
                loss_single(enc_cls_scores, enc_point_preds,
                            gt_points_list if mode == "local" else gt_points_list[-1],
                            binary_labels_list,
                            img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_point

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_points[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_point_i in zip(losses_cls[:-1],
                                            losses_points[:-1], ):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_point_i
            num_dec_layer += 1
        return loss_dict

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        # ! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        # ! it means that at least one sample for each group should be provided on each gpu.
        # ! In some situation, we can only put one image per gpu, we have to return the sum of loss
        # ! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        assert "local" in data_groups
        data_groups["local"] = self.stn(data_groups["local"])

        local_feat = self.locals.extract_feat(data_groups["local"]["trans_img"])
        local_outs = self.locals.bbox_head.forward(local_feat, data_groups["local"]["img_metas"])
        local_loss_inputs = local_outs + (
            data_groups["local"]["trans_gt_points"], data_groups["local"]["trans_gt_labels"],
            data_groups["local"]["img_metas"])

        local_loss = self.loss(*local_loss_inputs, mode="local")
        local_loss = {"local_" + k: v for k, v in local_loss.items()}
        loss.update(**local_loss)

        global_loss = weighted_loss(
            self.foward_global_train(
                data_groups["local"], local_outs
            ),
            weight=self.global_weight,
        )
        global_loss = {"global_" + k: v for k, v in global_loss.items()}
        loss.update(**global_loss)

        return loss

    def foward_global_train(self, local_data, local_outs):
        # sort the teacher and student input to avoid some bugs
        # tnames = [meta["filename"] for meta in global_data["img_metas"]]
        # snames = [meta["filename"] for meta in local_data["img_metas"]]
        # tidx = [tnames.index(name) for name in snames]
        tidx = [i for i in range(1)]
        with torch.no_grad():
            global_info = self.extract_global_info(
                local_data["img"][
                    torch.Tensor(tidx).to(local_data["img"].device).long()
                ],
                [local_data["img_metas"][idx] for idx in tidx],
                [local_data["proposals"][idx] for idx in tidx]
                if ("proposals" in local_data)
                   and (local_data["proposals"] is not None)
                else None,
            )

        pseudo_points = [[bat[:, :2].detach().clone() for bat in lay] for lay in global_info["det_points"]]
        pseudo_points = [[lay[0].clone() for bat in range(len(local_data["img_metas"]))] for lay in pseudo_points]
        pseudo_labels = [[lay[0].clone() for bat in range(len(local_data["img_metas"]))] for lay in
                         global_info["det_labels"]]

        # pseudo_points = [[bat[:, :2].detach().clone() for bat in lay] for lay in global_info["det_points"]]
        # pseudo_points = [[local_data["gt_points"][0].clone() for bat in range(len(local_data["img_metas"]))] for lay in pseudo_points]
        # pseudo_labels = [[local_data["gt_labels"][0].clone() for bat in range(len(local_data["img_metas"]))] for lay in global_info["det_labels"]]

        trans_matrix = local_data["trans_matrix"]
        for i, trans_mat in enumerate(trans_matrix[0]):
            img_shape = local_data["img_metas"][i]["img_shape"]
            ori_mat = global_info['transform_matrix'][0]
            trans_mat = torch.cat((trans_mat, torch.tensor([[0, 0, 1]]).to(trans_mat.device)))

            # Convert to Affine_mode
            # https://discuss.pytorch.org/t/how-to-convert-an-affine-transform-matrix-into-theta-to-use-torch-nn-functional-affine-grid/24315/4
            trans_mat[0, 1] = trans_mat[0, 1] * img_shape[1] / img_shape[0]
            trans_mat[0, 2] = (trans_mat[0, 2]) * img_shape[1] / 2
            trans_mat[1, 0] = trans_mat[1, 0] * img_shape[0] / img_shape[1]
            trans_mat[1, 2] = (trans_mat[1, 2]) * img_shape[0] / 2

            M = self._get_trans_mat([trans_mat], [ori_mat])
            for p, pseudo_point in enumerate(pseudo_points):
                # Trans Pseudo Points
                pseudo_points[p][i][:, 0] -= img_shape[1] / 2
                pseudo_points[p][i][:, 1] -= img_shape[0] / 2
                pseudo_points[p][i] = self._transform_points([pseudo_points[p][i]], M)[0]
                pseudo_points[p][i][:, 0] += img_shape[1] / 2
                pseudo_points[p][i][:, 1] += img_shape[0] / 2

                valid_inds = (pseudo_points[p][i][:, 0] >= 0) & (pseudo_points[p][i][:, 0] < img_shape[1]) & \
                             (pseudo_points[p][i][:, 1] >= 0) & (pseudo_points[p][i][:, 1] < img_shape[0])
                pseudo_points[p][i] = pseudo_points[p][i][valid_inds, :]
                pseudo_labels[p][i] = pseudo_labels[p][i][valid_inds]

        global_input = local_outs + (pseudo_points, pseudo_labels, local_data["img_metas"])
        global_loss = self.loss(*global_input, mode="global")

        return global_loss

    def crop_valid(self, det_points, det_labels, offset, crop_size):
        valid_inds = (det_points[:, 0] >= offset[0]) & (det_points[:, 1] >= offset[1]) & (
                det_points[:, 0] <= offset[0] + crop_size[1]) & (det_points[:, 1] <= offset[1] + crop_size[0])
        return det_points[valid_inds, :] - offset[:2], det_labels[valid_inds]

    @force_fp32(apply_to=["points", "trans_mat"])
    def _transform_points(self, points, trans_mat):
        points = Transform2D.transform_points(points, trans_mat)
        return points

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_global_info(self, img, img_metas, proposals=None, **kwargs):
        global_info = {}
        feat = self.globals.extract_feat(img)
        global_info["backbone_feature"] = feat
        global_info["proposals"] = proposals

        proposal_list, proposal_label_list = self.globals.bbox_head.simple_all_test_bboxes(
            feat, img_metas, rescale=False
        )

        proposal_list = [[p_b.to(feat[0].device) for p_b in p] for p in proposal_list]
        proposal_list = [
            [p_b if p_b.shape[0] > 0 else p_b.new_zeros(0, 3) for p_b in p] for p in proposal_list
        ]
        proposal_label_list = [[p_b.to(feat[0].device) for p_b in p] for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")

        for num_layer in range(len(proposal_list)):
            for batch_idx in range(len(proposal_list[num_layer])):
                valid = proposal_list[num_layer][batch_idx][:, -1] > thr
                proposal_list[num_layer][batch_idx] = proposal_list[num_layer][batch_idx][valid]
                proposal_label_list[num_layer][batch_idx] = proposal_label_list[num_layer][batch_idx][valid]

        det_points = proposal_list
        det_labels = proposal_label_list
        global_info["det_points"] = det_points
        global_info["det_labels"] = det_labels
        global_info["transform_matrix"] = [
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        global_info["img_metas"] = img_metas
        return global_info

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        if not any(["globals" in key or "locals" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"globals." + k: state_dict[k] for k in keys})
            state_dict.update({"locals." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

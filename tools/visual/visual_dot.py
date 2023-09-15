import os
import mmcv
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

num_classes = 3
Pred_Mat_File = "/data/huangjunjia/StomachDataset/CoNSeP/MCSpatNet/reproduce/MCSpatNet_eval/pred_mat"
mat_list = os.listdir(Pred_Mat_File)
for mat in mat_list:
    mat_path = os.path.join(Pred_Mat_File, mat)
    pred_mat = sio.loadmat(mat_path)

    pred_centroid = pred_mat["inst_centroid"]
    pred_inst_type = pred_mat["inst_type"]

    if pred_centroid.shape[0] != 0:
        pred_inst_type = pred_inst_type[:, 0]
    else:  # no instance at all
        pred_centroid = np.array([[0, 0]])
        pred_inst_type = np.array([0])

    img = mmcv.imread(os.path.join(
        "/data/huangjunjia/StomachDataset/CoNSeP/MCSpatNet/reproduce/MCSpatNet_eval/mcspatnet_consep_1_e285",
        mat[:-4] + ".png"))
    W = img.shape[0]
    H = img.shape[1]
    dpi = 400
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.imshow(img)
    # plt.imshow(img)
    for cls in range(1, num_classes + 1):
        axes.scatter(pred_centroid[:, 0][np.where(pred_inst_type == cls)],
                     pred_centroid[:, 1][np.where(pred_inst_type == cls)], s=3, marker=".", linewidths=0,
                     color=(int(1 == cls), int(2 == cls), int(3 == cls)))
    # plt.axis("off")
    plt.savefig(os.path.join("/data/huangjunjia/StomachDataset/CoNSeP/MCSpatNet/reproduce/MCSpatNet_eval/pred_visual",
                             mat[:-4] + '.png'))
    plt.show()
    plt.clf()

# #Save Visual Result
# pred_save_path = os.path.join(args.config.rsplit("/", 1)[0], "Pred_Image")
# gt_save_path = os.path.join(args.config.rsplit("/", 1)[0], "GT_Image")
# os.makedirs(gt_save_path, exist_ok=True)
# os.makedirs(pred_save_path, exist_ok=True)
#
# img = mmcv.imread(os.path.join("/data2/huangjunjia/coco/CoNSeP/CoNSeP_Test", img_name))
# plt.imshow(img)
# for cls in range(1, num_classes + 1):
#     plt.scatter(true_centroid[:, 0][np.where(true_inst_type == cls)],
#                 true_centroid[:, 1][np.where(true_inst_type == cls)], s=1,
#                 color=(int(1 == cls), int(2 == cls), int(3 == cls)))
# plt.savefig(os.path.join(gt_save_path, img_name))
# # plt.show()
# plt.clf()
#
# plt.imshow(img)
# for cls in range(1, num_classes + 1):
#     plt.scatter(pred_centroid[:, 0][np.where(pred_inst_type == cls)],
#                 pred_centroid[:, 1][np.where(pred_inst_type == cls)], s=1,
#                 color=(int(1 == cls), int(2 == cls), int(3 == cls)))
# plt.savefig(os.path.join(pred_save_path, img_name))
# # plt.show()
# plt.clf()

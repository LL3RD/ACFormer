import numpy as np
import glob
import os
import sys
import skimage.io as io
from scipy import ndimage
import mmcv
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import json

from pycocotools.coco import COCO
import cv2

segmentation_id = 0


def gaussian_filter_density(img, points, point_class_map, mat, start_y=0, start_x=0, end_y=-1, end_x=-1):
    '''
        Build a KD-tree from the points, and for each point get the nearest neighbor point.
        The default Guassian width is 9.
        Sigma is adaptively selected to be min(nearest neighbor distance*0.125, 2) and truncate at 2*sigma.
        After generation of each point Gaussian, it is normalized and added to the final density map.
        A visualization of the generated maps is saved in <slide_name>_<img_name>.png and <slide_name>_<img_name>_binary.png
    '''
    img_shape = [img.shape[0], img.shape[1]]
    # print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)

    density_class = np.zeros((img.shape[0], img.shape[1], point_class_map.shape[2]), dtype=np.float32)
    if (end_y <= 0):
        end_y = img.shape[0]
    if (end_x <= 0):
        end_x = img.shape[1]
    gt_count = len(points)
    if gt_count == 0:
        return density
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=2)
    # print('generate density...')

    max_sigma = 2  # kernel size = 4, kernel_width=9

    mat["inst_map"] = np.zeros(img_shape, dtype=np.int32)
    mat["type_map"] = np.zeros(img_shape, dtype=np.int32)

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if (pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x):
            continue
        pt[1] -= start_y
        pt[0] -= start_x
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]) * 0.125
            sigma = min(max_sigma, sigma)
        else:
            sigma = max_sigma

        kernel_size = min(max_sigma * 2, int(2 * sigma + 0.5))
        sigma = kernel_size / 2
        kernel_width = kernel_size * 2 + 1
        # if(kernel_width < 9):
        #     print('i',i)
        #     print('distances',distances.shape)
        #     print('kernel_width',kernel_width)
        pnt_density = scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant', truncate=2)
        pnt_density /= pnt_density.sum()

        density += pnt_density
        class_indx = point_class_map[int(pt[1]), int(pt[0])].argmax()
        density_class[:, :, class_indx] = density_class[:, :, class_indx] + pnt_density

        mat["inst_map"][np.where(pnt_density > 0)] = i + 1
        mat["type_map"][np.where(pnt_density > 0)] = class_indx

    return mat


def trans_to_coco():
    global segmentation_id
    imgs_files = os.listdir(patch_save_img_path)
    ann = {
        "info": {
            "description": "BRCA Dataset",
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "lymph", "id": 0, "name": "lymph"},
            {"supercategory": "tumor", "id": 1, "name": "tumor"},
            {"supercategory": "stromal", "id": 2, "name": "stromal"},
        ]
    }
    for imgid, image in enumerate(imgs_files):
        img_shape = mmcv.imread(os.path.join(patch_save_img_path, image)).shape
        img_info = {
            "file_name": image,
            "height": img_shape[0],
            "width": img_shape[1],
            "id": imgid
        }
        ann["images"].append(img_info)

        classes_ann = sio.loadmat(os.path.join(coco_save_gt_path, image[:-4] + '.mat'))
        for annid, (centroid, type) in enumerate(zip(classes_ann["inst_centroid"], classes_ann["inst_type"])):
            bbox = [centroid[0] - 2.0, centroid[1] - 2.0, 4.0, 4.0]
            area = 16
            type = int(type)
            annotation = {
                "segmentation": [],
                "area": area,
                "iscrowd": 0,
                "image_id": imgid,
                "bbox": bbox,
                "category_id": type - 1,
                "id": segmentation_id
            }

            ann["annotations"].append(annotation)
            segmentation_id = segmentation_id + 1

    with open(coco_save_json_path, "w") as write_f:
        json.dump(ann, write_f, indent=4, ensure_ascii=False)


def trans_to_patch():
    if mode[0] == "Train":
        win_size = 300
        win_step = 100
        imgs_files = os.listdir(hover_save_img_path)
        for imgid, image_name in enumerate(imgs_files):
            image_path = os.path.join(hover_save_img_path, image_name)
            img = mmcv.imread(image_path)
            gt_mat = sio.loadmat(os.path.join(hover_save_gt_path, image_name[:-4] + ".mat"))
            centroids = gt_mat["inst_centroid"]
            types = gt_mat["inst_type"]

            step_num_raw = (img.shape[1] - win_size) // win_step + 1
            step_num_col = (img.shape[0] - win_size) // win_step + 1

            gt_mat = [[{"inst_centroid": [], "inst_type": []} for j in range(step_num_col)] for i in
                      range(step_num_raw)]

            for i in range(step_num_col):
                for j in range(step_num_raw):
                    img_patch = img[i * win_step:i * win_step + win_size, j * win_step:j * win_step + win_size, :]

                    img_save_path = os.path.join(patch_save_img_path, image_name[:-4] + "_{}{}.png".format(i, j))
                    mmcv.imwrite(img_patch, img_save_path)

                    for centroid, type_ in zip(centroids, types):
                        if j * win_step <= centroid[0] < j * win_step + win_size and \
                                i * win_step <= centroid[1] < i * win_step + win_size:
                            centroid_ = centroid.copy()
                            centroid_[0] -= j * win_step
                            centroid_[1] -= i * win_step
                            gt_mat[i][j]["inst_centroid"].append(centroid_)
                            gt_mat[i][j]["inst_type"].append(type_)

                    gt_mat[i][j]['inst_centroid'] = np.asarray(gt_mat[i][j]['inst_centroid'])
                    gt_mat[i][j]['inst_type'] = np.asarray(gt_mat[i][j]['inst_type'])
                    sio.savemat(os.path.join(patch_save_gt_path, image_name[:-4] + "_{}{}.mat".format(i, j)), gt_mat[i][j])

                    # plt.imshow(img_patch)
                    # plt.scatter(gt_mat[i][j]["inst_centroid"][:, 0], gt_mat[i][j]["inst_centroid"][:, 1], s=1, color=(1, 0, 0))
                    # plt.show()

    else:
        imgs_files = os.listdir(hover_save_img_path)
        for imgid, image_name in enumerate(imgs_files):
            image_path = os.path.join(hover_save_img_path, image_name)
            img = mmcv.imread(image_path)
            gt_mat = sio.loadmat(os.path.join(hover_save_gt_path, image_name[:-4] + ".mat"))
            centroids = gt_mat["inst_centroid"]
            types = gt_mat["inst_type"]

            gt_mat_00 = {"inst_centroid": [], "inst_type": []}
            gt_mat_01 = {"inst_centroid": [], "inst_type": []}
            gt_mat_10 = {"inst_centroid": [], "inst_type": []}
            gt_mat_11 = {"inst_centroid": [], "inst_type": []}

            raw_offset = int(img.shape[1] / 2)
            col_offset = int(img.shape[0] / 2)

            for centroid, type_ in zip(centroids, types):
                if centroid[0] < raw_offset and centroid[1] < col_offset:
                    gt_mat_00['inst_centroid'].append(centroid)
                    gt_mat_00['inst_type'].append(type_)
                elif centroid[0] >= raw_offset and centroid[1] < col_offset:
                    centroid[0] -= raw_offset
                    gt_mat_01['inst_centroid'].append(centroid)
                    gt_mat_01['inst_type'].append(type_)
                elif centroid[0] < raw_offset and centroid[1] >= col_offset:
                    centroid[1] -= col_offset
                    gt_mat_10['inst_centroid'].append(centroid)
                    gt_mat_10['inst_type'].append(type_)
                elif centroid[0] >= raw_offset and centroid[1] >= col_offset:
                    centroid[0] -= raw_offset
                    centroid[1] -= col_offset
                    gt_mat_11['inst_centroid'].append(centroid)
                    gt_mat_11['inst_type'].append(type_)

            gt_mat_00['inst_centroid'] = np.asarray(gt_mat_00['inst_centroid'])
            gt_mat_01['inst_centroid'] = np.asarray(gt_mat_01['inst_centroid'])
            gt_mat_10['inst_centroid'] = np.asarray(gt_mat_10['inst_centroid'])
            gt_mat_11['inst_centroid'] = np.asarray(gt_mat_11['inst_centroid'])

            gt_mat_00['inst_type'] = np.asarray(gt_mat_00['inst_type'])
            gt_mat_01['inst_type'] = np.asarray(gt_mat_01['inst_type'])
            gt_mat_10['inst_type'] = np.asarray(gt_mat_10['inst_type'])
            gt_mat_11['inst_type'] = np.asarray(gt_mat_11['inst_type'])

            for i in range(2):
                for j in range(2):
                    img_patch = img[i * col_offset:(i + 1) * col_offset, j * raw_offset:(j + 1) * raw_offset, :]

                    img_save_path = os.path.join(patch_save_img_path, image_name[:-4] + "_{}{}.png".format(i, j))
                    mmcv.imwrite(img_patch, img_save_path)

            sio.savemat(os.path.join(patch_save_gt_path, image_name[:-4] + "_00.mat"), gt_mat_00)
            sio.savemat(os.path.join(patch_save_gt_path, image_name[:-4] + "_01.mat"), gt_mat_01)
            sio.savemat(os.path.join(patch_save_gt_path, image_name[:-4] + "_10.mat"), gt_mat_10)
            sio.savemat(os.path.join(patch_save_gt_path, image_name[:-4] + "_11.mat"), gt_mat_11)

        # plt.imshow(img_patch)
        # plt.scatter(gt_mat_11["inst_centroid"][:, 0], gt_mat_11["inst_centroid"][:, 1], s=1, color=(1, 0, 0))
        # plt.show()


def valid():
    color = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (100, 100, 100),
             (200, 200, 200)]
    img_path = patch_save_img_path
    file_path = coco_save_json_path
    coco = COCO(file_path)

    list_imgIds = coco.getImgIds(catIds=1)
    img = coco.loadImgs(list_imgIds[5])[0]
    image = io.imread(os.path.join(img_path, img['file_name']))

    for catid in range(5):
        img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catid)
        img_anns = coco.loadAnns(img_annIds)
        centroids = []
        # for i in range(len(img_annIds)):
        #     x, y, w, h = img_anns[i - 1]['bbox']  # 读取边框
        #     image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color[catid], 2)

    #     centroids.extend([x+int(w/2), y+int(h/2)])

    plt.imshow(image)
    coco.showAnns(img_anns)
    plt.show()


if __name__ == '__main__':
    mode = ['Train', 'train']

    coco_save_img_path = "/data2/huangjunjia/coco/COCO_CoNSeP_256/CoNSeP_{}/".format(mode[0])
    coco_save_gt_path = "/data2/huangjunjia/coco/COCO_CoNSeP_256/{}/gt_mat/".format(mode[1])
    coco_save_json_path = "/data2/huangjunjia/coco/COCO_CoNSeP_256/annotations/coco_CoNSeP_{}.json".format(mode[1])
    os.makedirs("/data2/huangjunjia/coco/COCO_CoNSeP_256/annotations", exist_ok=True)

    patch_save_img_path = coco_save_img_path
    patch_save_gt_path = coco_save_gt_path

    hover_save_gt_path = "/data2/huangjunjia/coco/CoNSeP/{}/gt_mat".format(mode[1])
    hover_save_img_path = "/data2/huangjunjia/coco/CoNSeP/CoNSeP_{}".format(mode[0])

    os.makedirs(coco_save_gt_path, exist_ok=True)
    os.makedirs(coco_save_img_path, exist_ok=True)

    classes_max_indx = 3

    trans_to_patch()
    trans_to_coco()

    # valid()

import pycocotools.mask as mask
import os
from skimage import io
import json
import cv2
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from pycocotools.coco import COCO
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation, create_coco_dict
from sahi.utils.file import save_json
from pathlib import Path

def trans2coco():
    global segmentation_id
    categories = ["nolabel", "Inflammatory", "Epithelial", "Spindle-shaped"]
    imgs_files = os.listdir(imgs_path)
    coco = Coco()
    coco.add_category(CocoCategory(id=1, name='Inflammatory'))
    coco.add_category(CocoCategory(id=2, name='Epithelial'))
    coco.add_category(CocoCategory(id=3, name='Spindle-shaped'))

    # images
    for imgid, image in enumerate(imgs_files):
        print("!" * 30)
        print(image)
        img = mmcv.imread(os.path.join(imgs_path, image))
        coco_image = CocoImage(file_name=image, height=img.shape[0], width=img.shape[1])

        classes_ann = sio.loadmat(os.path.join(hover_anns_path, image[:-4] + '.mat'))
        classes_ann["inst_type"][(classes_ann["inst_type"] == 2)] = 9
        classes_ann["inst_type"][(classes_ann["inst_type"] == 3) | (classes_ann["inst_type"] == 4)] = 2
        classes_ann["inst_type"][(classes_ann["inst_type"] == 1) | (classes_ann["inst_type"] == 5)
                                 | (classes_ann["inst_type"] == 6) | (classes_ann["inst_type"] == 7)] = 3
        classes_ann["inst_type"][(classes_ann["inst_type"] == 9)] = 1

        # assert len(classes_ann["inst_type"]) == max(np.unique(classes_ann["inst_map"])), print(
        #     "Vertify, Something wrong happen")
        # print(len(classes_ann["inst_type"]))
        # print(max(np.unique(classes_ann["inst_map"])))

        for inst_num in range(len(classes_ann["inst_type"])):
            # for annid, (centroid, type) in enumerate(zip(classes_ann["inst_centroid"], classes_ann["inst_type"])):
            centroid = classes_ann["inst_centroid"][inst_num]

            if len(CocoAnnotation(
                    bbox=[centroid[0] - 2.0, centroid[1] - 2.0, 4.0, 4.0], #ground_truth_bounding_box.tolist(),
                    # segmentation=[segmentation],
                    category_id=int(classes_ann["inst_type"][inst_num][0]),
                    category_name=categories[int(classes_ann["inst_type"][inst_num][0])]
            ).bbox) == 0:
                print(111111111111111)
                continue
            coco_image.add_annotation(
                CocoAnnotation(
                    bbox=[centroid[0] - 2.0, centroid[1] - 2.0, 4.0, 4.0], # ground_truth_bounding_box.tolist(),
                    # segmentation=[segmentation],
                    category_id=int(classes_ann["inst_type"][inst_num][0]),
                    category_name=categories[int(classes_ann["inst_type"][inst_num][0])]
                )
            )
            if len(coco_image.annotations[-1].bbox) == 0:
                print(1)
        coco.add_image(coco_image)

    save_json(data=coco.json, save_path=json_save_path)


def trans_to_patch(json_original_path, img_orinial_path, json_save_path, patch_save_path, mode):
    from sahi.slicing import slice_coco, slice_image
    from sahi.utils.file import load_json
    coco_dict = load_json(json_original_path)
    coco = Coco.from_coco_dict_or_path(coco_dict)
    sliced_coco_images = []

    for coco_image in coco.images:
        print("Slicing :", coco_image.file_name)
        image_path = os.path.join(img_orinial_path, coco_image.file_name)
        patch_scale_height = coco_image.height // 250
        patch_scale_width = coco_image.width // 250
        slice_height = coco_image.height // patch_scale_height if patch_scale_height != 0 else coco_image.height
        slice_width = coco_image.width // patch_scale_width if patch_scale_width != 0 else coco_image.width

        if mode == "Train":
            overlap_width_ratio = 0.2
            overlap_height_ratio = 0.2
        else:
            overlap_width_ratio = 0.0
            overlap_height_ratio = 0.0
        slice_image_result = slice_image(
            image=image_path,
            coco_annotation_list=coco_image.annotations,
            output_file_name=Path(coco_image.file_name).stem,
            output_dir=patch_save_path,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            min_area_ratio=0.1,
            out_ext=None,
            verbose=True,
            mode=mode,
        )
        sliced_coco_images.extend(slice_image_result.coco_images)

    coco_dict = create_coco_dict(
        sliced_coco_images, coco_dict["categories"], ignore_negative_samples=False
    )

    save_json(coco_dict, json_save_path)


def SAHI_SLICING():
    from sahi.slicing import slice_coco
    from sahi.utils.file import load_json
    from PIL import Image, ImageDraw

    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=json_save_path,
        image_dir=imgs_path,
        output_coco_annotation_file_name="{}_sliced".format(mode[0]),
        ignore_negative_samples=False,
        output_dir="/data2/huangjunjia/coco/CoNSeP/HoverNet_CoNSeP/SAHI_250_8/{}".format(mode[0]),
        slice_height=250,
        slice_width=250,
        overlap_width_ratio=0.0 if mode == "Test" else 0.2,
        overlap_height_ratio=0.0 if mode == "Test" else 0.2,
        min_area_ratio=0.1,
        verbose=True
    )

    coco_dict = load_json(json_save_path)
    f, axarr = plt.subplots(1, 1, figsize=(12, 12))
    img_ind = 0
    img = Image.open(os.path.join(imgs_path, coco_dict["images"][img_ind]["file_name"])).convert('RGBA')
    for idx, ann_ind in enumerate(range(len(coco_dict["annotations"]))):
        # convert coco bbox to pil bbox
        if coco_dict["annotations"][ann_ind]['image_id'] != img_ind + 1:
            break
        xywh = coco_dict["annotations"][ann_ind]["bbox"]
        print(xywh)
        try:
            xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
        except:
            print(1)
        ImageDraw.Draw(img, 'RGBA').rectangle(xyxy, width=5)
    axarr.imshow(img)
    plt.show()
    # pass

def valid(save):
    coco = COCO(save)

    # 统计类别
    cats = coco.loadCats(coco.getCatIds())
    cat_nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))
    for cat_name in cat_nms:
        catId = coco.getCatIds(catNms=cat_name)
        imgId = coco.getImgIds(catIds=catId)
        annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)

        print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))

    return

if __name__ == '__main__':
    mode = ["Train", "train"]
    imgs_path = "Path to CoNSeP 40x/{}/Images/".format(mode[0])

    num_class = 3

    hover_anns_path = "Path to CoNSeP 40x/{}/Labels".format(mode[0])
    json_save_path = "Path to CoNSeP 40x/annotations/coco_CoNSeP_{}_centroid.json".format(
        mode[0])

    trans2coco()
    trans_to_patch(json_save_path, imgs_path,
                   "Path to CoNSeP 40x/SAHI/annotations/CoNSeP_{}.json".format(mode[0]),
                   "Path to CoNSeP 40x/SAHI/{}".format(mode[0]),
                   mode=mode[0])
    # SAHI_SLICING()
    valid("Path to CoNSeP 40x/SAHI/annotations/CoNSeP_{}.json".format(mode[0]))

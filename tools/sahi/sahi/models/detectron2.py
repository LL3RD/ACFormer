# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)


class Detectron2DetectionModel(DetectionModel):
    def check_dependencies(self):
        check_requirements(["torch", "detectron2"])

    def load_model(self):
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog
        from detectron2.engine import DefaultPredictor
        from detectron2.model_zoo import model_zoo

        cfg = get_cfg()

        try:  # try to load from model zoo
            config_file = model_zoo.get_config_file(self.config_path)
            cfg.merge_from_file(config_file)
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_path)
        except Exception as e:  # try to load from local
            print(e)
            if self.config_path is not None:
                from detectron2.projects.deeplab import add_deeplab_config
                from mask2former import add_maskformer2_config
                # from mask2dino import add_maskformer2_config
                from detectron2.data.datasets import register_coco_instances
                add_deeplab_config(cfg)
                add_maskformer2_config(cfg)
                cfg.merge_from_file(self.config_path)
                register_coco_instances("CoNSeP_SAHI_Train", {},
                                        "/mntnfs/med_data2/huangjunjia/dataset/HoverNet_SAHI_250_8/annotations/Train_sliced_coco.json",
                                        "/mntnfs/med_data2/huangjunjia/dataset/HoverNet_SAHI_250_8/Train/")
                register_coco_instances("CoNSeP_SAHI_Test", {},
                                        "/mntnfs/med_data2/huangjunjia/dataset/HoverNet_SAHI_250_8/annotations/Test_sliced_coco.json",
                                        "/mntnfs/med_data2/huangjunjia/dataset/HoverNet_SAHI_250_8/Test/")
            cfg.MODEL.WEIGHTS = self.model_path

        # set model device
        cfg.MODEL.DEVICE = self.device.type
        # set input image size
        if self.image_size is not None:
            cfg.INPUT.MIN_SIZE_TEST = self.image_size
            cfg.INPUT.MAX_SIZE_TEST = self.image_size
        # init predictor
        model = DefaultPredictor(cfg)

        self.model = model

        # detectron2 category mapping
        if self.category_mapping is None:
            try:  # try to parse category names from metadata
                metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
                category_names = metadata.thing_classes
                self.category_names = category_names
                self.category_mapping = {
                    str(ind): category_name for ind, category_name in enumerate(self.category_names)
                }
            except Exception as e:
                logger.warning(e)
                # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
                if cfg.MODEL.META_ARCHITECTURE == "RetinaNet":
                    num_categories = cfg.MODEL.RETINANET.NUM_CLASSES
                else:  # fasterrcnn/maskrcnn etc
                    num_categories = cfg.MODEL.ROI_HEADS.NUM_CLASSES
                self.category_names = [str(category_id) for category_id in range(num_categories)]
                self.category_mapping = {
                    str(ind): category_name for ind, category_name in enumerate(self.category_names)
                }
        else:
            self.category_names = list(self.category_mapping.values())

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        # if isinstance(image, np.ndarray) and self.model.input_format == "BGR":
        #     # convert RGB image to BGR format
        #     image = image[:, :, ::-1]

        image = image[:, :, ::-1]
        prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        num_categories = len(self.category_mapping)
        return num_categories

    def convert_original_predictions_cell(
            self,
            shift_amount: Optional[List[int]] = [0, 0],
            full_shape: Optional[List[int]] = None,
            img_shape=None,
    ):
        """
        Converts original predictions of the detection model to a list of
        prediction.ObjectPrediction object. Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        self._create_object_prediction_list_from_original_predictions_remove_edge_boxes(
            shift_amount_list=shift_amount,
            full_shape_list=full_shape,
            img_shape=img_shape
        )
        if self.category_remapping:
            self._apply_category_remapping()

    def _create_object_prediction_list_from_original_predictions_remove_edge_boxes(
            self,
            shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
            full_shape_list: Optional[List[List[int]]] = None,
            img_shape=None, # (w, h)
    ):
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        # parse boxes, masks, scores, category_ids from predictions
        boxes = original_predictions["instances"].pred_boxes.tensor.tolist()
        scores = original_predictions["instances"].scores.tolist()
        category_ids = original_predictions["instances"].pred_classes.tolist()

        # check if predictions contain mask
        try:
            masks = original_predictions["instances"].pred_masks.tolist()
        except AttributeError:
            masks = None

        # create object_prediction_list
        object_prediction_list_per_image = []
        object_prediction_list = []

        # detectron2 DefaultPredictor supports single image
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)

        # plt.imshow(np.ones((250, 250, 3)))

        for ind in range(len(boxes)):
            score = scores[ind]
            if score < self.confidence_threshold:
                continue


            # Visual
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 1, 1)
            #
            # plt.imshow(np.ones((500, 500, 3)))
            # for box in boxes:
            #     x1, y1, x2, y2 = box
            #     rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            #     ax.add_patch(rect)
            # plt.show()

            # remove edge boxes
            # det_box = boxes[ind]  # (w1, h1, w2, h2)
            # if abs(img_shape[0] - det_box[2]) <= 5 and shift_amount[0] + img_shape[0] != full_shape[0]:
            #     continue
            # if abs(img_shape[1] - det_box[3]) <= 5 and shift_amount[1] + img_shape[1] != full_shape[1]:
            #     continue
            # if abs(det_box[0] - 0) <= 5 and shift_amount[0] != 0:
            #     continue
            # if abs(det_box[1] - 0) <= 5 and shift_amount[1] != 0:
            #     continue

            # x1, y1, x2, y2 = det_box
            # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            # ax.add_patch(rect)


            category_id = category_ids[ind]

            if masks is None:
                bbox = boxes[ind]
                mask = None
            else:
                mask = np.array(masks[ind])

                # check if mask is valid
                # https://github.com/obss/sahi/issues/389
                if get_bbox_from_bool_mask(mask) is None:
                    continue
                else:
                    bbox = None

            object_prediction = ObjectPrediction(
                bbox=bbox,
                bool_mask=mask,
                category_id=category_id,
                category_name=self.category_mapping[str(category_id)],
                shift_amount=shift_amount,
                score=score,
                full_shape=full_shape,
            )
            object_prediction_list.append(object_prediction)

        # plt.show()
        # detectron2 DefaultPredictor supports single image
        object_prediction_list_per_image = [object_prediction_list]

        self._object_prediction_list_per_image = object_prediction_list_per_image

    def _create_object_prediction_list_from_original_predictions(
            self,
            shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
            full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        # parse boxes, masks, scores, category_ids from predictions
        boxes = original_predictions["instances"].pred_boxes.tensor.tolist()
        scores = original_predictions["instances"].scores.tolist()
        category_ids = original_predictions["instances"].pred_classes.tolist()

        # check if predictions contain mask
        try:
            masks = original_predictions["instances"].pred_masks.tolist()
        except AttributeError:
            masks = None

        # create object_prediction_list
        object_prediction_list_per_image = []
        object_prediction_list = []

        # detectron2 DefaultPredictor supports single image
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        for ind in range(len(boxes)):
            score = scores[ind]
            if score < self.confidence_threshold:
                continue

            category_id = category_ids[ind]

            if masks is None:
                bbox = boxes[ind]
                mask = None
            else:
                mask = np.array(masks[ind])

                # check if mask is valid
                # https://github.com/obss/sahi/issues/389
                if get_bbox_from_bool_mask(mask) is None:
                    continue
                else:
                    bbox = None

            object_prediction = ObjectPrediction(
                bbox=bbox,
                bool_mask=mask,
                category_id=category_id,
                category_name=self.category_mapping[str(category_id)],
                shift_amount=shift_amount,
                score=score,
                full_shape=full_shape,
            )
            object_prediction_list.append(object_prediction)

        # detectron2 DefaultPredictor supports single image
        object_prediction_list_per_image = [object_prediction_list]

        self._object_prediction_list_per_image = object_prediction_list_per_image

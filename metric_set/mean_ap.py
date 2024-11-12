# author: Lekang Wen
# date: 2021/11/12

import contextlib
import io
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import _input_validator, _validate_iou_type_arg
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import (
    _FASTER_COCO_EVAL_AVAILABLE,
    _PYCOCOTOOLS_AVAILABLE,
    _TORCHVISION_GREATER_EQUAL_0_8,
)
from typing_extensions import Literal

from .cocoeval import COCOeval, Params


class MeanAveragePrecision(MeanAveragePrecision, Metric):
    r"""Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)`_ for object detection predictions.

    .. math::
        \text{mAP} = \frac{1}{n} \sum_{i=1}^{n} AP_i

    where :math:`AP_i` is the average precision for class :math:`i` and :math:`n` is the number of classes. The average
    precision is defined as the area under the precision-recall curve. For object detection the recall and precision are
    defined based on the intersection of union (IoU) between the predicted bounding boxes and the ground truth bounding
    boxes e.g. if two boxes have an IoU > t (with t being some threshold) they are considered a match and therefore
    considered a true positive. The precision is then defined as the number of true positives divided by the number of
    all detected boxes and the recall is defined as the number of true positives divided by the number of all ground
    boxes.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes``
          detection boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates, but can be changed
          using the ``box_format`` parameter. Only required when `iou_type="bbox"`.
        - ``scores`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes)`` containing detection scores for the
          boxes.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed detection
          classes for the boxes.
        - ``masks`` (:class:`~torch.Tensor`): boolean tensor of shape ``(num_boxes, image_height, image_width)``
          containing boolean masks. Only required when `iou_type="segm"`.

    - ``target`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground
          truth boxes of the format specified in the constructor. only required when `iou_type="bbox"`.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed ground truth
          classes for the boxes.
        - ``masks`` (:class:`~torch.Tensor`): boolean tensor of shape ``(num_boxes, image_height, image_width)``
          containing boolean masks. Only required when `iou_type="segm"`.
        - ``iscrowd`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0/1 values indicating
          whether the bounding box/masks indicate a crowd of objects. Value is optional, and if not provided it will
          automatically be set to 0.
        - ``area`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes)`` containing the area of the object.
          Value is optional, and if not provided will be automatically calculated based on the bounding box/masks
          provided. Only affects which samples contribute to the `map_small`, `map_medium`, `map_large` values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``map_dict``: A dictionary containing the following key-values:

        - map: (:class:`~torch.Tensor`), global mean average precision which by default is defined as mAP50-95 e.g. the
          mean average precision for IoU thresholds 0.50, 0.55, 0.60, ..., 0.95 averaged over all classes and areas. If
          the IoU thresholds are changed this value will be calculated with the new thresholds.
        - map_small: (:class:`~torch.Tensor`), mean average precision for small objects (area < 32^2 pixels)
        - map_medium:(:class:`~torch.Tensor`), mean average precision for medium objects (32^2  pixels < area < 96^2
          pixels)
        - map_large: (:class:`~torch.Tensor`), mean average precision for large objects (area > 96^2 pixels)
        - mar_{mdt[0]}: (:class:`~torch.Tensor`), mean average recall for `max_detection_thresholds[0]` (default 1)
          detection per image
        - mar_{mdt[1]}: (:class:`~torch.Tensor`), mean average recall for `max_detection_thresholds[1]` (default 10)
          detection per image
        - mar_{mdt[1]}: (:class:`~torch.Tensor`), mean average recall for `max_detection_thresholds[2]` (default 100)
          detection per image
        - mar_small: (:class:`~torch.Tensor`), mean average recall for small objects (area < 32^2  pixels)
        - mar_medium: (:class:`~torch.Tensor`), mean average recall for medium objects (32^2 pixels < area < 96^2
          pixels)
        - mar_large: (:class:`~torch.Tensor`), mean average recall for large objects (area > 96^2  pixels)
        - map_50: (:class:`~torch.Tensor`) (-1 if 0.5 not in the list of iou thresholds), mean average precision at
          IoU=0.50
        - map_75: (:class:`~torch.Tensor`) (-1 if 0.75 not in the list of iou thresholds), mean average precision at
          IoU=0.75
        - map_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average precision per
          observed class
        - mar_{mdt[2]}_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average recall for
          `max_detection_thresholds[2]` (default 100) detections per image per observed class
        - classes (:class:`~torch.Tensor`), list of all observed classes

    For an example on how to use this metric check the `torchmetrics mAP example`_.

    .. attention::
        The ``map`` score is calculated with @[ IoU=self.iou_thresholds | area=all | max_dets=max_detection_thresholds ]
        e.g. the mean average precision for IoU thresholds 0.50, 0.55, 0.60, ..., 0.95 averaged over all classes and
        all areas and all max detections per image. If the IoU thresholds are changed this value will be calculated with
        the new thresholds.
        **Caution:** If the initialization parameters are changed, dictionary keys for mAR can change as well.

    .. important::
        This metric supports, at the moment, two different backends for the evaluation. The default backend is
        ``"pycocotools"``, which either require the official `pycocotools`_ implementation or this
        `fork of pycocotools`_ to be installed. We recommend using the fork as it is better maintained and easily
        available to install via pip: `pip install pycocotools`. It is also this fork that will be installed if you
        install ``torchmetrics[detection]``. The second backend is the `faster-coco-eval`_ implementation, which can be
        installed with ``pip install faster-coco-eval``. This implementation is a maintained open-source implementation
        that is faster and corrects certain corner cases that the official implementation has. Our own testing has shown
        that the results are identical to the official implementation. Regardless of the backend we also require you to
        have `torchvision` version 0.8.0 or newer installed. Please install with ``pip install torchvision>=0.8`` or
        ``pip install torchmetrics[detection]``.

    Args:
        box_format:
            Input format of given boxes. Supported formats are:

                - 'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
                - 'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being
                  width and height. This is the default format used by pycoco and all input formats will be converted
                  to this.
                - 'cxcywh': boxes are represented via centre, width and height, cx, cy being center of box, w, h being
                  width and height.

        iou_type:
            Type of input (either masks or bounding-boxes) used for computing IOU. Supported IOU types are
            ``"bbox"`` or ``"segm"`` or both as a tuple.
        iou_thresholds:
            IoU thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0.5,...,0.95]``
            with step ``0.05``. Else provide a list of floats.
        rec_thresholds:
            Recall thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0,...,1]``
            with step ``0.01``. Else provide a list of floats.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use thresholds ``[1, 10, 100]``.
            Else, please provide a list of ints of length 3, which is the only supported length by both backends.
        class_metrics:
            Option to enable per-class metrics for mAP and mAR_100. Has a performance impact that scales linearly with
            the number of classes in the dataset.
        extended_summary:
            Option to enable extended summary with additional metrics including IOU, precision and recall. The output
            dictionary will contain the following extra key-values:

                - ``ious``: a dictionary containing the IoU values for every image/class combination e.g.
                  ``ious[(0,0)]`` would contain the IoU for image 0 and class 0. Each value is a tensor with shape
                  ``(n,m)`` where ``n`` is the number of detections and ``m`` is the number of ground truth boxes for
                  that image/class combination.
                - ``precision``: a tensor of shape ``(TxRxKxAxM)`` containing the precision values. Here ``T`` is the
                  number of IoU thresholds, ``R`` is the number of recall thresholds, ``K`` is the number of classes,
                  ``A`` is the number of areas and ``M`` is the number of max detections per image.
                - ``recall``: a tensor of shape ``(TxKxAxM)`` containing the recall values. Here ``T`` is the number of
                  IoU thresholds, ``K`` is the number of classes, ``A`` is the number of areas and ``M`` is the number
                  of max detections per image.
                - ``scores``: a tensor of shape ``(TxRxKxAxM)`` containing the confidence scores.  Here ``T`` is the
                  number of IoU thresholds, ``R`` is the number of recall thresholds, ``K`` is the number of classes,
                  ``A`` is the number of areas and ``M`` is the number of max detections per image.

        average:
            Method for averaging scores over labels. Choose between "``"macro"`` and ``"micro"``.
        backend:
            Backend to use for the evaluation. Choose between ``"pycocotools"`` and ``"faster_coco_eval"``.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``pycocotools`` is not installed
        ModuleNotFoundError:
            If ``torchvision`` is not installed or version installed is lower than 0.8.0
        ValueError:
            If ``box_format`` is not one of ``"xyxy"``, ``"xywh"`` or ``"cxcywh"``
        ValueError:
            If ``iou_type`` is not one of ``"bbox"`` or ``"segm"``
        ValueError:
            If ``iou_thresholds`` is not None or a list of floats
        ValueError:
            If ``rec_thresholds`` is not None or a list of floats
        ValueError:
            If ``max_detection_thresholds`` is not None or a list of ints
        ValueError:
            If ``class_metrics`` is not a boolean

    Example::

        Basic example for when `iou_type="bbox"`. In this case the ``boxes`` key is required in the input dictionaries,
        in addition to the ``scores`` and ``labels`` keys.

        >>> from torch import tensor
        >>> from torchmetrics.detection import MeanAveragePrecision
        >>> preds = [
        ...   dict(
        ...     boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
        ...     scores=tensor([0.536]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> target = [
        ...   dict(
        ...     boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> metric = MeanAveragePrecision(iou_type="bbox")
        >>> metric.update(preds, target)
        >>> from pprint import pprint
        >>> pprint(metric.compute)
        {'classes': tensor(0, dtype=torch.int32),
         'map': tensor(0.6000),
         'map_50': tensor(1.),
         'map_75': tensor(1.),
         'map_large': tensor(0.6000),
         'map_medium': tensor(-1.),
         'map_per_class': tensor(-1.),
         'map_small': tensor(-1.),
         'mar_1': tensor(0.6000),
         'mar_10': tensor(0.6000),
         'mar_100': tensor(0.6000),
         'mar_100_per_class': tensor(-1.),
         'mar_large': tensor(0.6000),
         'mar_medium': tensor(-1.),
         'mar_small': tensor(-1.)}

    Example::

        Basic example for when `iou_type="segm"`. In this case the ``masks`` key is required in the input dictionaries,
        in addition to the ``scores`` and ``labels`` keys.

        >>> from torch import tensor
        >>> from torchmetrics.detection import MeanAveragePrecision
        >>> mask_pred = [
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 0, 0, 0],
        ... ]
        >>> mask_tgt = [
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 1, 0, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 1, 0, 0],
        ...   [0, 0, 0, 0, 0],
        ... ]
        >>> preds = [
        ...   dict(
        ...     masks=tensor([mask_pred], dtype=torch.bool),
        ...     scores=tensor([0.536]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> target = [
        ...   dict(
        ...     masks=tensor([mask_tgt], dtype=torch.bool),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> metric = MeanAveragePrecision(iou_type="segm")
        >>> metric.update(preds, target)
        >>> from pprint import pprint
        >>> pprint(metric.compute)
        {'classes': tensor(0, dtype=torch.int32),
         'map': tensor(0.2000),
         'map_50': tensor(1.),
         'map_75': tensor(0.),
         'map_large': tensor(-1.),
         'map_medium': tensor(-1.),
         'map_per_class': tensor(-1.),
         'map_small': tensor(0.2000),
         'mar_1': tensor(0.2000),
         'mar_10': tensor(0.2000),
         'mar_100': tensor(0.2000),
         'mar_100_per_class': tensor(-1.),
         'mar_large': tensor(-1.),
         'mar_medium': tensor(-1.),
         'mar_small': tensor(0.2000)}

    """

    detection_box: List[Tensor] = []
    detection_mask: List[Tensor] = []
    detection_scores: List[Tensor] = []
    detection_labels: List[Tensor] = []
    groundtruth_box: List[Tensor] = []
    groundtruth_mask: List[Tensor] = []
    groundtruth_labels: List[Tensor] = []
    groundtruth_crowds: List[Tensor] = []
    groundtruth_area: List[Tensor] = []

    conf_mat: Tensor

    def __init__(
            self,
            box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
            iou_type: Union[Literal["bbox", "segm"], Tuple[str]] = "bbox",
            iou_thresholds: Optional[List[float]] = None,
            rec_thresholds: Optional[List[float]] = None,
            max_detection_thresholds: Optional[List[int]] = None,
            class_metrics: bool = False,
            extended_summary: bool = False,
            average: Literal["macro", "micro"] = "macro",
            backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",

            # for num_classes
            num_classes: int = 1,
            # for log
            prefix: str = "",
            keyword: Optional[str] = None,

            **kwargs: Any,
    ) -> None:
        Metric.__init__(self, **kwargs)

        if not (_PYCOCOTOOLS_AVAILABLE or _FASTER_COCO_EVAL_AVAILABLE):
            raise ModuleNotFoundError(
                "`MAP` metric requires that `pycocotools` or `faster-coco-eval` installed."
                " Please install with `pip install pycocotools` or `pip install faster-coco-eval` or"
                " `pip install torchmetrics[detection]`."
            )
        if not _TORCHVISION_GREATER_EQUAL_0_8:
            raise ModuleNotFoundError(
                "`MeanAveragePrecision` metric requires that `torchvision` version 0.8.0 or newer is installed."
                " Please install with `pip install torchvision>=0.8` or `pip install torchmetrics[detection]`."
            )

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")
        self.box_format = box_format

        self.iou_type = _validate_iou_type_arg(iou_type)

        if iou_thresholds is not None and not isinstance(iou_thresholds, list):
            raise ValueError(
                f"Expected argument `iou_thresholds` to either be `None` or a list of floats but got {iou_thresholds}"
            )
        self.iou_thresholds = iou_thresholds or torch.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()

        if rec_thresholds is not None and not isinstance(rec_thresholds, list):
            raise ValueError(
                f"Expected argument `rec_thresholds` to either be `None` or a list of floats but got {rec_thresholds}"
            )
        self.rec_thresholds = rec_thresholds or torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()

        if max_detection_thresholds is not None and not isinstance(max_detection_thresholds, list):
            raise ValueError(
                f"Expected argument `max_detection_thresholds` to either be `None` or a list of ints"
                f" but got {max_detection_thresholds}"
            )
        if max_detection_thresholds is not None and len(max_detection_thresholds) != 3:
            raise ValueError(
                "When providing a list of max detection thresholds it should have length 3."
                f" Got value {len(max_detection_thresholds)}"
            )
        max_det_threshold, _ = torch.sort(torch.tensor(max_detection_thresholds or [1, 10, 100], dtype=torch.int))
        self.max_detection_thresholds = max_det_threshold.tolist()

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics

        if not isinstance(extended_summary, bool):
            raise ValueError("Expected argument `extended_summary` to be a boolean")
        self.extended_summary = extended_summary

        if average not in ("macro", "micro"):
            raise ValueError(f"Expected argument `average` to be one of ('macro', 'micro') but got {average}")
        self.average = average

        if backend not in ("pycocotools", "faster_coco_eval"):
            raise ValueError(
                f"Expected argument `backend` to be one of ('pycocotools', 'faster_coco_eval') but got {backend}"
            )
        self.backend = backend

        self.params = Params(iouType="bbox")
        self.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
        self.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
        self.params.maxDets = self.max_detection_thresholds

        T = len(self.params.iouThrs)
        K = num_classes  # 对应的是 num_classes
        A = len(self.params.areaRngLbl)  # 与pycocotools中的areaRng对应
        M = len(self.params.maxDets)

        self.conf_mat_shape = [T, K, A, M, 3]

        self.num_classes = num_classes
        self.prefix = prefix
        self.keyword = keyword

        self.add_state("conf_mat", default=torch.zeros(self.conf_mat_shape), dist_reduce_fx="sum")

    @property
    def cocoeval(self) -> object:
        """Returns the coco eval module for the given backend, done in this way to make metric picklable."""
        return COCOeval

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:
        """Update metric state.

        Raises:
            ValueError:
                If ``preds`` is not of type (:class:`~List[Dict[str, Tensor]]`)
            ValueError:
                If ``target`` is not of type ``List[Dict[str, Tensor]]``
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1

        """
        _input_validator(preds, target, iou_type=self.iou_type)  # type: ignore[arg-type]

        for item in preds:
            bbox_detection, mask_detection = self._get_safe_item_values(item, warn=self.warn_on_many_detections)
            if bbox_detection is not None:
                self.detection_box.append(bbox_detection)
            if mask_detection is not None:
                self.detection_mask.append(mask_detection)  # type: ignore[arg-type]
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            bbox_groundtruth, mask_groundtruth = self._get_safe_item_values(item)
            if bbox_groundtruth is not None:
                self.groundtruth_box.append(bbox_groundtruth)
            if mask_groundtruth is not None:
                self.groundtruth_mask.append(mask_groundtruth)  # type: ignore[arg-type]
            self.groundtruth_labels.append(item["labels"])
            self.groundtruth_crowds.append(item.get("iscrowd", torch.zeros_like(item["labels"])))
            self.groundtruth_area.append(item.get("area", torch.zeros_like(item["labels"])))

        # 添加
        device = self.conf_mat.device
        add_conf_mat = self.compute_conf_mat().to(device)
        self.conf_mat += add_conf_mat

        # 清空列表
        self.detection_box.clear()
        self.detection_mask.clear()
        self.detection_labels.clear()
        self.detection_scores.clear()
        self.groundtruth_box.clear()
        self.groundtruth_mask.clear()
        self.groundtruth_labels.clear()
        self.groundtruth_crowds.clear()
        self.groundtruth_area.clear()

    def compute_conf_mat(self) -> Tensor:
        """Computes the metric."""
        coco_preds, coco_target = self._get_coco_datasets(average=self.average)

        with contextlib.redirect_stdout(io.StringIO()):
            for i_type in self.iou_type:
                if len(self.iou_type) > 1:
                    # the area calculation is different for bbox and segm and therefore to get the small, medium and
                    # large values correct we need to dynamically change the area attribute of the annotations
                    for anno in coco_preds.dataset["annotations"]:
                        anno["area"] = anno[f"area_{i_type}"]

                if len(coco_preds.imgs) == 0 or len(coco_target.imgs) == 0:
                    conf_mat = torch.zeros(*self.conf_mat_shape)
                else:
                    coco_eval: COCOeval = self.cocoeval(coco_target, coco_preds,
                                                        iouType=i_type)  # type: ignore[operator]
                    coco_eval.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
                    coco_eval.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
                    coco_eval.params.maxDets = self.max_detection_thresholds

                    coco_eval.evaluate()
                    conf_mat = torch.from_numpy(coco_eval.accumulate_conf_mat())

        return conf_mat

    def compute(self):
        out = self._coco_stats_to_tensor_dict(self.summarize(), prefix=self.prefix)
        if self.keyword is not None:
            out = out.get(self.prefix + self.keyword) or out
        return out

    def summarize(self):
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxKxAxM]
                tp = self.conf_mat[..., 0]  # self.eval['recall']
                fp = self.conf_mat[..., 1]  # self.eval['recall']

                nu = tp
                de = tp + fp
                s = torch.where(de != 0, nu / de, -1)

                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[..., aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                tp = self.conf_mat[..., 0]  # self.eval['recall']
                fn = self.conf_mat[..., 2]  # self.eval['recall']

                nu = tp
                de = tp + fn
                s = torch.where(de != 0, nu / de, -1)

                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[..., aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = s[s > -1].mean()
            return mean_s

        stats = torch.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats

    def _coco_stats_to_tensor_dict(self, stats: List[float], prefix: str) -> Dict[str, Tensor]:
        """Converts the output of stats from COCOeval to a dict of tensors."""
        mdt = self.max_detection_thresholds
        return {
            f"{prefix}map": torch.tensor([stats[0]], dtype=torch.float32),
            f"{prefix}map_50": torch.tensor([stats[1]], dtype=torch.float32),
            f"{prefix}map_75": torch.tensor([stats[2]], dtype=torch.float32),
            f"{prefix}map_small": torch.tensor([stats[3]], dtype=torch.float32),
            f"{prefix}map_medium": torch.tensor([stats[4]], dtype=torch.float32),
            f"{prefix}map_large": torch.tensor([stats[5]], dtype=torch.float32),
            f"{prefix}mar_{mdt[0]}": torch.tensor([stats[6]], dtype=torch.float32),
            f"{prefix}mar_{mdt[1]}": torch.tensor([stats[7]], dtype=torch.float32),
            f"{prefix}mar_{mdt[2]}": torch.tensor([stats[8]], dtype=torch.float32),
            f"{prefix}mar_small": torch.tensor([stats[9]], dtype=torch.float32),
            f"{prefix}mar_medium": torch.tensor([stats[10]], dtype=torch.float32),
            f"{prefix}mar_large": torch.tensor([stats[11]], dtype=torch.float32),
        }

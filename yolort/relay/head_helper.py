# Copyright (c) 2022, yolort team. All rights reserved.

import random

import torch
from torch import nn, Tensor


class FakeYOLO(nn.Module):
    """
    Fake YOLO used to export an ONNX models for ONNX Runtime and OpenVINO.
    """

    def __init__(
        self,
        yolo_stem: nn.Module,
        size: int = 640,
        iou_thresh: float = 0.45,
        score_thresh: float = 0.35,
        detections_per_img: int = 100,
    ):
        super().__init__()

        self.yolo_stem = yolo_stem
        self.post_process = FakePostProcess(
            size,
            iou_thresh=iou_thresh,
            score_thresh=score_thresh,
            detections_per_img=detections_per_img,
        )

    def forward(self, x):
        x = self.yolo_stem(x)[0]
        out = self.post_process(x)
        return out


class FakePostProcess(nn.Module):
    """
    Fake PostProcess used to export an ONNX models containing NMS for ONNX Runtime and OpenVINO.

    Args:
        size (int): width and height of the images.
        iou_thresh (float, optional): NMS threshold used for postprocessing the detections.
            Default to 0.45
        score_thresh (float, optional): Score threshold used for postprocessing the detections.
            Default to 0.35
        detections_per_img (int, optional): Number of best detections to keep after NMS.
            Default to 100
    """

    def __init__(
        self,
        size: int,
        iou_thresh: float = 0.45,
        score_thresh: float = 0.35,
        detections_per_img: int = 100,
    ):
        super().__init__()
        self.detections_per_img = detections_per_img
        self.iou_threshold = iou_thresh
        self.score_threshold = score_thresh
        self.size = size
        self.convert_matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]

    def forward(self, x: Tensor):
        device = x.device()
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor(self.convert_matrix, dtype=torch.float32, device=device)
        box @= convert_matrix
        obj_scores, obj_classes = score.max(2, keepdim=True)
        dis = obj_classes.float() * self.size
        rel_boxes = box + dis
        obj_scores_t = obj_scores.transpose(1, 2).contiguous()

        detections_per_img = torch.tensor([self.detections_per_img]).to(device)
        iou_threshold = torch.tensor([self.iou_thresh]).to(device)
        score_threshold = torch.tensor([self.score_thresh]).to(device)
        selected_indices = NonMaxSupressionOp.apply(
            rel_boxes,
            obj_scores_t,
            detections_per_img,
            iou_threshold,
            score_threshold,
        )
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        res_boxes = box[X, Y, :]
        res_classes = obj_classes[X, Y, :]
        res_scores = obj_scores[X, Y, :]
        X = X.unsqueeze(1)
        X = X.float()
        res_classes = res_classes.float()
        out = torch.concat([X, res_boxes, res_classes, res_scores], 1)
        return out


class NonMaxSupressionOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: Tensor = torch.tensor([100]),
        iou_threshold: Tensor = torch.tensor([0.45]),
        score_threshold: Tensor = torch.tensor([0.35]),
    ):
        """
        Args:
            boxes (Tensor): An input tensor with shape [num_batches, spatial_dimension, 4].
                have been multiplied original size here.
            scores (Tensor): An input tensor with shape [num_batches, num_classes, spatial_dimension].
                only one class score here.
            max_output_boxes_per_class (Tensor, optional): Integer representing the maximum number of
                boxes to be selected per batch per class. It is a scalar.
            iou_threshold (Tensor, optional): Float representing the threshold for deciding whether
                boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
            score_threshold (Tensor, optional): Float representing the threshold for deciding when to
                remove boxes based on score. It is a scalar.
        """
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype=torch.int64)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: Tensor,
        iou_threshold: Tensor,
        score_threshold: Tensor,
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )


class EfficientNMSOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        background_class: int = -1,
        box_coding: int = 0,
        iou_threshold: float = 0.45,
        max_output_boxes: int = 100,
        plugin_version: str = "1",
        score_activation: int = 0,
        score_threshold: float = 0.35,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1))
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes))

        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes: Tensor,
        scores: Tensor,
        background_class: int = -1,
        box_coding: int = 0,
        iou_threshold: float = 0.45,
        max_output_boxes: int = 100,
        plugin_version: str = "1",
        score_activation: int = 0,
        score_threshold: float = 0.35,
    ):

        return g.op(
            "TRT::EfficientNMS_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4,
        )

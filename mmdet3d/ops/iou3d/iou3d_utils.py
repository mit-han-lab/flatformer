import torch

from . import iou3d_cuda


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def multi_class_nms_gpu(
    num_classes,
    boxes_for_nms,
    boxes,
    scores,
    labels,
    threshs,
    pre_max_sizes=None,
    post_max_sizes=None,
):
    """Class-separate nms function with gpu implementation.

    Args:
        num_classes (int): The number of classes
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        labels (torch.Tensor): Class predictions for of shape [N]
        threshs (List[int]): Thresholds for all classes (size: num_classes).
        pre_maxsizes (List[int]): Max size of boxes before nms. Default: None (size: num_classes).
        post_maxsizes (List[int]): Max size of boxes after nms. Default: None (size: num_classes).

    Returns:
        torch.Tensor: Indexes after nms.
    """
    assert isinstance(pre_max_sizes, list)
    assert isinstance(post_max_sizes, list)
    pred_boxes, pred_scores, pred_labels = [], [], []
    # class separate
    for k in range(num_classes):
        pre_max_size = pre_max_sizes[k]
        post_max_size = post_max_sizes[k]
        thresh = threshs[k]
        cur_boxes = boxes[labels == k]
        cur_boxes_for_nms = boxes_for_nms[labels == k]
        cur_scores = scores[labels == k]
        cur_labels = labels[labels == k]
        if len(cur_boxes) == 0:
            continue
        cur_order = cur_scores.sort(0, descending=True)[1]
        if pre_max_size is not None:
            cur_order = cur_order[:pre_max_size]
        cur_boxes = cur_boxes[cur_order].contiguous()
        cur_boxes_for_nms = cur_boxes_for_nms[cur_order].contiguous()
        cur_scores = cur_scores[cur_order].contiguous()
        cur_labels = cur_labels[cur_order].contiguous()

        cur_keep = torch.zeros(cur_boxes.size(0), dtype=torch.long)
        num_out = iou3d_cuda.nms_gpu(
            cur_boxes_for_nms, cur_keep, thresh, cur_boxes.device.index
        )
        cur_keep = cur_order[cur_keep[:num_out].cuda(cur_boxes.device)].contiguous()
        if post_max_size is not None:
            cur_keep = cur_keep[:post_max_size]
        pred_boxes.append(cur_boxes[cur_keep])
        pred_scores.append(cur_scores[cur_keep])
        pred_labels.append(cur_labels[cur_keep])

    pred_boxes = torch.cat(pred_boxes, 0)
    pred_scores = torch.cat(pred_scores, 0)
    pred_labels = torch.cat(pred_labels, 0)
    return pred_boxes, pred_scores, pred_labels


def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh, boxes.device.index)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()

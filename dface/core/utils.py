import numpy as np
import torch

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = torch.max(box[0], boxes[:, 0])
    yy1 = torch.max(box[1], boxes[:, 1])
    xx2 = torch.max(box[2], boxes[:, 2])
    yy2 = torch.max(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = torch.max(0, xx2 - xx1 + 1)
    h = torch.max(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    #ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

def nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: []
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
       
    Parameters
    ----------
    dets : torch.Tensor Nx5
        N bounding boxes with each [left, top, right, bottom, confidence]
    thresh : [type]
        [description]
    mode : str, optional
        [description] (the default is "Union", which [default_description])
    
    Returns
    -------
    [type]
        [description]
    """


    left = dets[:, 0]
    top = dets[:, 1]
    right = dets[:, 2]
    bottom = dets[:, 3]
    confidence = dets[:, 4]

    areas = (right - left + 1) * (bottom - top + 1)
    order = confidence.argsort().flip(0)

    keep = []
    while order.nelement() > 1:
        i = order[0]
        
        # print(f'Order len: {order.shape}')
        keep.append(i.item())
        xx1 = torch.max(left[i], left[order[1:]])
        yy1 = torch.max(top[i], top[order[1:]])
        xx2 = torch.min(right[i], right[order[1:]])
        yy2 = torch.min(bottom[i], bottom[order[1:]])

        w = (xx2 - xx1 + 1).clamp_(min=0)
        h = (yy2 - yy1 + 1).clamp_(min=0)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / torch.min(areas[i], areas[order[1:]])

        inds = (ovr <= thresh).nonzero().squeeze()
        # print(f'order : {order} inds: {inds}')
        order = order[inds + 1]
        # print(f'After Order len: {order.shape}, {order.nelement()}')



    return keep





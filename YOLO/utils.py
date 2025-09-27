import torch
import torchvision
import supervision as sv

def process_detections(result, orig_img, transform_params, conf_threshold=0.4, iou_threshold=0.55):
    """
    Process OpenVINO YOLO output into Supervision Detections.
    
    Args:
        result (np.ndarray): OpenVINO model output (1, C, N)
        orig_img (np.ndarray): Original image
        transform_params (dict): {'scale': float, 'pad_w': int, 'pad_h': int}
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        sv.Detections
    """

    # Convert output to tensor and reshape
    pred = torch.from_numpy(result).squeeze(0)  # (C, N)
    pred = pred.permute(1, 0)                   # (N, C)

    # Split boxes and class scores
    boxes = pred[:, :4]       # x, y, w, h
    scores = pred[:, 4:]      # class confidences

    # Best class per prediction
    conf, cls = scores.max(1, keepdim=True)
    pred = torch.cat((boxes, conf, cls.float()), 1)  # (N, 6)

    # Apply NMS
    det = non_max_suppression(pred, conf_thres=conf_threshold, iou_thres=iou_threshold)

    if det.numel() == 0:
        return sv.Detections.empty()

    # Rescale boxes from model input back to original image
    scale = transform_params['scale']
    pad_w = transform_params['pad_w']
    pad_h = transform_params['pad_h']
    
    det[:, [0, 2]] -= pad_w
    det[:, [1, 3]] -= pad_h
    det[:, :4] /= scale

    # Clip to image bounds
    h0, w0 = orig_img.shape[:2]
    det[:, [0, 2]] = det[:, [0, 2]].clamp(0, w0)
    det[:, [1, 3]] = det[:, [1, 3]].clamp(0, h0)

    # Convert to Supervision Detections
    xyxy = det[:, :4].cpu().numpy()
    confidence = det[:, 4].cpu().numpy()
    class_ids = det[:, 5].cpu().numpy().astype(int)

    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_ids)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
    """
    Performs Non-Maximum Suppression (NMS) on YOLO outputs.
    
    Args:
        prediction (Tensor): (N, 6) tensor [x, y, w, h, conf, cls]
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold for NMS
        classes (list[int] | None): filter by class IDs
        agnostic (bool): class-agnostic NMS
        max_det (int): max detections
    
    Returns:
        Tensor: (n, 6) with [x1, y1, x2, y2, conf, cls]
    """

    if prediction.numel() == 0:
        return torch.zeros((0,6), device=prediction.device)

    # Filter low confidence
    mask = prediction[:, 4] > conf_thres
    prediction = prediction[mask]
    if prediction.shape[0] == 0:
        return torch.zeros((0,6), device=prediction.device)

    # Convert [x,y,w,h] -> [x1,y1,x2,y2]
    boxes = prediction[:, :4].clone()
    boxes[:, 0] = prediction[:, 0] - prediction[:, 2]/2
    boxes[:, 1] = prediction[:, 1] - prediction[:, 3]/2
    boxes[:, 2] = prediction[:, 0] + prediction[:, 2]/2
    boxes[:, 3] = prediction[:, 1] + prediction[:, 3]/2

    scores = prediction[:, 4]
    classes_pred = prediction[:, 5]

    # Filter by class
    if classes is not None:
        mask = torch.isin(classes_pred.long(), torch.tensor(classes, device=prediction.device))
        boxes, scores, classes_pred = boxes[mask], scores[mask], classes_pred[mask]

    if boxes.shape[0] == 0:
        return torch.zeros((0,6), device=prediction.device)

    # Class-aware NMS
    if not agnostic:
        max_wh = 4096
        offsets = classes_pred.view(-1,1) * max_wh
        boxes_for_nms = boxes + offsets
    else:
        boxes_for_nms = boxes
        offsets = torch.zeros_like(boxes)

    keep = torchvision.ops.nms(boxes_for_nms, scores, iou_thres)
    keep = keep[:max_det]

    # Return boxes without offsets
    return torch.cat([
        boxes[keep] - (offsets[keep] if not agnostic else 0),
        scores[keep, None],
        classes_pred[keep, None]
    ], 1)
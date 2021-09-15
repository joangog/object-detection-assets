# Copied and adjusted from repo pytorch/vision/references/detection/engine.py

import sys
import math
import time

import torch
import torchvision.models.detection as M
import torchvision.transforms.functional as F

from pycocotools import mask as cocomask

import scripts.utils as SU
import scripts.coco_utils as SCU
from scripts.coco_eval import CocoEvaluator


def convert_to_xyxy(bboxes):  # formats bboxes from (xmin,ymin,w,h) to (xmin,ymin,xmax,ymax)
  for bbox in bboxes:
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
  return bboxes


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = SU.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SU.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = SU.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)
        if model.__class__.__name__ == 'AutoShape':  # If model is from YOLOv5 package
          images = [F.to_pil_image(image) for image in images]  # Convert images from tensor to PIL

        # Format targets for torchvision models
        formatted_targets = []
        for i, img_targets in enumerate(targets):
          # Stack boxes, masks (optionally) and labels of image targets into tensor
          boxes = torch.stack([torch.squeeze(torch.Tensor(convert_to_xyxy([target['bbox']])),0) for target in img_targets]).long()
          labels = torch.Tensor([target['category_id'] for target in img_targets]).long()
          if model.__class__.__name__ == 'MaskRCNN':  # Get masks only for segmentation models
            masks = []
            # Convert every mask from polygon to binary mask
            for target in img_targets:
              for mask in target['segmentation']:
                height = images[i].shape[1]
                width = images[i].shape[2]
                formatted_mask = torch.Tensor(cocomask.decode(cocomask.frPyObjects([mask], height, width)))
                masks.append(formatted_mask)
            masks = torch.stack(masks).long()
            formatted_targets.append({'boxes': boxes, 'masks': masks, 'labels': labels})
          else:
            formatted_targets.append({'boxes': boxes, 'labels': labels})

        formatted_targets = [{k: v.to(device) for k, v in t.items()} for t in formatted_targets]

        loss_dict = model(images, formatted_targets)

        losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = SU.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, M.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, M.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = SU.MetricLogger(delimiter="  ")
    header = 'Test:'
    coco = SCU.get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Get label names
    label_ids = data_loader.dataset.coco.getCatIds()
    label_info = data_loader.dataset.coco.loadCats(label_ids)
    label_names = [label['name'] for label in label_info]
    labels = dict(zip(label_ids, label_names))  # Label dictionary with id-name as key-value
    labels_inv = dict(zip(label_names, label_ids))  # Inverse label dictionary with name-id as key-value

    for images, targets in metric_logger.log_every(data_loader, 100, header):

        images = list(img.to(device) for img in images)
        if model.__class__.__name__ == 'AutoShape':  # If model is from YOLO package
            images = [F.to_pil_image(image) for image in images]  # Convert images from tensor to PIL

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)  # Get model predictions
        model_time = time.time() - model_time

        if model.__class__.__name__ == 'AutoShape':  # If model is from YOLO package
            # Format outputs to COCO format
            outputs_formatted = []
            for img_outputs in outputs.xyxy:
                output_bboxes = img_outputs[:, :4]
                output_scores = img_outputs[:, 4]
                output_labels = img_outputs[:, 5].to(cpu_device).apply_(
                    lambda x: labels_inv[label_names[int(x)]])  # Convert YOLO label ids to COCO label ids
                outputs_formatted.append({
                    'boxes': output_bboxes,
                    'scores': output_scores,
                    'labels': output_labels
                })
            outputs = outputs_formatted

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target[0]["image_id"]: output for target, output in zip(targets, outputs) if len(target) != 0}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    # Model FPS
    batch_size = data_loader.batch_size
    fps = batch_size / metric_logger.meters['model_time'].global_avg

    # Model maximum memory usage
    MB = 1024.0 * 1024.0
    max_mem = torch.cuda.max_memory_allocated() / MB  # in MegaBytes

    return coco_evaluator, fps, max_mem, outputs
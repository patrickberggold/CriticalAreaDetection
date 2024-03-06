import os
import torch
import numpy as np
from collections import OrderedDict
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import cv2
SEP = os.sep

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    def clip_boxes(boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def dir_maker(store_folder_path: str, description_log: str, config: dict, train_config: dict):
    if os.path.isdir(store_folder_path):
        print('Path already exists!')
        quit()
    else:
        os.mkdir(store_folder_path)
        with open(os.path.join(store_folder_path, 'description.txt'), 'w') as f:
            f.write(description_log)
            f.write("\n\nCONFIG: {\n")
            for k in config.keys():
                f.write("'{}':'{}'\n".format(k, str(config[k])))
            f.write("}")
            f.write("\n\nTRAIN_CONFIG: {\n")
            for k in train_config.keys():
                f.write("'{}':'{}'\n".format(k, str(train_config[k])))
            f.write("}\n\n")
        f.close()


def load_ckpt(module: LightningModule, ckpt_path):
    CKPT_PATH = SEP.join(['checkpoints', ckpt_path])
    model_file_path = [file for file in os.listdir(CKPT_PATH) if file.endswith('.ckpt') and not file.startswith('last')]
    assert len(model_file_path) == 1
    CKPT_PATH = SEP.join([CKPT_PATH, model_file_path[0]])
    state_dict = torch.load(CKPT_PATH)['state_dict']
    module_state_dict = module.state_dict()

    mkeys_missing_in_loaded = [module_key for module_key in list(module_state_dict.keys()) if module_key not in list(state_dict.keys())]
    lkeys_missing_in_module = [loaded_key for loaded_key in list(state_dict.keys()) if loaded_key not in list(module_state_dict.keys())]
    assert len(mkeys_missing_in_loaded) < 10 or len(lkeys_missing_in_module) < 10, 'Checkpoint loading went probably wrong...'

    load_dict = OrderedDict()
    for key, tensor in module_state_dict.items():
        if key in state_dict.keys() and tensor.size()==state_dict[key].size():
            load_dict[key] = state_dict[key]
        else:
            load_dict[key] = tensor

    module.load_state_dict(load_dict)

    return module


def show_boxed_image(img, batch_result):
    # visualize the boxes
    def insert_bbox(img_, bbox, class_name, score=None, color=(255, 0, 0), thickness=2):

        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = round(x_min), round(y_min), round(x_max), round(y_max)
    
        img_ = np.zeros((img_.shape[0], img_.shape[1], 3), dtype=np.uint8) + img_
        cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color, thickness=thickness)
        
        boxText = f'{class_name}'
        if score is not None: boxText += f': {round(score*100)}%'
        ((text_width, text_height), _) = cv2.getTextSize(boxText, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
        cv2.rectangle(img_, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
        cv2.putText(
            img_,
            text=boxText,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35, 
            color=(255, 255, 255), 
            lineType=cv2.LINE_AA,
        )
        return img_

    plt.rcParams['figure.figsize'] = [20, 14]
    category_id_to_name_gt = {0: 'critArea', 1: 'Background'}
    category_id_to_name_pr = {0: 'Prediction', 1: 'Background'}
    color_gt = (0, 170, 0)
    color_pr = (170, 0, 0)

    true_boxes, true_labels, pred_boxes, confidences, pred_labels = batch_result
    
    for b in range(img.size(0)):
        img_np = img[b].cpu().numpy().copy().transpose(1,2,0)
        img_np = (img_np * 255).astype(np.uint8)
        labels_gt = true_labels[b].cpu().numpy()
        bboxes_gt = true_boxes[b].cpu().numpy()

        labels_pr = pred_labels[b].cpu().numpy()
        bboxes_pr = pred_boxes[b].detach().cpu().numpy()
        scores_pr = confidences[b].detach().cpu().numpy()

        # draw gt boxes individually
        for lgt, bgt in zip(labels_gt, bboxes_gt):
            class_name_gt = category_id_to_name_gt[lgt]
            img_np = insert_bbox(img_np, bgt, class_name_gt, score=None, color=color_gt)

        # if show_pred_images:
        #     plt.imshow(img_np)
        #     plt.close('all')
        # if save_pred_images:
        #     plt.imsave(f"ObjectDetection\\results\\{ARCH}\\OD_result_stage={stage}_batch={id_batch}_i={b}_gt.png", img_np)
        
        # draw pr boxes individually
        for lpr, bpr, score in zip(labels_pr, bboxes_pr, scores_pr):
            class_name_pr = category_id_to_name_pr[lpr]
            img_np = insert_bbox(img_np, bpr, class_name_pr, score=score, color=color_pr)

        plt.imshow(img_np)
        plt.close('all')


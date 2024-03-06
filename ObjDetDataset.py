from torch.utils.data import Dataset
from helper import SEP, xyxy2xywhn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch
import cv2

class ObjDetDataset(Dataset):
    def __init__(
        self, 
        config: dict,
        img_paths: list, 
        bboxes: list, 
        batch_size: int,
        transform = None,
    ):
        self.transform = transform
        self.img_paths = img_paths
        self.bboxes = bboxes
        self.arch = config['arch']
        self.img_max_width, self.img_max_height = config['img_max_size']
        self.augment_brightness = True if config['additional_queries'] == 'vanilla_imgAugm' else False
        self.color_mappings = {
            0: {(255, 255, 0): (100, 100, 0), (255, 0, 0): (100, 0, 0)},
            1: {(255, 255, 0): (175, 175, 0), (255, 0, 0): (175, 0, 0)}
        } if self.augment_brightness else None

        self.batch_size = batch_size
        self.agentIds = {'10': 0, '30': 1, '50': 2}
        self.ascentIds = {'En_Sn': 0, 'E1_S0': 1, 'E2_S0': 2, 'E3_S0': 3, 'E1_S2.4': 4, 'E2_S2.4': 5, 'E3_S2.4': 6}
        self.obstaclesIds = {'C0': 0, 'C2': 1}

        assert len(self.bboxes) == len(self.img_paths), 'Length of image paths and trajectory paths do not match, something went wrong!'

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        bbox_format = 'pascal_voc' # 'coco', 'pascal_voc'
        img_path, boxes_per_image = self.img_paths[idx], self.bboxes[idx]
        a, e, s, es, ss, c = boxes_per_image[-1]
        agentId = self.agentIds[a]
        central_id, sides_id, obstacles_id = self.ascentIds[f'E{e}_S{s}'], self.ascentIds[f'E{es}_S{ss}'], self.obstaclesIds[f'C{c}']
        boxes_per_image = boxes_per_image[:-1]

        labels = np.zeros((len(boxes_per_image)), dtype=np.int64)

        img = np.array(Image.open(img_path))[...,:3]
        if self.augment_brightness and agentId != 2:
            # Create masks for the colors to be replaced
            masks = [np.all(img == color, axis=-1) for color in self.color_mappings[agentId]]
            # Apply the color mappings using masks
            for color, mask in zip(self.color_mappings[agentId].values(), masks):
                img[mask] = color
            # plt.imshow(img)
        img = img.astype(np.float32) / 255.
       
        img, bboxes, labels = self.augmentations(img, boxes_per_image, labels, format=bbox_format)
        if len(bboxes) > 0:
            bboxes = np.array([(round(box[0], 3), round(box[1], 3), round(box[2], 3), round(box[3], 3)) for box in bboxes])
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0,4), dtype=np.float32)
            labels = np.zeros((0), dtype=np.int64)

        if bboxes.shape[0] != 0:
            assert (bboxes[:, 2:] >= bboxes[:, :2]).all()
        bboxes = xyxy2xywhn(bboxes, self.img_max_width, self.img_max_height)
        if bboxes.shape[0] > 0: assert 1.0 >= np.max(bboxes)

        img = torch.tensor(img).permute(2, 0, 1).float()
        bboxes = torch.tensor(bboxes).float()
        labels = torch.tensor(labels).long()

        if self.transform:
            img = self.transform(img)
        
        return img, labels, bboxes, torch.tensor([agentId, central_id, sides_id, obstacles_id], dtype=torch.long)


    def augmentations(self, image, bboxes, labels, format='pascal_voc', file=None, vis=False):
        p_trans = 1.0
        max_long_size = 1500
        transform_rectangular = A.Compose([
            A.LongestMaxSize(max_size=max_long_size, always_apply=True),
            A.PadIfNeeded(self.img_max_height, self.img_max_width, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True), # (4224, 4224) (561, 4174) # 1344
            A.augmentations.geometric.transforms.Affine(translate_px={'x': (0, 50), 'y': (0, 50)}, p=p_trans),
            A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
            A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
            # A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        ], bbox_params=A.BboxParams(format=format, label_fields=['category_ids']))
        transformed = transform_rectangular(image=image, bboxes=bboxes, category_ids=labels)
        # if vis is not None: visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], {0: 'Critical area'}, file=file)
        return transformed['image'], transformed['bboxes'], transformed['category_ids']
    
    
def visualize(image, bboxes, category_ids, category_id_to_name, format='pascal_voc',file=None):

    def visualize_bbox(img, bbox, class_name, color=(0, 175, 0), thickness=2, format=format):
        
        img = np.ascontiguousarray(img) 
        if format=='coco':
            x_min, y_min, w, h = bbox
            x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        elif format=='pascal_voc':
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (0, 175, 0), -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35, 
            color=(255, 255, 255), 
            lineType=cv2.LINE_AA,
        )
        # img_cp = img_cp.astype(np.float32) / 255.
        return img

    img = (image.copy()*255).astype(np.uint8)
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)

    # plt.figure(figsize=(12, 12))
    if file is not None:
        plt.imsave(f'{file.split(SEP)[-1]}', img)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.close('all')


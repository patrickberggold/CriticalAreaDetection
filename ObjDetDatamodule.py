import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os, random
from ObjDetDataset import ObjDetDataset
from helper import SEP
from tqdm import tqdm

class ObjDetDatamodule(pl.LightningDataModule):
    def __init__(self, config: dict, num_workers: int = 0):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.arch = config['arch']
        self.cuda_device = config['cuda_device']
        self.num_workers = num_workers
        self.transforms = None

        assert self.arch in ['Detr_custom']
        self.transforms = None # ToTensor()

        self.set_data_paths()

    def setup(self, stage):
        self.train_dataset = ObjDetDataset(self.config, self.train_imgs_list, self.train_targets_list, transform=self.transforms, batch_size = self.batch_size)
        self.val_dataset = ObjDetDataset(self.config, self.val_imgs_list, self.val_targets_list, transform=self.transforms, batch_size = self.batch_size)
        self.test_dataset = ObjDetDataset(self.config, self.test_imgs_list, self.test_targets_list, transform=self.transforms, batch_size = self.batch_size)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)
        

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_collate)
        

    def custom_collate(self, batch):
        images, labels, bboxes, numAgentsIds = zip(*batch)
        return torch.stack(images, dim=0), labels, bboxes, torch.stack(numAgentsIds, dim=0)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.cuda_device != 'cpu':
            device = torch.device('cuda', self.cuda_device)
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            return batch

    def set_data_paths(self):
        
        self.splits = [0.7, 0.15, 0.15]

        # load example dataset from the following paths
        self.img_path = SEP.join(['ExampleDataset', 'inputs'])
        self.boxes_path = SEP.join(['ExampleDataset', 'targets'])

        self.set_filepaths()

        assert len(self.img_list) == len(self.bboxes), 'Images list and trajectory list do not have same length, something went wrong!'
    	
        val_split_factor = self.splits[1]
        test_split_factor = self.splits[2]
        
        self.indices = list(range(len(self.img_list)))
        
        val_split_index = int(len(self.indices) * val_split_factor)
        test_split_index = int(len(self.indices) * test_split_factor)
        
        random.seed(42)
        random.shuffle(self.indices)

        self.train_imgs_list = [self.img_list[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        self.train_targets_list = [self.bboxes[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        
        self.val_imgs_list = [self.img_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]
        self.val_targets_list = [self.bboxes[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]

        self.test_imgs_list = [self.img_list[idx] for idx in self.indices[:test_split_index]]
        self.test_targets_list = [self.bboxes[idx] for idx in self.indices[:test_split_index]]


    def set_filepaths(self):
        self.img_list = []
        self.crowdit_list = []
        self.bboxes = []

        for box_file, img_file in zip(os.listdir(self.boxes_path), os.listdir(self.img_path)):
            assert box_file.split('_critArea')[0] == img_file.replace('.png', '')
            self.img_list.append(os.path.join(self.img_path, img_file))

            boxes_per_file = []
            f = open(os.path.join(self.boxes_path, box_file), "r")
            file_lines = f.readlines()
            for line in file_lines:
                x1, y1, x2, y2 = line.strip().split(',')
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if y1==y2 or x1==x2:
                    continue
                assert y2 > y1 and x2 > x1
                boxes_per_file.append([x1, y1, x2, y2])
            f.close()

            additional_info = box_file.split('_critArea')[0].split('_')
            a, e, s, es, ss, c = additional_info[1][1:], additional_info[4][1:], additional_info[5][1:], additional_info[8][2:], additional_info[9][2:], additional_info[2][1]
            assert a in ['10', '30', '50'] and e in ['n', '1', '2', '3'] and es in ['n', '1', '2', '3'] and s in ['n', '0', '2.4'] and ss in ['n', '0', '2.4'] and c in ['0', '2']
            boxes_per_file.append([a, e, s, es, ss, c])
            self.bboxes.append(boxes_per_file)

        assert len(self.img_list) == len(self.bboxes), 'Images list and boxes list do not have same length, something went wrong!'


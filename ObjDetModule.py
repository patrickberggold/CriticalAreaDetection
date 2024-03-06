import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import pytorch_lightning as pl
import numpy as np
from helper import SEP, xywhn2xyxy
from metrics_calc import metrics_sklearn
from decimal import Decimal

class ObjDetModule(pl.LightningModule):
    def __init__(
        self, 
        config: dict,
        train_config: dict,
        ):
        super(ObjDetModule, self).__init__()
        self.config = config
        self.arch = config['arch']
        self.batch_size = config['batch_size']
        
        assert self.arch in ['Detr_custom'], 'Unknown arch setting!'

        self.learning_rate = train_config['learning_rate']
        self.lr_scheduler = train_config['lr_scheduler']
        self.lr_sch_gamma4redOnPlat_and_stepLR = train_config['lr_sch_gamma4redOnPlat']
        self.lr_sch_patience4redOnPlat = train_config['lr_sch_patience4redOnPlat']
        self.opt = train_config['opt']
        self.weight_decay = train_config['weight_decay']

        self.num_heads = 1

        self.train_losses = {}
        self.train_losses_per_epoch = {}
        self.val_losses = {}
        self.val_losses_per_epoch = {}
        
        self.log_result = {'validation': [], 'training': []}
        self.backbone_frozen = False
        self.tversky_weights = None
        self.post_inference_call = True
        self.num_use_cases = 2
        self.pred_boxes, self.pred_labels, self.confidences = [], [], []
        self.true_boxes, self.true_labels = [], []

        self.img_max_width, self.img_max_height = config['img_max_size']
        self.save_results = config['save_results']
        self.txt_path = config['store_path'] if self.save_results and 'store_path' in config else None

        if self.arch == "Detr_custom":
            from models.Detr_custom import Detr_custom
            self.model = Detr_custom.from_pretrained("facebook/detr-resnet-50")
            self.model.update_model(self.config)

    
    def training_step(self, batch, batch_idx: int):

        if self.arch in ['Detr_custom']:
            img, labels, bboxes, numAgentsIds = batch

            target = [{"class_labels": l, "boxes": b} for l, b in zip(labels, bboxes)]
            prediction = self.model(img, labels=target) if self.arch == 'Detr' else self.model(img, numAgentsIds, labels=target)
            train_loss = prediction.loss

        self.internal_log({'train_loss': train_loss}, stage='train')
        self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
        
        return {'loss' : train_loss}


    def validation_step(self, batch, batch_idx: int) -> None:

        if self.arch in ['Detr_custom']:
            img, labels, bboxes, numAgentsIds = batch

            target = [{"class_labels": l, "boxes": b} for l, b in zip(labels, bboxes)]
            prediction = self.model(img, numAgentsIds, labels=target)
            val_loss = prediction.loss

            for i in range(prediction.logits.size(0)):
                scores = torch.softmax(prediction.logits[i].detach(), dim=-1)
                argmaxes = torch.argmax(scores, dim=-1)

                box_detection_indices = torch.argwhere(argmaxes == 0).squeeze() # select only boxes with class 0 (class 1 == 'no-object')
                selected_scores = scores[box_detection_indices, 0]
                assert torch.all(selected_scores > scores[box_detection_indices,1])
                box_proposals_per_batch = prediction.pred_boxes[i].detach()
                selected_boxes = box_proposals_per_batch[box_detection_indices]
                selected_labels = argmaxes[box_detection_indices].detach()

                # for score predictions consisting of a single number
                if selected_scores.ndim==0:
                    selected_boxes, selected_labels, selected_scores = selected_boxes.unsqueeze(0), selected_labels.unsqueeze(0), selected_scores.unsqueeze(0)

                self.pred_boxes += [xywhn2xyxy(selected_boxes, img.size(3), img.size(2))] # 2048, 789
                self.confidences += [selected_scores]
                self.pred_labels += [selected_labels.long()]
            
            self.true_labels += labels
            self.true_boxes += [xywhn2xyxy(box, img.size(3), img.size(2)) for box in bboxes]

        self.internal_log({'val_loss': val_loss}, stage='val')
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {'val_loss': val_loss}


    def internal_log(self, losses_it, stage):
        if self.trainer.state.stage == 'sanity_check': return

        losses_logger = self.train_losses if stage=='train' else self.val_losses

        for key, val in losses_it.items():
            if isinstance(val, torch.Tensor): val = val.detach()
            if key not in losses_logger:
                losses_logger.update({key: [val]})
            else:
                losses_logger[key].append(val)


    def configure_optimizers(self):
        if self.arch in ['Detr_custom']:
            param_dicts = [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n]},
                {"params": [p for n, p in self.named_parameters() if "backbone" in n], "lr": 1e-5}
            ]
            opt = Adam(param_dicts, lr=self.learning_rate)
            sch = ReduceLROnPlateau(opt, factor=self.lr_sch_gamma4redOnPlat_and_stepLR, patience=self.lr_sch_patience4redOnPlat, verbose=True)
            # Because of a weird issue with ReduceLROnPlateau, the monitored value needs to be returned... See https://github.com/PyTorchLightning/pytorch-lightning/issues/4454
            return {
                'optimizer': opt,
                'lr_scheduler': sch,
                'monitor': 'val_loss'
            }
        else:
            raise NotImplementedError


    def on_train_epoch_start(self) -> None:
        if self.trainer.state.stage in ['sanity_check']: return super().on_train_epoch_end()
        # print logs to console
        if self.current_epoch > 0: 
            self.print_logs()
    

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.state.stage == 'sanity_check':
            map, _ = metrics_sklearn(self.true_boxes, self.true_labels, self.pred_boxes, self.pred_labels, self.confidences)
            self.internal_log({'AP': map}, stage='val')
            self.true_boxes, self.true_labels, self.pred_boxes, self.pred_labels, self.confidences = [], [], [], [], []
        return super().on_validation_epoch_end()


    def print_logs(self):
        # Training Logs
        for key, val in self.train_losses.items():
            if key not in self.train_losses_per_epoch:
                mean = torch.as_tensor(val).nanmean()
                self.train_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.train_losses_per_epoch[key].append(torch.as_tensor(val).nanmean().item())

        # Validation logs
        for key, val in self.val_losses.items():
            if key not in self.val_losses_per_epoch:
                mean = torch.as_tensor(val).nanmean()
                self.val_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.val_losses_per_epoch[key].append(torch.as_tensor(val).nanmean().item())

        # Reset
        self.train_losses = {}
        self.val_losses = {}
        
        # print('\nTRAINING RESULT:')
        train_string = f'TRAINING RESULT:\nEpoch\t'
        train_vals = [val for val in self.train_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.train_losses_per_epoch.keys())):
            if id_k == 0:
                train_string += key+':'
            else:
                train_string += '\t\t' + key+':'
        for i_epoch in range(len(train_vals[0])):
            for i_loss in range(len(train_vals)):
                if i_loss == 0:
                    train_string += f'\n{i_epoch}:\t{Decimal(train_vals[i_loss][i_epoch]):.3e}'
                else:
                    train_string += f'\t\t{Decimal(train_vals[i_loss][i_epoch]):.3e}'
        print('\n\n'+train_string) 


        # print('\nVALIDATION RESULT:')
        val_string = f'\nVALIDATION RESULT:\nEpoch\t'
        val_vals = [val for val in self.val_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.val_losses_per_epoch.keys())):
            if id_k == 0:
                val_string += key+':'
            else:
                val_string += '\t\t' + key+':'
        for i_epoch in range(len(val_vals[0])):
            for i_loss in range(len(val_vals)):
                if i_loss == 0:
                    # val_string += f'\n{i_epoch}:\t{val_vals[i_loss][i_epoch]:.5f}'
                    val_string += f'\n{i_epoch}:\t{Decimal(val_vals[i_loss][i_epoch]):.3e}'
                else:
                    # val_string += f'\t\t\t{val_vals[i_loss][i_epoch]:.5f}'
                    val_string += f'\t\t{Decimal(val_vals[i_loss][i_epoch]):.3e}'
        print(val_string+'\n')

        if self.save_results and self.txt_path is not None:
            save_string = train_string+'\n\n'+val_string
            f = open(self.txt_path, 'w')
            f.write(f'Latest learning rate:{self.learning_rate}\n\n')
            f.write(save_string)
            f.close()




import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ObjDetDatamodule import ObjDetDatamodule
from ObjDetModule import ObjDetModule
from helper import SEP, dir_maker, load_ckpt, xywhn2xyxy, show_boxed_image
from tqdm import tqdm

CUDA_DEVICE = 0
ARCH = 'Detr_custom' # Detr, Detr_custom

test_run = False # limit_batches -> 2 
save_model = False # create folder and save the model

do_training = True

CONFIG = {
    'arch': ARCH,
    'cuda_device': CUDA_DEVICE,
    'batch_size': 4,
    'from_ckpt_path': None, # load the model
    'load_to_ckpt_path': None, # save the model
    'early_stopping_patience': 50,
    'run_test_epoch': test_run,
    'save_model': save_model,
    'img_max_size': (1536, 640),
    'num_classes': 1, # class = critical area
    'top_k': 100,
    'save_results': True,
    # input options: 'vanilla', 'vanilla_imgAugm', 'before_encoder', 'after_encoder', 'before_encoder+', 'after_encoder+',
    'additional_queries': 'after_encoder+',
    'facebook_pretrained': True,
}

TRAIN_DICT = {
    # Learning rate and schedulers
    'learning_rate': 0.0001,
    'lr_scheduler': 'ReduceLROnPlateau', 
    'lr_sch_gamma4redOnPlat': 0.75,
    'lr_sch_patience4redOnPlat': 10,
    # optimizer
    'opt': 'Adam',
    'weight_decay': None, # 1e-6,
    'gradient_clip': 100.,
    # (un)freezing layers
    'unfreeze_backbone_at_epoch': None,
}

if __name__ == '__main__':

    # dataloading and applying the splits
    datamodule = ObjDetDatamodule(config=CONFIG)
    
    ############
    # training #
    ############
    if do_training:
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=CONFIG['early_stopping_patience']), LearningRateMonitor(logging_interval='epoch')]
        if (not test_run) and save_model:
            # save the model checkpoint + config in a new folder (store_folder_path)
            assert CONFIG['load_to_ckpt_path'] is not None, 'Please provide a name for the folder to store the model...'
            store_folder_path = SEP.join(['checkpoints', CONFIG['load_to_ckpt_path']])
            description_log = ''
            CONFIG.update({'store_path': store_folder_path+SEP+'results.txt'})
            dir_maker(store_folder_path, description_log, CONFIG, TRAIN_DICT)

            # save the model checkpoint
            model_checkpoint = ModelCheckpoint(
                dirpath = store_folder_path,
                filename = 'model_{epoch}-{step}',
                save_top_k = 1,
                verbose = True, 
                monitor = 'val_loss',
                mode = 'min',
                save_last = False,
            )
            callbacks.append(model_checkpoint)

        # object detection module
        module = ObjDetModule(config=CONFIG, train_config=TRAIN_DICT)

        # load from checkpoint
        if CONFIG['from_ckpt_path']:
            module = load_ckpt(module, ckpt_path = CONFIG['from_ckpt_path'])
        
        limit_batches = 2 if CONFIG['run_test_epoch'] else None
        trainer = pl.Trainer(
            devices=[CUDA_DEVICE], #f'cuda:{str(CUDA_DEVICE)}', 
            max_epochs = 500, 
            callbacks=callbacks,
            limit_train_batches=limit_batches,
            limit_val_batches=limit_batches,
            # progress_bar_refresh_rate=125,
            gradient_clip_val=TRAIN_DICT['gradient_clip'],
            )

        start_training_time = time.time()
        trainer.fit(module, datamodule=datamodule)
        print(f'Training took {(time.time() - start_training_time)/60./(module.current_epoch+1):.3f} minutes per epoch...')

    #############
    # inference #
    #############
    else:
        # object detection module
        module = ObjDetModule(config=CONFIG, train_config=TRAIN_DICT)
        if CONFIG['from_ckpt_path']:
            module = load_ckpt(module, ckpt_path = CONFIG['from_ckpt_path'])
        module.to(f'cuda:{CUDA_DEVICE}')
        module.eval()

        # only use dataloader from the test set
        datamodule.setup(stage='test')
        loader = datamodule.test_dataloader()

        # iterate over the test set
        # can be used to calculate the AP, recall, etc. over the test set
        all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_confidences = [], [], [], [], []
        for id_batch, batch in tqdm(enumerate(loader), total=len(loader.dataset.img_paths)//CONFIG['batch_size']):
            # format batch inputs correctly
            img, labels_gt, bboxes_gt, numAgentsIds_b = batch
            img, labels_gt, bboxes_gt, numAgentsIds_b = img.to(f'cuda:{CUDA_DEVICE}'), [l.to(f'cuda:{CUDA_DEVICE}') for l in labels_gt], [b.to(f'cuda:{CUDA_DEVICE}') for b in bboxes_gt], numAgentsIds_b.to(f'cuda:{CUDA_DEVICE}')

            target = [{"class_labels": l, "boxes": b} for l, b in zip(labels_gt, bboxes_gt)]
            prediction = module.model(img, labels=target) if ARCH == 'Detr' else module.model(img, numAgentsIds_b, labels=target)
            val_loss = prediction.loss

            batch_gt_boxes = [xywhn2xyxy(box, img.size(3), img.size(2)).detach().cpu() for box in bboxes_gt]
            batch_gt_labels = [l.detach().cpu() for l in labels_gt]
            batch_pr_boxes, batch_pr_scores, batch_pr_labels = [], [], []
            for i in range(prediction.logits.size(0)):
                scores = torch.softmax(prediction.logits[i].detach(), dim=-1)
                argmaxes = torch.argmax(scores, dim=-1)
                box_detection_indices = torch.argwhere(argmaxes == 0).squeeze() # select only boxes with class 0 (class 1 == 'no-object')
                selected_scores = scores[box_detection_indices, 0]
                # assert torch.all(selected_scores > scores[box_detection_indices, 1])
                box_proposals_per_batch = prediction.pred_boxes[i].detach()
                selected_boxes = box_proposals_per_batch[box_detection_indices]
                selected_labels = argmaxes[box_detection_indices].detach()
                # for score predictions consisting of a single number, get the dimensions right
                if selected_scores.ndim==0:
                    selected_boxes, selected_labels, selected_scores = selected_boxes.unsqueeze(0), selected_labels.unsqueeze(0), selected_scores.unsqueeze(0)
                
                batch_pr_boxes.append(xywhn2xyxy(selected_boxes, img.size(3), img.size(2)).detach().cpu())
                all_pred_boxes += batch_pr_boxes
                batch_pr_scores.append(selected_scores.cpu())
                all_confidences += batch_pr_scores
                batch_pr_labels.append(selected_labels.long().cpu())
                all_pred_labels += batch_pr_labels

            all_true_labels += batch_gt_labels
            all_true_boxes += batch_gt_boxes

            batch_result = [batch_gt_boxes, batch_gt_labels, batch_pr_boxes, batch_pr_scores, batch_pr_labels]

            # display predicted bounding boxes on the images
            show_boxed_image(img, batch_result)
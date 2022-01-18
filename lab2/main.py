import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils import data
from tqdm import tqdm

from tvid import TvidDataset
from detector import Detector
from utils import compute_iou


lr = 5e-3
batch = 64
epochs = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"
iou_thr = 0.5


class PL_Detector(pl.LightningModule):

    # TODO Implement the Pytorch_Lightning  Module pipeline.

    def __init__(self, backbone, lengths, num_classes):
        super().__init__()
        self.model = Detector(backbone, lengths, num_classes)
        self.criterion = { 'cls': nn.CrossEntropyLoss(), 'box': nn.L1Loss() }
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 128, 128), dtype=torch.float32)

    def forward(self, imgs):
        logits, bbox = self.model(imgs)
        return logits, bbox
    
    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        gt_cls, gt_bbox = labels['cls'], labels['bbox']
        logits, bbox = self.forward(imgs)

        loss = self.criterion['cls'](logits, gt_cls) + 10 * self.criterion['box'](bbox, gt_bbox)
        mAP = (logits.argmax(dim=-1) == labels['cls']).float().mean()
        
        self.log("train_mAP", mAP, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        
        return loss

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        X, gt_cls, gt_bbox = imgs, labels['cls'], labels['bbox']
        logits, bbox = self.model(X)
        mAP = (logits.argmax(dim=-1) == labels['cls']).float().mean()

        self.log("test_mAP", mAP)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        X, gt_cls, gt_bbox = imgs, labels['cls'], labels['bbox']
        logits, bbox = self.model(X)
        mAP = (logits.argmax(dim=-1) == labels['cls']).float().mean()

        self.log("val_mAP", mAP)
    # End of todo



def main():
    # TODO Implement the `main()`.
    
    # Set the trainer
    trainer = pl.Trainer(
        default_root_dir = 'lab2/model_saved',
        gpus = 1 if str(device) == "cuda:0" else 0,
        max_epochs = epochs,
        log_every_n_steps=5
        # progress_bar_refresh_rate = 1
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    
    # Split the data for train, test, validation
    trainset = TvidDataset(root='~/data/tiny_vid', mode='train')
    pl.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(trainset, [704, 46])
    pl.seed_everything(42)
    _, val_set = torch.utils.data.random_split(trainset, [704, 46])
    test_set = TvidDataset(root='~/data/tiny_vid', mode='test')
 
    # Define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=batch, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=batch, shuffle=False, drop_last=False, num_workers=4)

    # Define the model
    pl.seed_everything(42) # To be reproducable
    model = PL_Detector(backbone='resnet50', lengths=(2048 * 4 * 4, 2048, 512), num_classes=5)
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_mAP"], "val": val_result[0]["test_mAP"]}

    print('results:', result)
    print("Models has been saved in floder `model_saved`!")
    
    # End of todo

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn

from torch.utils import data
from tqdm import tqdm

from tvid import TvidDataset
from detector import Detector
from utils import compute_iou


lr = 5e-3
batch = 32
epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
iou_thr = 0.5


def train_epoch(model, dataloader, criterion: dict, optimizer,
                scheduler, epoch, device):
    model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch {epoch:2}')
    correct, total = 0, 0
    for X, y in bar:

        # TODO Implement the train pipeline.

        ...

        # End of todo

        bar.set_postfix_str(f'lr={scheduler.get_last_lr()[0]:.4f}'
                            ' acc={correct / total * 100:.2f}'
                            ' loss={loss.item():.2f}')
    scheduler.step()


def test_epoch(model, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        correct, correct_cls, total = 0, 0, 0
        for X, y in dataloader:

            # TODO Implement the test pipeline.

            ...

            # End of todo

        print(f' val acc: {correct / total * 100:.2f}')


def main():
    trainloader = data.DataLoader(TvidDataset(root='~/data/tiny_vid', mode='train'),
                                  batch_size=batch, shuffle=True, num_workers=4)
    testloader = data.DataLoader(TvidDataset(root='~/data/tiny_vid', mode='test'),
                                 batch_size=batch, shuffle=True, num_workers=4)
    model = Detector(backbone='resnet50', lengths=(2048 * 4 * 4, 2048, 512),
                     num_classes=5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95,
                                                last_epoch=-1)
    criterion = {'cls': nn.CrossEntropyLoss(), 'box': nn.L1Loss()}

    for epoch in range(epochs):
        train_epoch(model, trainloader, criterion, optimizer,
                    scheduler, epoch, device)
        test_epoch(model, testloader, device, epoch)


if __name__ == '__main__':
    main()

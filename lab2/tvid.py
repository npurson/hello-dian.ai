import os
from torch.utils import data

from utils import transforms


class TvidDataset(data.Dataset):

    CLASSES = [ 'bird', 'car', 'dog', 'lizard', 'turtle' ]

    def __init__(self, root, mode):

        assert mode in ['train', 'test']
        if not os.path.isabs(root):
            root = os.path.expanduser(root) if root[0] == '~' else os.path.abspath(root)

        self.images = []
        self.transforms = transforms.Compose([
            transforms.LoadImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomResize(0.8, 1.2),
            # transforms.RandomCrop((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if mode == 'train' else \
        transforms.Compose([
            transforms.LoadImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomResize(0.5, 2.5),
            # transforms.RandomCrop((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        for c, cls in enumerate(self.CLASSES):
            dir = os.path.join(root, cls)
            ann = os.path.join(root, cls + '_gt.txt')

            with open(ann) as f:
                for i, line in enumerate(f):
                    if (mode == 'train' and i >= 150) or i >= 180:
                        break
                    if mode == 'test' and i < 150:
                        continue

                    idx, *xyxy = line.strip().split(' ')
                    self.images.append({
                        'path': os.path.join(root, cls, '%06d.JPEG' % int(idx)),
                        'cls': c,
                        'bbox': [int(c) for c in xyxy],
                    })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img, bbox = self.transforms(img_info['path'], img_info['bbox'])
        return img, { 'cls': img_info['cls'], 'bbox': bbox }


if __name__ == '__main__':

    dataset = TvidDataset(root='~/data/tiny_vid', mode='train')
    import pdb; pdb.set_trace()

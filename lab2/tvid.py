import os
from torch.utils import data
import transforms


class TvidDataset(data.Dataset):

    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.
    def __init__(self, root, mode) -> None:
        super(data.Dataset, TvidDataset).__init__()
        




    def __getitem__(self, idx):



    def __len__(self):
        


    ...

    # End of todo


if __name__ == '__main__':

    dataset = TvidDataset(root='~/data/tiny_vid', mode='train')
    import pdb; pdb.set_trace()

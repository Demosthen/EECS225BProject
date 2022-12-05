import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os


def make_dataset(root: str) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rgb_path, gt_path)
    """
    dataset = []
    # Directory Names
    rgb_dir = 'input'
    gt_dir = 'gt'
    rgb_fnames = sorted(os.listdir(os.path.join(root, rgb_dir)))
    for gt_fname in sorted(os.listdir(os.path.join(root, gt_dir))):
        if gt_fname in rgb_fnames:
            # if we have a match, create pair of full paths and append
            rgb_path = os.path.join(root, rgb_dir, gt_fname)
            gt_path = os.path.join(root, gt_dir, gt_fname)
            item = (rgb_path, gt_path)
            dataset.append(item)
    return dataset


class InputGTDataset(VisionDataset):

    def __init__(self, root, loader=default_loader, input_transform=None, gt_transform=None):
        super().__init__(root, transform=input_transform, target_transform=gt_transform)
        samples = make_dataset(self.root)
        self.loader = loader
        self.samples = samples
        self.rgb_samples = [s[0] for s in samples]
        self.gt_samples = [s[1] for s in samples]

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        rgb_path, gt_path = self.samples[index]
        rgb_sample = self.loader(rgb_path)
        gt_sample = self.loader(gt_path)
        # potential transforms
        if self.transform is not None:
            rgb_sample = self.transform(rgb_sample)
        if self.target_transform is not None:
            gt_sample = self.target_transform(gt_sample)
        return rgb_sample, gt_sample, rgb_path

    def __len__(self):
        return len(self.samples)


def get_dataloader(root: str, batch_size: int, shuffle: bool):
    transforms = ToTensor()
    dataset = InputGTDataset(
        root, input_transform=transforms, gt_transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


<<<<<<< HEAD
# dataloader = get_dataloader(
#     "datasets/test_data_loader/", batch_size=4, shuffle=True)
# print(dataloader.dataset.rgb_samples) # not sorted to batches
# for i, (rgb, gt, rgb_path) in enumerate(dataloader):
#     print(i)
#     for i in range(4):
#         # print(rgb_path)
#         plt.figure(figsize=(10, 5))
#         plt.subplot(221)
#         plt.imshow(rgb[i].squeeze().permute(1, 2, 0))
#         plt.title(f'RGB img{i+1}')
#         plt.subplot(222)
#         plt.imshow(gt[i].squeeze().permute(1, 2, 0))
#         plt.title(f'GT img{i+1}')
#         plt.show()
=======
import matplotlib.pyplot as plt

dataloader = get_dataloader("datasets/test_data_loader/", batch_size=4, shuffle=True)
for i, (rgb, gt) in enumerate(dataloader):
    print(i)
    for i in range(4):
        plt.figure(figsize=(10, 5))
        plt.subplot(221)
        plt.imshow(rgb[i].squeeze().permute(1, 2, 0))
        plt.title(f'RGB img{i+1}')
        plt.subplot(222)
        plt.imshow(gt[i].squeeze().permute(1, 2, 0))
        plt.title(f'GT img{i+1}')
        plt.show()
>>>>>>> 87932b63720390e76a26dfdb5da86ee825dacee4

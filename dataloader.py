import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Grayscale, Resize
import os
from utils.common_utils import *

def equal_fnames(rgb_fname, gt_fname):
    return gt_fname == rgb_fname

def make_dataset(root: str, compare_func) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rgb_path, gt_path)
    """
    if compare_func == None:
        compare_func = equal_fnames
    dataset = []
    # Directory Names
    rgb_dir = 'input'
    gt_dir = 'gt'
    gt_fnames = sorted(os.listdir(os.path.join(root, gt_dir)))
    for rgb_fname in sorted(os.listdir(os.path.join(root, rgb_dir))):
        for gt_fname in gt_fnames:
            if compare_func(rgb_fname, gt_fname):
                # if we have a match, create pair of full paths and append
                rgb_path = os.path.join(root, rgb_dir, gt_fname)
                gt_path = os.path.join(root, gt_dir, gt_fname)
                item = (rgb_path, gt_path)
                dataset.append(item)
    return dataset

def make_dataset_gopro(root: str, compare_func) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rgb_path, gt_path)
    """
    if compare_func == None:
        compare_func = equal_fnames
    dataset = []
    # Directory Names
    rgb_dir = 'train_blur'
    gt_dir = 'train_sharp'

    gt_folders = os.listdir(os.path.join(root, gt_dir))
    rgb_folders = os.listdir(os.path.join(root, rgb_dir))
    for gt_folder, rgb_folder in zip(gt_folders, rgb_folders):
        full_gt_folder = os.listdir(os.path.join(root, gt_dir, gt_folder))
        for fname in full_gt_folder:
            rgb_path = os.path.join(root, rgb_dir, rgb_folder, fname)
            gt_path = os.path.join(root, gt_dir, gt_folder, fname)
            item = (rgb_path, gt_path)
            dataset.append(item)
        if len(dataset) > 1000:
            break
    return dataset


class InputGTDataset(VisionDataset):

    def __init__(self, root, compare_func, loader=default_loader, input_transform=None, gt_transform=None, use_gopro_data=False):
        super().__init__(root, transform=input_transform, target_transform=gt_transform)
        if use_gopro_data:
            samples = make_dataset_gopro(self.root, compare_func)
        else:
            samples = make_dataset(self.root, compare_func)
        self.loader = loader
        self.samples = samples
        input_depth = 8
        n_k = 200
        self.net_inputs = [get_noise(input_depth, 'noise',
                              (300, 300)) for s in samples]
        self.net_input_kernels = [get_noise(n_k, 'noise', (1, 1)).squeeze() for s in samples]
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
        return rgb_sample, gt_sample, rgb_path, self.net_inputs[index], self.net_input_kernels[index]

    def __len__(self):
        return len(self.samples)


def get_dataloader(root: str, batch_size: int, shuffle: bool, compare_func=None, use_gopro_data=False):
    transforms = Compose([Resize([255, 255]), Grayscale(3), ToTensor()])
    dataset = InputGTDataset(
        root, compare_func, input_transform=transforms, gt_transform=transforms, use_gopro_data=use_gopro_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# dataloader = get_dataloader(
#     "datasets/test_data_loader/", batch_size=4, shuffle=True)
# for i, (rgb, gt, rgb_path) in enumerate(dataloader):
#     print(i)
#     for i in range(4):
#         plt.figure(figsize=(10, 5))
#         plt.subplot(221)
#         plt.imshow(rgb[i].squeeze().permute(1, 2, 0))
#         plt.title(f'RGB img{i+1}')
#         plt.subplot(222)
#         plt.imshow(gt[i].squeeze().permute(1, 2, 0))
#         plt.title(f'GT img{i+1}')
#         plt.show()
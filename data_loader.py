import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import torchvision.transforms as T
import torch
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def save_image(tensor, index):
    t = T.ToPILImage()
    pil_image = t(tensor)
    pil_image.save(f'images/{index}.png')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)



class FilteredDatasetFolder(data.Dataset):

    def __init__(self, root, extensions, split, transform=None, target_transform=None, transform_basic=None, filter_classes=None):
        classes, class_to_idx = self.find_classes(root)
        samples = self.make_dataset(root, split, filter_classes)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                   "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.basic_transform = transform_basic
        self.target_transform = target_transform
        self.filter_classes = filter_classes

    def get_samples_class(self, y):
        return list(filter(lambda x: x[1] == y, self.samples))

    def reduce(self, size=50):
        train_samples = []
        val_samples = []

        # for each class in the iteration
        for label in torch.unique(torch.tensor(self.targets)):
            samples = self.get_samples_class(label.item())
            # 20% for validation
            val_samples = val_samples + samples[:size]
            # 80% for training
            train_samples = train_samples + samples[size:]

        # replace training samples with the reducted version
        self.samples = train_samples

        return val_samples

    def extend(self, size=0.2, val_set=None):
        train_samples = []
        val_samples = []

        # for each class in the iteration
        for label in torch.unique(torch.tensor(self.targets)):
            samples = self.get_samples_class(label.item())
            # compute the 20% of the total samples
            val_size = int(size * len(samples))
            # 20% for validation
            val_samples = val_samples + samples[:val_size]
            # 80% for training
            train_samples = train_samples + samples[val_size:]

        # replace training samples with the reducted version
        self.samples = train_samples

        if val_set is not None:
            val_samples = val_set + val_samples

        return val_samples

    def make_dataset(self, dir, class_to_idx, extensions, classes):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            if class_to_idx[target] not in classes:
                continue

            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images

    def set_transform(self, transform=None):
        if transform is not None:
            self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = Image.open(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            if target in self.target_transform.keys():
                target = self.target_transform[target]
            else:
                target = len(self.target_transform.keys())
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class LightFilteredDatasetFolder(data.Dataset):
    """A data loader only for validation samples """

    def __init__(self, samples=None, transform=None, target_transform=None):
        if len(samples) == 0 or samples is None:
            raise (RuntimeError("Found 0 files"))

        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = Image.open(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            if target in self.target_transform.keys():
                target = self.target_transform[target]
            else:
                target = len(self.target_transform.keys())
        return sample, target

    def get_samples_class(self, y):
        return list(filter(lambda x: x[1] == y, self.samples))

    def __len__(self):
        return len(self.samples)

    def set_transform(self, transform=None):
        if transform is not None:
            self.transform = transform

class RODFolder(FilteredDatasetFolder):
    """ Dataset to load both RGB and D images."""

    def __init__(self, root, split, transform=None, target_transform=None, transform_basic=None, classes=None):
        super(RODFolder, self).__init__(root, IMG_EXTENSIONS, split, transform=transform,
                                        target_transform=target_transform, transform_basic=transform_basic, filter_classes=classes)
        self.imgs = self.samples

    def find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def make_dataset(self, dir, split, classes):
        path = dir.split("/")[:-2]
        split_file = np.genfromtxt(f'{path[0]}/{path[1]}/additionals/{split}.txt', dtype='unicode')
        images = []

        for sample in split_file:
            path = str(sample[0])
            label = int(sample[1])

            if label not in classes:
                continue
            images.append((dir+path, label))

        return images

    def __len__(self):
        return len(self.samples)
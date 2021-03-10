from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader  # import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import argparse

# from torchvision import transforms, utils
import cv2
from PIL import Image

# Ignore warnings
import warnings

from torchvision.transforms import ToTensor

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


# setup ended

class OCR1to9Dataset(Dataset):
    # """Face Landmarks dataset."""

    def __init__(self, data_mode='train', noise_range=0, rotate_range=0, im_shape=(1, 28, 28)):
        # """
        # Args:
        #     data_mode (string): 'train' or 'test'
        #     noise = 0: option: noise = number ~ 1 - 3
        #     rotate = 0: option: rotate = number ~ 1 - 3
        #
        # """
        self.data_mode = data_mode
        self.path_list = []
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.noise_range = noise_range
        self.rotate_range = rotate_range
        self.im_shape = im_shape
        # fills the path list with the paths of all the figures
        self.find_figures()
        self.len = self.__len__()

    def find_figures(self):
        data_dir = os.path.join(self.root_dir, 'data', 'digit_fonts', self.data_mode)
        extension = '.png'
        for root, dirs, files in os.walk(data_dir):
            for file_ in files:
                if file_.endswith(extension):
                    self.path_list.append(os.path.join(root, file_))

    def show_image(self, idx):
        # read the image
        im = Image.open(self.path_list(idx))
        # show image
        im.show()

    def __len__(self):
        return len(self.path_list)

    def noise(self, sample):
        # row, col, ch = sample['image'].shape
        try:
            row, col = sample['image'].shape
        except:
            pass

        mean = 0
        var = 0.1
        sigma = var ** 0.5
        # gauss = np.random.normal(mean, sigma, (row, col, ch))
        # gauss = gauss.reshape(row, col, ch)
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        sample['image'] = sample['image'] + self.noise_range * gauss
        return sample

    def rotate(self, sample):
        image = sample['image']
        angle = self.rotate_range * np.random.normal()
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        sample['image'] = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_loc = self.path_list[idx]
        # image = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
        image = np.double(cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE))
        label = img_loc[-5]
        sample = {'image': image, 'label': int(label)}

        self.sample_normalization(sample)

        if self.noise:
            sample = self.noise(sample)
        if self.rotate:
            sample = self.rotate(sample)
        # shaping sample
        sample = self.reshape_image(sample)
        return sample

    def reshape_image(self, sample):

        im_shape = sample['image'].shape
        if np.prod(self.im_shape) == np.prod(im_shape):
            sample['image'] = np.resize(sample['image'], self.im_shape)
            return sample
        else:
            print("size isn't matching")


    @staticmethod
    def sample_normalization(sample):
        sample['image'] = sample['image'] - np.mean(sample['image'])  # mean zero
        if not np.std(sample['image']) == 0:
            sample['image'] = sample['image'] / np.std(sample['image'])  # STD one
        return sample

    @staticmethod
    def to_tensor(sample):
        # image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(sample['image']),
                'label': sample['label']}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, 1)   # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(30, 60, 3, 1)  # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8640, 128) # nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = sample_batched['image'], torch.tensor(sample_batched['label'])
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      # for data, target in test_loader:
        for batch_idx, sample_batched in enumerate(test_loader):
            data, target = sample_batched['image'], torch.tensor(sample_batched['label'])
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=60, metavar='N',         # default=64
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = OCR1to9Dataset(data_mode='train', noise_range=1, rotate_range=3)
    dataset2 = OCR1to9Dataset(data_mode='test', noise_range=1, rotate_range=3)

    train_loader = DataLoader(dataset1, batch_size=120,  # Training
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size=600,  # Testing
                             shuffle=True, num_workers=4)

    # net = net.double() ___________________________________________________________________________________________________
    model = Net().to(device)
    model = model.double()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "ocr_google_fonts_cnn.pt")


if __name__ == '__main__':

    main()

    # test dataset
    numbers_dataset = OCR1to9Dataset(data_mode='train', noise_range=1, rotate_range=3)
    dataloader = DataLoader(numbers_dataset, batch_size=15,
                            shuffle=True, num_workers=4)

    for idx in range(4):
        # sample = numbers_dataset.transformed_dataset(idx)
        sample = numbers_dataset.__getitem__(idx)
        img = sample['image']
        img = np.array(img)

        # normalization of the image to range 0 - 255
        min_img, max_img = np.min(img), np.max(img)
        img = np.multiply(img + min_img, (max_img - min_img) * 255)

        cv2.imshow(sample['label'], img)
        cv2.waitKey(1)



    cv2.waitKey(0)

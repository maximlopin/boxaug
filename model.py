import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num):
        super(ResidualBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num, num // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num // 2, num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return x + self.features(x)


class Model(nn.Module):
    """ 128x128 -> 16x16 """

    def __init__(self, num_anchors, num_classes=1):
        """
        Parameters:
            activation_fns: list of activation functions for each output layer
        """
        super(Model, self).__init__()

        if num_classes in [0, 1]:
            num_classes = 0

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            ResidualBlock(128),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            ResidualBlock(256),
           
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            
            ResidualBlock(512),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )
        self.features_head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=0), # 7x7 after this
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0), # 5x5 after this
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0), # 3x3 after this
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0), # 1x1 after this
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )
        self.detection_head = nn.Conv2d(1024, 3 + num_anchors + num_classes, 
                                        kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        latent = self.layers(x)

        features1x1 = self.features_head(latent)
        features16x16 = nn.functional.interpolate(features1x1, size=16, mode='nearest')
        y = self.detection_head(torch.cat((features16x16, latent), dim=1))
        y[:, :3, :, :].sigmoid_()
        return y


def focal_loss(y_hat, y, gamma=2.0):
    bce = F.binary_cross_entropy(y_hat, y.float(), reduction='none')
    pt = torch.exp(-bce)
    fl = (1 - pt) ** gamma * bce
    return fl.mean()


def multiclass_loss(y_hat, y, num_anchors, gamma=2.0):
    """
    Parameters:
        y_hat: tensor of shape (N, num_anchors * (3 + C), H, W)
        y: tensor of shape (N, num_anchors * 4, W, H)
        num_anchors: int
    """
    conf_loss = 0.0
    shift_loss = 0.0
    class_loss = 0.0

    for i in range(num_anchors):
        n = 4 * i

        only_conf = (y_hat[:, n, :, :] == 1.0).long()

        conf_loss += focal_loss(y_hat[:, n, :, :], y[:, n, :, :])
        shift_loss += only_conf * F.binary_cross_entropy(y_hat[:, n + 1:n + 3, :, :], y[:, n + 1:n + 3, :, :].float(), reduction='none')
        class_loss += only_conf * F.binary_cross_entropy(y_hat[:, n + 3:, :, :], y[:, n + 3:, :, :].float(), reduction='none')

    return conf_loss + shift_loss.mean() + class_loss.mean()


def single_class_loss(y_hat, y, num_anchors, gamma=2.0):
    """
    Parameters:
        y_hat: tensor of shape (N, num_anchors * 3, H, W)
        y: tensor of shape (N, 3, H, W)
    """
    conf_loss = 0.0
    shift_loss = 0.0

    for i in range(num_anchors):
        n = 3 * i

        only_conf = (y_hat[:, n, :, :] == 1.0).long()
    
        conf_loss += focal_loss(y_hat[:, n, :, :], y[:, n, :, :])
        shift_loss += only_conf * F.binary_cross_entropy(y_hat[:, n + 1:n + 3, :, :], y[:, n + 1:n + 3, :, :].float(), reduction='none')

    return conf_loss + shift_loss.mean()

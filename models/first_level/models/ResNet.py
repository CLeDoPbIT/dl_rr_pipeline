import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:,:,0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class ResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64
        layers = [1, 1, 1, 1]
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.maxpool3 = nn.AvgPool1d(kernel_size=12, stride=2, padding=0)


        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=9, stride=1, padding=0)


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=4, stride=1, padding=0)

        self.fc = nn.Linear(128*3, num_classes)

        self.batch_norm_1 = nn.BatchNorm1d(128*3)
        self.batch_norm_2 = nn.BatchNorm1d(128*2)
        self.batch_norm_3 = nn.BatchNorm1d(128*1)
        self.relu = nn.ReLU(True)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.fc_1 = nn.Linear(128*3, 128*2)
        self.fc_2 = nn.Linear(128*2, 128*1)
        self.fc_3 = nn.Linear(128*1, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)


    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)


    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = F.relu(self.bn1(x0))
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.maxpool3(x)

        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.maxpool5(y)

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.maxpool7(z)

        out = torch.cat([x,y,z], dim=1)
        out_conv = out.squeeze()

        out = self.batch_norm_1(out_conv)
        out = self.fc_1(out)
        out = F.relu(self.batch_norm_2(out))
        out = self.dropout_1(out)

        out = self.fc_2(out)
        out = F.relu(self.batch_norm_3(out))
        out = self.dropout_2(out)

        out = self.fc_3(out)
        out = self.softmax(out)
        return out

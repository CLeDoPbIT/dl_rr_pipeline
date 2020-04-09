import torch.nn as nn
import torch.nn.functional as F


class FcNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FcNet, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.batch_norm_3 = nn.BatchNorm1d(128)
        self.batch_norm_4 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.dropout_1 = nn.Dropout(p=0.8)
        self.dropout_2 = nn.Dropout(p=0.8)
        self.dropout_3 = nn.Dropout(p=0.8)
        self.fc_0 = nn.Linear(input_size, 128)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, batch_input):
        out = self.batch_norm(batch_input)
        out = self.fc_0(out)
        out = F.relu(self.batch_norm_2(out))
        out = self.dropout_1(out)

        out = self.fc_1(out)
        out = F.relu(self.batch_norm_3(out))
        out = self.dropout_2(out)

        out = self.fc_2(out)
        out = F.relu(self.batch_norm_4(out))
        out = self.dropout_3(out)

        out = self.fc_3(out)
        out = self.softmax(out)
        return out

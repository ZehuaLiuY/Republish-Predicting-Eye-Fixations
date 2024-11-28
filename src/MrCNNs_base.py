import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MrCNNs(nn.Module):
    def __init__(self, dropout = 0.5):
        super().__init__()

        # Convolutional layers for each branch
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(96, 160, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(160, 288, kernel_size=3, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.branch_fc = nn.Linear(2592, 512)
        self.fc_combined = nn.Linear(512 * 3, 512)
        self.output = nn.Linear(512, 1)

    def forward_branch(self, x, conv1, conv2, conv3):
        with torch.no_grad():
            norm = conv1.weight.norm(2, dim=(1, 2, 3), keepdim=True)
            desired_norm = torch.clamp(norm, max=0.1)
            conv1.weight = nn.Parameter(conv1.weight * desired_norm / (norm + 1e-8))

        x = F.relu(conv1(x))
        x = self.pool(x)

        x = F.relu(conv2(x))
        x = self.pool(x)

        x = F.relu(conv3(x))
        x = self.dropout(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.branch_fc(x))
        x = self.dropout(x)

        return x

    def forward(self, input1, input2, input3):

        branch1_output = self.forward_branch(input1, self.conv1, self.conv2, self.conv3)
        branch2_output = self.forward_branch(input2, self.conv1, self.conv2, self.conv3)
        branch3_output = self.forward_branch(input3, self.conv1, self.conv2, self.conv3)

        combined = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)
        combined = F.relu(self.fc_combined(combined))
        combined = self.dropout(combined)
        output = torch.sigmoid(self.output(combined))


        return output




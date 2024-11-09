import torch
import torch.nn as nn
import torch.nn.functional as F

class MrCNN(nn.Module):
    def __init__(self):
        super(MrCNN, self).__init__()
        ## Convolutional layers for each branch
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(96, 160, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(160, 288, kernel_size=3, stride=1, padding=0)

        ## layers for all branches
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch_fc = nn.Linear(3 * 3 * 288, 512)

        self.fc_combined = nn.Linear(512 * 3, 512)

        self.output = nn.Linear(512, 1)

    def forward_branch(self, x, conv1, conv2, conv3):
        x = F.relu(conv1(x))
        x = self.pool(x)
        x = F.relu(conv2(x))
        x = self.pool(x)
        x = F.relu(conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.sigmoid(self.branch_fc(x))
        return x

    def forward(self, input1, input2, input3):
        branch1_output = self.forward_branch(input1, self.conv1, self.conv2, self.conv3)
        branch2_output = self.forward_branch(input2, self.conv1, self.conv2, self.conv3)
        branch3_output = self.forward_branch(input3, self.conv1, self.conv2, self.conv3)

        combined = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)

        combined = F.relu(self.fc_combined(combined))

        output = torch.sigmoid(self.output(combined))
        return output

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
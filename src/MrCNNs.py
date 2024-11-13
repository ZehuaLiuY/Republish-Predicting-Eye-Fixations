import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout


class MrCNNs(nn.Module):
    def __init__(self):
        super().__init__()
        ## Convolutional layers for each branch
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(96, 160, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(160, 288, kernel_size=3, stride=1, padding=0)

        # initialise the layers
        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        self.initialise_layer(self.conv3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # the paper uses a dropout layer with a rate of 0.5
        # TODO: add this argument to parser to make it configurable
        self.droupout = nn.Dropout(0.5)

        self.branch_fc = nn.Linear(2592, 512)

        self.fc_combined = nn.Linear(512 * 3, 512)

        self.output = nn.Linear(512, 1)

    def forward_branch(self, x, conv1, conv2, conv3):
        # # Apply L2 norm constraint to the first convolutional layer
        # # reference: paper section 3.2
        with torch.no_grad():
            norm = conv1.weight.norm(2, dim=(1, 2, 3), keepdim=True)
            desired_norm = torch.clamp(norm, max=0.1)
            # conv1.weight *= desired_norm / (norm + 1e-8)
            new_weight = conv1.weight * desired_norm / (norm + 1e-8)
            conv1.weight = nn.Parameter(new_weight)

        x = F.relu(conv1(x))
        x = self.pool(x)
        x = F.relu(conv2(x))
        x = self.pool(x)
        x = F.relu(conv3(x))
        x = self.droupout(x)
        x = self.pool(x)
        # print(f'before view x.shape: {x.shape}')
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.branch_fc(x))
        # print(f'after view x.shape: {x.shape}')
        # add dropout layer after the FC layer
        x = self.droupout(x)
        return x

    def forward(self, input1, input2, input3):
        branch1_output = self.forward_branch(input1, self.conv1, self.conv2, self.conv3)
        branch2_output = self.forward_branch(input2, self.conv1, self.conv2, self.conv3)
        branch3_output = self.forward_branch(input3, self.conv1, self.conv2, self.conv3)

        combined = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)

        combined = F.relu(self.fc_combined(combined))
        combined = self.droupout(combined)

        output = torch.sigmoid(self.output(combined))
        return output

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MrCNNs(nn.Module):
    def __init__(self, first_batch_only=True, visualize=False):
        super().__init__()
        self.visualization_counter = 1
        self.first_batch_only = first_batch_only
        self.visualize = visualize
        self.first_batch_processed = False

        # Convolutional layers for each branch
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(96, 160, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(160, 288, kernel_size=3, stride=1, padding=0)

        # Initialize the layers
        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        self.initialise_layer(self.conv3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.branch_fc = nn.Linear(2592, 512)
        self.fc_combined = nn.Linear(512 * 3, 512)
        self.output = nn.Linear(512, 1)

    def forward_branch(self, x, original_input, conv1, conv2, conv3):
        with torch.no_grad():
            norm = conv1.weight.norm(2, dim=(1, 2, 3), keepdim=True)
            desired_norm = torch.clamp(norm, max=0.1)
            conv1.weight = nn.Parameter(conv1.weight * desired_norm / (norm + 1e-8))

        x = F.relu(conv1(x))
        if self.visualize and not self.first_batch_processed:
            self.visualize_feature_map(x, original_input, layer_name="conv1", num_feature_maps=3)
        x = self.pool(x)

        x = F.relu(conv2(x))
        if self.visualize and not self.first_batch_processed:
            self.visualize_feature_map(x, original_input, layer_name="conv2", num_feature_maps=3)
        x = self.pool(x)

        x = F.relu(conv3(x))
        if self.visualize and not self.first_batch_processed:
            self.visualize_feature_map(x, original_input, layer_name="conv3", num_feature_maps=3)
        x = self.dropout(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.branch_fc(x))
        x = self.dropout(x)

        return x

    def forward(self, input1, input2, input3):
        # Ensure the first batch of each epoch visualizes feature maps
        if self.first_batch_only and not self.first_batch_processed:
            self.visualize = True
            self.first_batch_processed = True
        else:
            self.visualize = False
            
        # 每个 epoch 重置 first_batch_processed 标志
        self.first_batch_processed = False
        branch1_output = self.forward_branch(input1, input1, self.conv1, self.conv2, self.conv3)
        branch2_output = self.forward_branch(input2, input2, self.conv1, self.conv2, self.conv3)
        branch3_output = self.forward_branch(input3, input3, self.conv1, self.conv2, self.conv3)

        self.first_batch_processed = True

        combined = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)
        combined = F.relu(self.fc_combined(combined))
        combined = self.dropout(combined)
        output = torch.sigmoid(self.output(combined))
        return output

    def visualize_feature_map(self, x, original_input, layer_name, num_feature_maps=3):
        if not self.visualize or (self.first_batch_only and self.first_batch_processed):
            return

        folder_path = f"../feature_maps/{layer_name}"
        os.makedirs(folder_path, exist_ok=True)

        x = (x - x.min()) / (x.max() - x.min())
        x = x[:3].detach().cpu().numpy()

        original_input = original_input[:3].detach().cpu().numpy()
        original_input = (original_input - original_input.min()) / (original_input.max() - original_input.min())

        num_feature_maps = min(num_feature_maps, x.shape[1])

        for sample_idx in range(3):
            fig, axes = plt.subplots(2, num_feature_maps, figsize=(20, 10))

            for i in range(num_feature_maps):
                axes[0, i].imshow(original_input[sample_idx].transpose(1, 2, 0))
                axes[0, i].axis('off')

            for i in range(num_feature_maps):
                axes[1, i].imshow(x[sample_idx, i], cmap='viridis')
                axes[1, i].axis('off')

            fig.suptitle(f"Original and Feature maps after {layer_name} - Sample {sample_idx}")
            plt.savefig(f"{folder_path}/{layer_name}_visualization_{self.visualization_counter}_sample_{sample_idx}.png")
            plt.close(fig)

        self.visualization_counter += 1

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
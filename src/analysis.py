import torch
from tqdm import tqdm

from MrCNNs import MrCNNs
from dataset import MIT
import matplotlib.pyplot as plt
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(state_dict, strict=False)
    return model


def analyze_feature_strength(model, val_dataset, save_dir="feature_strengths"):
    model.eval()
    activations = {"conv1": [], "conv2": [], "conv3": []}

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for index in tqdm(range(len(val_dataset)), desc="Analyzing feature strength"):
            img, _ = val_dataset[index]
            img_400_crop = img[0, :, :, :].unsqueeze(0).to(device)
            img_250_crop = img[1, :, :, :].unsqueeze(0).to(device)
            img_150_crop = img[2, :, :, :].unsqueeze(0).to(device)

            _, layer_activations = model(
                img_400_crop, img_250_crop, img_150_crop, return_activations=True
            )

            for layer in layer_activations:
                for key in activations.keys():
                    activations[key].append(layer[key])

    for key in activations.keys():
        activations[key] = sum(activations[key]) / len(activations[key])
        print(f"Average activation for {key}: {activations[key]}")

    activation_path = os.path.join(save_dir, "activations.json")
    with open(activation_path, "w") as f:
        json.dump(activations, f)
    print(f"Activations saved to {activation_path}")

    plot_feature_strength(activations, save_dir)

def plot_feature_strength(activations, save_dir):
    layers = list(activations.keys())
    strengths = list(activations.values())

    plt.bar(layers, strengths, color='skyblue')
    plt.title("Average Feature Strength by Layer")
    plt.xlabel("Layer")
    plt.ylabel("Average Activation Value")

    save_path = os.path.join(save_dir, "feature_strength.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature strength plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # 路径设置
    checkpoint_path = "./pre_trained_models/MrCNNsfinal_model.pth"
    val_dataset_path = "../dataset/val_data.pth.tar"

    val_dataset = MIT(val_dataset_path)

    model = MrCNNs()

    model = load_model(checkpoint_path, model)

    analyze_feature_strength(model, val_dataset, save_dir="feature_strengths")

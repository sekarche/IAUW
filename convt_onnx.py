import os
import torch
import onnx
import torch.nn as nn
import torch.onnx
import argparse
from torchvision import transforms
from collections import OrderedDict

# --- Model Definitions ---

class NovelUnderwaterImageAssessmentModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.raw_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.quality_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.degradation_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, raw_img, ref_img=None):
        features = self.raw_features(raw_img)
        quality = self.quality_head(features)
        degradation = self.degradation_head(features)
        return quality, degradation


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DeblurModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class LowLightEnhancementNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=5):
        super().__init__()
        layers = [
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return torch.clamp(x - noise, 0, 1)

# --- Utility Functions ---

def load_model_weights(model, weights_path):
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        if "module." in list(state_dict.keys())[0]:
            state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        model.load_state_dict(state_dict)
        print(f"✅ Loaded weights from {weights_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading weights from {weights_path}: {e}")
        return False

def convert_to_onnx(model, model_name, output_dir, input_shape=(1, 3, 224, 224), dynamic=True):
    dummy_input = torch.randn(*input_shape)
    output_path = os.path.join(output_dir, f"{model_name}_lite.onnx")

    if "assessment" in model_name.lower():
        dummy_ref = torch.randn(*input_shape)
        try:
            torch.onnx.export(
                model,
                (dummy_input, dummy_ref),
                output_path,
                verbose=False,
                input_names=["input", "reference"],
                output_names=["quality", "degradation"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "reference": {0: "batch_size"},
                    "quality": {0: "batch_size"},
                    "degradation": {0: "batch_size"}
                } if dynamic else {},
                opset_version=11
            )
        except Exception as e:
            print(f"⚠️ Failed dual-input export: {e}, falling back to single input")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                verbose=False,
                input_names=["input"],
                output_names=["quality", "degradation"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "quality": {0: "batch_size"},
                    "degradation": {0: "batch_size"}
                } if dynamic else {},
                opset_version=11
            )
    else:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            } if dynamic else {},
            opset_version=11
        )

    print(f"✅ Saved ONNX model to {output_path}")

    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model {model_name}_lite.onnx is valid")
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")

# --- Main Conversion Logic ---

def main():
    parser = argparse.ArgumentParser(description="Convert UWA models to ONNX")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Path to .pth model weights")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output path for ONNX models")
    parser.add_argument("--size", type=int, default=224, help="Input image size")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic batch size in ONNX")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_configs = [
        {"name": "assessment_model", "class": NovelUnderwaterImageAssessmentModel, "kwargs": {"num_classes": 4}},
        {"name": "color_correction", "class": UNet, "kwargs": {"in_channels": 3, "out_channels": 3}},
        {"name": "deblur", "class": DeblurModel, "kwargs": {"in_channels": 3, "out_channels": 3}},
        {"name": "lowlight", "class": LowLightEnhancementNet, "kwargs": {"in_channels": 3, "out_channels": 3}},
        {"name": "denoise", "class": DnCNN, "kwargs": {"channels": 3, "num_layers": 5}},
    ]

    for cfg in model_configs:
        model = cfg["class"](**cfg["kwargs"])
        model.eval()

        weights_path = os.path.join(args.weights_dir, f"{cfg['name']}.pth")
        if not load_model_weights(model, weights_path):
            print(f"⚠️ Using random weights for {cfg['name']}")

        input_shape = (1, 3, args.size, args.size)
        convert_to_onnx(model, cfg["name"], args.output_dir, input_shape, args.dynamic)

    print("\n🎉 All models converted to ONNX successfully.")

if __name__ == "__main__":
    main()

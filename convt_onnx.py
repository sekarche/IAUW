import os
import torch
import onnx
import torch.nn as nn
import torch.onnx
import argparse
from torchvision.models import resnet18
from collections import OrderedDict

# --- Correct Assessment Model with ResNet18 backbone ---
class AssessmentResNetModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.quality_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.degradation_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, ref=None):
        features = self.backbone(x)
        quality = self.quality_head(features)
        degradation = self.degradation_head(features)
        return quality, degradation

# Enhancement models
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
        return self.decoder(self.encoder(x))

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
        layers = [nn.Conv2d(channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(x - self.dncnn(x), 0, 1)

# Utility: load weights

def load_model_weights(model, weights_path):
    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if "module." in list(state_dict.keys())[0]:
            state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        model.load_state_dict(state_dict)
        print(f"✅ Loaded weights from {weights_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading weights from {weights_path}: {e}")
        return False

# Utility: convert to ONNX

def convert_to_onnx(model, model_name, output_dir, input_shape=(1, 3, 224, 224), dynamic=True):
    dummy_input = torch.randn(*input_shape)
    output_path = os.path.join(output_dir, f"{model_name}_lite.onnx")
    model.eval()

    if "assessment" in model_name:
        dummy_ref = torch.randn(*input_shape)
        try:
            torch.onnx.export(model, (dummy_input, dummy_ref), output_path,
                              input_names=["input", "reference"],
                              output_names=["quality", "degradation"],
                              dynamic_axes={"input": {0: "batch"}, "reference": {0: "batch"},
                                            "quality": {0: "batch"}, "degradation": {0: "batch"}} if dynamic else {},
                              opset_version=11)
        except:
            torch.onnx.export(model, dummy_input, output_path,
                              input_names=["input"],
                              output_names=["quality", "degradation"],
                              dynamic_axes={"input": {0: "batch"}} if dynamic else {},
                              opset_version=11)
    else:
        torch.onnx.export(model, dummy_input, output_path,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else {},
                          opset_version=11)

    print(f"✅ Saved ONNX: {output_path}")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model {model_name}_lite.onnx is valid")
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")

# Main entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, default="./weights")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_list = [
        ("assessment_model", AssessmentResNetModel(num_classes=4)),
        ("color_correction", UNet()),
        ("deblur", DeblurModel()),
        ("lowlight", LowLightEnhancementNet()),
        ("denoise", DnCNN())
    ]

    for name, model in model_list:
        weight_path = os.path.join(args.weights_dir, f"{name}.pth")
        if os.path.exists(weight_path):
            load_model_weights(model, weight_path)
        else:
            print(f"⚠️  No weights for {name}, using random init")

        convert_to_onnx(model, name, args.output_dir, (1, 3, args.size, args.size), args.dynamic)

    print("\n🎉 All models converted!")

if __name__ == "__main__":
    main()

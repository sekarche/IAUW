import os
import torch
import onnx
import torch.nn as nn
import torch.onnx
import numpy as np
from torchvision import transforms
import argparse
from PIL import Image

# Import  model classes.


class NovelUnderwaterImageAssessmentModel(nn.Module):
    def __init__(self, num_classes=4):
        super(NovelUnderwaterImageAssessmentModel, self).__init__()
        # Creating a simplified version for ONNX export
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
        
        # Quality Head
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
        
        # Degradation Head
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
    # Simplified version of UNet for ONNX export
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
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
        super(DeblurModel, self).__init__()
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
        super(LowLightEnhancementNet, self).__init__()
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
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))
        
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        noise = self.dncnn(x)
        return torch.clamp(x - noise, 0, 1)

def load_model_weights(model, weights_path):
    """Load model weights from file"""
    try:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        # Handle model saved with DataParallel
        if "module." in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
        return True
    except Exception as e:
        print(f"Error loading weights from {weights_path}: {e}")
        return False

def convert_to_onnx(model, model_name, input_shape=(1, 3, 224, 224), dynamic=True):
    """Convert PyTorch model to ONNX format"""
    print(f"Converting {model_name} to ONNX...")
    
    dummy_input = torch.randn(*input_shape)
    
    if "assessment" in model_name.lower():
        # Assessment model might have two inputs (raw and reference)
        dummy_ref = torch.randn(*input_shape)
        try:
            torch.onnx.export(
                model,
                (dummy_input, dummy_ref),
                f"{model_name}_lite.onnx",
                verbose=False,
                input_names=['input', 'reference'],
                output_names=['quality', 'degradation'],
                dynamic_axes={
                    'input': {0: 'batch_size'} if dynamic else {},
                    'reference': {0: 'batch_size'} if dynamic else {},
                    'quality': {0: 'batch_size'} if dynamic else {},
                    'degradation': {0: 'batch_size'} if dynamic else {}
                },
                opset_version=11
            )
        except Exception as e:
            print(f"Error exporting with reference input: {e}")
            # Fallback to single input export
            torch.onnx.export(
                model,
                dummy_input,
                f"{model_name}_lite.onnx",
                verbose=False,
                input_names=['input'],
                output_names=['quality', 'degradation'],
                dynamic_axes={
                    'input': {0: 'batch_size'} if dynamic else {},
                    'quality': {0: 'batch_size'} if dynamic else {},
                    'degradation': {0: 'batch_size'} if dynamic else {}
                },
                opset_version=11
            )
    else:
        # Enhancement models have a single input
        torch.onnx.export(
            model,
            dummy_input,
            f"{model_name}_lite.onnx",
            verbose=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'} if dynamic else {},
                'output': {0: 'batch_size'} if dynamic else {}
            },
            opset_version=11
        )
    
    print(f"Saved {model_name}_lite.onnx")
    
    # Verify the model
    try:
        onnx_model = onnx.load(f"{model_name}_lite.onnx")
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model {model_name}_lite.onnx is valid")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert underwater enhancement PyTorch models to ONNX format")
    parser.add_argument("--weights_dir", type=str, default="./weights",
                        help="Directory containing PyTorch model weights (.pth files)")
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="Output directory for ONNX models")
    parser.add_argument("--size", type=int, default=224,
                        help="Input image size for models")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic axes for batch dimension")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Change working directory to output directory
    os.chdir(args.output_dir)
    
    # Model configurations
    model_configs = [
        {
            "name": "assessment_model",
            "class": NovelUnderwaterImageAssessmentModel,
            "kwargs": {"num_classes": 4}
        },
        {
            "name": "color_correction",
            "class": UNet,
            "kwargs": {"in_channels": 3, "out_channels": 3}
        },
        {
            "name": "deblur",
            "class": DeblurModel,
            "kwargs": {"in_channels": 3, "out_channels": 3}
        },
        {
            "name": "lowlight",
            "class": LowLightEnhancementNet,
            "kwargs": {"in_channels": 3, "out_channels": 3}
        },
        {
            "name": "denoise",
            "class": DnCNN,
            "kwargs": {"channels": 3, "num_layers": 5}
        }
    ]
    
    # Convert each model
    for config in model_configs:
        model_name = config["name"]
        model_class = config["class"]
        model_kwargs = config["kwargs"]
        
        # Initialize model
        model = model_class(**model_kwargs)
        
        # Load weights if they exist
        weights_path = os.path.join(args.weights_dir, f"{model_name}.pth")
        if os.path.exists(weights_path):
            load_model_weights(model, weights_path)
        else:
            print(f"Warning: Weights not found at {weights_path}. Using random weights.")
        
        # Set model to evaluation mode
        model.eval()
        
        # Convert to ONNX
        input_shape = (1, 3, args.size, args.size)
        convert_to_onnx(model, model_name, input_shape, args.dynamic)
    
    print("Conversion completed!")

if __name__ == "__main__":
    main()

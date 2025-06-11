import torch
import torch.nn as nn

class LiteDnCNN(nn.Module):
    def __init__(self, depth=7):
        super().__init__()
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(depth-2):
            layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(32))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer
        layers.append(nn.Conv2d(32, 3, kernel_size=3, padding=1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.clamp(x - self.net(x), 0, 1)
    
    def quantize(self):
        self.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(self, inplace=True)
        torch.quantization.convert(self, inplace=True)
        return self

def export_denoise_model():
    model = LiteDnCNN().eval()
    sample = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        sample,
        "denoise_lite.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Denoising model exported")

if __name__ == "__main__":
    export_denoise_model()
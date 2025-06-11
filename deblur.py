import torch
import torch.nn as nn

class LiteDeblur(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Reconstruction
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.reconstruct(features)
    
    def quantize(self):
        self.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(self, inplace=True)
        torch.quantization.convert(self, inplace=True)
        return self

def export_deblur_model():
    model = LiteDeblur().eval()
    sample = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        sample,
        "deblur_lite.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Deblur model exported")

if __name__ == "__main__":
    export_deblur_model()
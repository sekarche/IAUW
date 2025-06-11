import torch
import torch.nn as nn

class LiteLowLight(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Processing
        self.process = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU()
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        
        # Process
        processed = self.process(enc2)
        
        # Decoder
        dec2 = self.dec2(processed)
        dec1 = self.dec1(dec2 + enc1)
        
        return self.output(dec1)
    
    def quantize(self):
        self.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(self, inplace=True)
        torch.quantization.convert(self, inplace=True)
        return self

def export_lowlight_model():
    model = LiteLowLight().eval()
    sample = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        sample,
        "lowlight_lite.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Low light model exported")

if __name__ == "__main__":
    export_lowlight_model()
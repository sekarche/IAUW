import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import cv2
from PIL import Image
import random
import glob
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Device configuration - use CPU for Raspberry Pi
device = torch.device("cpu")
print(f"Using device: {device}")

# Simplified Attention Fusion Module
class LiteAttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, raw_features, ref_features):
        attention_map = self.attention(torch.cat([raw_features, ref_features], dim=1))
        return raw_features * attention_map + ref_features

# Optimized Dual-Stream Model
class LiteUnderwaterImageAssessmentModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Use MobileNetV2 backbones for efficiency
        self.raw_stream = models.mobilenet_v2(pretrained=True).features
        self.ref_stream = models.mobilenet_v2(pretrained=True).features
        
        # Attention fusion
        self.fusion = LiteAttentionFusion(1280)
        
        # Shared feature processing
        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3)
        )
        
        # Quality Head
        self.quality_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Degradation Head
        self.degradation_head = nn.Linear(1280, num_classes)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, raw_img, ref_img=None):
        raw_features = self.raw_stream(raw_img)
        
        if ref_img is not None:
            ref_features = self.ref_stream(ref_img)
            fused_features = self.fusion(raw_features, ref_features)
        else:
            fused_features = raw_features
        
        shared_features = self.shared(fused_features)
        quality = self.quality_head(shared_features).squeeze()
        degradation = self.degradation_head(shared_features)
        
        return quality, degradation

    def quantize(self):
        """Quantize the model for efficient inference"""
        self.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(self, inplace=True)
        # Calibration would happen here with representative data
        torch.quantization.convert(self, inplace=True)
        return self

# Optimized Dataset Class
class LiteUnderwaterImageDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, transform=None, is_train=True):
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.transform = transform
        self.is_train = is_train
        
        # Pair raw and reference images
        self.pairs = []
        for ref_file in os.listdir(ref_dir):
            if ref_file.lower().endswith(('.jpg', '.png')):
                raw_path = os.path.join(raw_dir, ref_file)
                if os.path.exists(raw_path):
                    self.pairs.append((raw_path, os.path.join(ref_dir, ref_file)))
        
        # Simple quality scores based on degradation presence
        self.quality_scores = [0.85] * len(self.pairs)  # Default score

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_path, ref_path = self.pairs[idx]
        
        # Load images with PIL (more memory efficient)
        raw_img = Image.open(raw_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')
        
        if self.transform:
            if self.is_train and random.random() > 0.5:
                raw_img = transforms.functional.hflip(raw_img)
                ref_img = transforms.functional.hflip(ref_img)
            raw_img = self.transform(raw_img)
            ref_img = self.transform(ref_img)
        
        # For assessment model, we don't need degradation labels in deployment
        quality_score = torch.tensor(self.quality_scores[idx], dtype=torch.float32)
        
        return raw_img, ref_img, quality_score

# Training Function (simplified)
def train_model(model, train_loader, val_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, ref_images, quality_scores in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images, ref_images = images.to(device), ref_images.to(device)
            quality_scores = quality_scores.to(device)
            
            optimizer.zero_grad()
            pred_quality, _ = model(images, ref_images)
            loss = criterion(pred_quality, quality_scores)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss, pearson_corr = validate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Pearson: {pearson_corr:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_lite.pth')
    
    return model

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    pred_quality, true_quality = [], []
    
    with torch.no_grad():
        for images, ref_images, q_scores in val_loader:
            images, ref_images = images.to(device), ref_images.to(device)
            q_scores = q_scores.to(device)
            
            q_pred, _ = model(images, ref_images)
            loss = criterion(q_pred, q_scores)
            
            val_loss += loss.item()
            pred_quality.extend(q_pred.cpu().numpy())
            true_quality.extend(q_scores.cpu().numpy())
    
    val_loss /= len(val_loader)
    pearson_corr, _ = pearsonr(true_quality, pred_quality)
    return val_loss, pearson_corr

# Export to ONNX for deployment
def export_to_onnx(model, sample_input, filename="assessment_model.onnx"):
    torch.onnx.export(
        model,
        sample_input,
        filename,
        opset_version=11,
        input_names=['raw_image', 'ref_image'],
        output_names=['quality_score', 'degradation_type'],
        dynamic_axes={
            'raw_image': {0: 'batch_size'},
            'ref_image': {0: 'batch_size'},
            'quality_score': {0: 'batch_size'},
            'degradation_type': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {filename}")

# Main execution for deployment
def deploy_model():
    # Load model
    model = LiteUnderwaterImageAssessmentModel().to(device)
    model.load_state_dict(torch.load('best_model_lite.pth', map_location=device))
    model.eval()
    
    # Quantize for Raspberry Pi
    model = model.quantize()
    
    # Prepare sample input for ONNX export
    sample_input = (
        torch.randn(1, 3, 224, 224).to(device),
        torch.randn(1, 3, 224, 224).to(device)
    )
    
    # Export to ONNX
    export_to_onnx(model, sample_input)
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return model, transform

if __name__ == "__main__":
    # For training (would need dataset paths)
    # model = run_training()
    
    # For deployment
    model, transform = deploy_model()
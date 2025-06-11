import torch
import sys

# --- Assessment Model ---
from assessment import LiteUnderwaterImageAssessmentModel

def export_assessment():
    model = LiteUnderwaterImageAssessmentModel()
    model.load_state_dict(torch.load("../model-points/best_underwater_assessment_model.pth", map_location="cpu"))
    model.eval()
    # Model expects two images: raw_img, ref_img (both shape [1, 3, 224, 224])
    dummy_raw = torch.randn(1, 3, 224, 224)
    dummy_ref = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, (dummy_raw, dummy_ref),
        "../models/assessment_model_lite.onnx",
        input_names=['raw_image', 'ref_image'],
        output_names=['quality_score', 'degradation_type'],
        opset_version=11,
        dynamic_axes={
            'raw_image': {0: 'batch_size'},
            'ref_image': {0: 'batch_size'},
            'quality_score': {0: 'batch_size'},
            'degradation_type': {0: 'batch_size'},
        }
    )
    print("Assessment model exported to ../models/assessment_model_lite.onnx")

# --- Deblur Model ---
from deblur import LiteDeblur

def export_deblur():
    model = LiteDeblur()
    model.load_state_dict(torch.load("../model-points/deblur.pth", map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy,
        "../models/deblur_lite.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Deblur model exported to ../models/deblur_lite.onnx")

# --- Denoise Model ---
from denoise import LiteDnCNN

def export_denoise():
    model = LiteDnCNN()
    model.load_state_dict(torch.load("../model-points/denoising.pth", map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy,
        "../models/denoise_lite.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Denoising model exported to ../models/denoise_lite.onnx")

# --- Low Light Model ---
from low_light_enhancement import LiteLowLight

def export_lowlight():
    model = LiteLowLight()
    model.load_state_dict(torch.load("../model-points/lowligh.pth", map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy,
        "../models/lowlight_lite.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Low light model exported to ../models/lowlight_lite.onnx")

if __name__ == "__main__":
    export_assessment()
    export_deblur()
    export_denoise()
    export_lowlight()

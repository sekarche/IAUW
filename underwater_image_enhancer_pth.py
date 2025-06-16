
import os
import numpy as np
import cv2
import time
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from inference.pytorch_to_ONNX_converter import (
    NovelUnderwaterImageAssessmentModel,
    UNet, DeblurModel, DnCNN, LowLightEnhancementNet,
    load_model_weights
)

class UnderwaterImageEnhancer:
    def __init__(self, model_dir="./weights", device="cpu"):
        self.model_dir = model_dir
        self.device = torch.device(device)

        print("Loading .pth models...")
        start = time.time()

        self.assessment_model = NovelUnderwaterImageAssessmentModel(num_classes=4).to(self.device)
        load_model_weights(self.assessment_model, os.path.join(model_dir, "assessment_model.pth"))
        self.assessment_model.eval()

        self.color_model = UNet().to(self.device)
        load_model_weights(self.color_model, os.path.join(model_dir, "color_correction.pth"))
        self.color_model.eval()

        self.deblur_model = DeblurModel().to(self.device)
        load_model_weights(self.deblur_model, os.path.join(model_dir, "deblur.pth"))
        self.deblur_model.eval()

        self.lowlight_model = LowLightEnhancementNet().to(self.device)
        load_model_weights(self.lowlight_model, os.path.join(model_dir, "lowlight.pth"))
        self.lowlight_model.eval()

        self.denoise_model = DnCNN().to(self.device)
        load_model_weights(self.denoise_model, os.path.join(model_dir, "denoise.pth"))
        self.denoise_model.eval()

        print(f"Models loaded in {time.time() - start:.2f} seconds")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            img_tensor = self.transform(image).unsqueeze(0)
            return img_tensor, original_size
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None, None

    def assess_image(self, image_tensor, ref_tensor=None):
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                ref_tensor = ref_tensor.to(self.device) if ref_tensor is not None else None

                if ref_tensor is not None:
                    quality, degradation = self.assessment_model(image_tensor, ref_tensor)
                else:
                    quality, degradation = self.assessment_model(image_tensor)

                quality_score = quality.item()
                degradation_type = torch.argmax(degradation, dim=1).item()

                return quality_score, degradation_type
        except Exception as e:
            print(f"Assessment error: {e}")
            return 0.5, 0

    def enhance_image(self, image_tensor, degradation_type):
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)

                if degradation_type == 0:
                    model = self.color_model
                    model_name = "Color Correction"
                elif degradation_type == 1:
                    model = self.deblur_model
                    model_name = "Deblur"
                elif degradation_type == 2:
                    model = self.lowlight_model
                    model_name = "Low Light"
                else:
                    model = self.denoise_model
                    model_name = "Denoise"

                print(f"Applying {model_name} enhancement...")
                enhanced_tensor = model(image_tensor)
                return enhanced_tensor.cpu()
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image_tensor.cpu()

    def postprocess_image(self, enhanced_tensor, original_size):
        try:
            enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
            if original_size:
                enhanced_np = cv2.resize(enhanced_np, (original_size[0], original_size[1]))
            return enhanced_np
        except Exception as e:
            print(f"Postprocessing error: {e}")
            return None

    def process_image(self, input_path, output_path=None, ref_path=None):
        start_time = time.time()
        print(f"Processing image: {input_path}")

        image_tensor, original_size = self.preprocess_image(input_path)
        if image_tensor is None:
            return None

        ref_tensor = None
        if ref_path:
            ref_tensor, _ = self.preprocess_image(ref_path)

        quality, degradation_type = self.assess_image(image_tensor, ref_tensor)
        degradation_names = ["color cast", "blur", "low light", "noise"]
        print(f"Assessment: Quality score: {quality:.2f}, Degradation: {degradation_names[degradation_type]}")

        enhanced_tensor = self.enhance_image(image_tensor, degradation_type)
        enhanced_image = self.postprocess_image(enhanced_tensor, original_size)

        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
            print(f"Enhanced image saved to {output_path}")

        print(f"Processing completed in {time.time() - start_time:.2f} seconds")
        return enhanced_image

    def process_directory(self, input_dir, output_dir=None):
        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist")
            return

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        count = 0

        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"enhanced_{filename}") if output_dir else None
                self.process_image(input_path, output_path)
                count += 1

        print(f"Processed {count} images")

def main():
    parser = argparse.ArgumentParser(description="Underwater Image Enhancement using .pth models")
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output", type=str, help="Output path or directory (optional)")
    parser.add_argument("--reference", type=str, help="Reference image path (optional)")
    parser.add_argument("--models", type=str, default="./weights", help="Directory containing .pth weights")

    args = parser.parse_args()
    enhancer = UnderwaterImageEnhancer(model_dir=args.models)

    if os.path.isdir(args.input):
        enhancer.process_directory(args.input, args.output)
    else:
        enhancer.process_image(args.input, args.output, args.reference)

if __name__ == "__main__":
    main()

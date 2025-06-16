
import os
import time
import torch
import argparse
import platform
import psutil
import csv
import cv2
from datetime import datetime
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from models.resnet_assessment_model import ResNetAssessmentModel
from models.color_correction_model import ColorCorrectionModel
from models.deblur_model import DeblurModel
from models.denoise_model import DenoiseModel
from models.lowlight_model import LowLightEnhancementNet

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return None

def enhance_image(model, img_tensor):
    with torch.no_grad():
        return model(img_tensor.unsqueeze(0)).squeeze(0)

def image_to_tensor(image):
    image = cv2.resize(image, (256, 256))  # Ensure uniform input size
    image = image.astype('float32') / 255.0
    return torch.tensor(image.transpose(2, 0, 1))

def tensor_to_image(tensor):
    img = tensor.detach().numpy().transpose(1, 2, 0)
    img = (img * 255.0).clip(0, 255).astype('uint8')
    return img

def calculate_metrics(original, enhanced):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    ssim = ssim_metric(original_gray, enhanced_gray)
    psnr = psnr_metric(original_gray, enhanced_gray)
    return ssim, psnr

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(args):
    ensure_dir(args.output)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    assessment_model = ResNetAssessmentModel().to(device)
    color_model = ColorCorrectionModel().to(device)
    deblur_model = DeblurModel().to(device)
    denoise_model = DenoiseModel().to(device)
    lowlight_model = LowLightEnhancementNet().to(device)

    # Load weights
    def load_weights(model, path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded weights: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    load_weights(assessment_model, os.path.join(args.models, "assessment_model.pth"))
    load_weights(color_model, os.path.join(args.models, "color_correction.pth"))
    load_weights(deblur_model, os.path.join(args.models, "deblur.pth"))
    load_weights(denoise_model, os.path.join(args.models, "denoise.pth"))
    load_weights(lowlight_model, os.path.join(args.models, "lowlight.pth"))

    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.png'))]
    results = []

    for img_name in image_files:
        img_path = os.path.join(args.input, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        tensor = image_to_tensor(img).to(device)

        # Quality Assessment
        with torch.no_grad():
            out = assessment_model(tensor.unsqueeze(0))
            quality_score = out['quality'].item()
            degradation = torch.argmax(out['degradation']).item()
            degradation_type = ["color cast", "blur", "noise", "low light"][degradation]

        # Enhance (only one enhancement applied based on degradation)
        if degradation_type == "color cast":
            enhanced_tensor = enhance_image(color_model, tensor)
        elif degradation_type == "blur":
            enhanced_tensor = enhance_image(deblur_model, tensor)
        elif degradation_type == "noise":
            enhanced_tensor = enhance_image(denoise_model, tensor)
        else:
            enhanced_tensor = enhance_image(lowlight_model, tensor)

        enhanced_image = tensor_to_image(enhanced_tensor)
        output_path = os.path.join(args.output, f"enhanced_{img_name}")
        cv2.imwrite(output_path, enhanced_image)

        # Metrics
        ssim, psnr = calculate_metrics(img, enhanced_image)

        # System stats
        cpu_temp = get_cpu_temp()
        cpu_usage = psutil.cpu_percent(interval=0.2)
        ram_usage = psutil.virtual_memory().percent

        results.append([
            img_name, quality_score, degradation_type, ssim, psnr,
            cpu_temp, cpu_usage, ram_usage, output_path
        ])

        print(f"Processed: {img_name} | SSIM: {ssim:.3f} | PSNR: {psnr:.2f}")

    # Write to CSV
    with open(args.csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Quality Score", "Degradation Type", "SSIM", "PSNR",
            "CPU Temp (°C)", "CPU Usage (%)", "RAM Usage (%)", "Output Path"
        ])
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--csv", default="results.csv")
    args = parser.parse_args()
    main(args)

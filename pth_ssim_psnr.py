
import os
import time
import cv2
import torch
import psutil
import csv
from datetime import datetime
from flask import Flask
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from models import (
    ResNetAssessmentModel,
    UNet,
    DeblurModel,
    LowLightEnhancementNet,
    DnCNN
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_ssim_psnr(input_path, output_path):
    img1 = cv2.imread(input_path)
    img2 = cv2.imread(output_path)
    if img1 is None or img2 is None:
        return None, None
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    ssim = ssim_metric(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    psnr = psnr_metric(img1, img2)
    return round(ssim, 4), round(psnr, 2)

def load_model_weights(model, weight_path):
    try:
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {weight_path}")
    except Exception as e:
        print(f"Error loading weights from {weight_path}: {e}")

def enhance_images(input_dir, output_dir, model_dir, csv_path="results.csv"):
    assessment_model = ResNetAssessmentModel().to(device)
    color_model = UNet().to(device)
    deblur_model = DeblurModel().to(device)
    lowlight_model = LowLightEnhancementNet().to(device)
    denoise_model = DnCNN().to(device)

    load_model_weights(assessment_model, os.path.join(model_dir, "assessment_model.pth"))
    load_model_weights(color_model, os.path.join(model_dir, "color_correction.pth"))
    load_model_weights(deblur_model, os.path.join(model_dir, "deblur.pth"))
    load_model_weights(lowlight_model, os.path.join(model_dir, "lowlight.pth"))
    load_model_weights(denoise_model, os.path.join(model_dir, "denoise.pth"))

    assessment_model.eval()
    color_model.eval()
    deblur_model.eval()
    lowlight_model.eval()
    denoise_model.eval()

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for img_file in sorted(os.listdir(input_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        input_path = os.path.join(input_dir, img_file)
        image = cv2.imread(input_path)
        if image is None:
            continue
        print(f"Processing image: {img_file}")

        # Dummy assessment - replace with actual model inference
        quality_score = 0.5
        degradation_type = "color cast"
        enhanced_img = image.copy()

        # Apply dummy enhancement - replace with actual model call
        output_path = os.path.join(output_dir, f"enhanced_{img_file}")
        cv2.imwrite(output_path, enhanced_img)

        ssim, psnr = compute_ssim_psnr(input_path, output_path)

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_image": img_file,
            "output_image": os.path.basename(output_path),
            "quality_score": quality_score,
            "degradation": degradation_type,
            "enhancement": "Color Correction",
            "ssim": ssim,
            "psnr": psnr,
            "cpu_temp": get_cpu_temp(),
            "cpu_usage": psutil.cpu_percent()
        }
        results.append(result)
        print(f"Enhanced and logged: {img_file}")

    write_csv(csv_path, results)

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return round(int(f.read()) / 1000.0, 2)
    except:
        return None

def write_csv(csv_path, results):
    fieldnames = list(results[0].keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--output", required=True, help="Output image directory")
    parser.add_argument("--models", required=True, help="Model weights directory")
    parser.add_argument("--csv", default="results.csv", help="CSV output path")
    args = parser.parse_args()

    start = time.time()
    enhance_images(args.input, args.output, args.models, args.csv)
    print(f"Processing complete in {round(time.time() - start, 2)} seconds.")

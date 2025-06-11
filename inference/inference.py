import os
import numpy as np
import cv2
import time
import argparse
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torchvision import transforms

class UnderwaterImageEnhancer:
    """Optimized Underwater Image Enhancement for Raspberry Pi"""
    
    def __init__(self, model_dir="./models", device="CPU"):
        """Initialize the enhancer with paths to ONNX models"""
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Set up ONNX runtime options for Raspberry Pi
        # These options can be adjusted based on your specific Pi model
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4  # Adjust based on your Pi's CPU cores
        
        # Set execution providers (CPU for Raspberry Pi)
        self.execution_providers = ['CPUExecutionProvider']
        
        # Load models
        print("Loading models...")
        start_time = time.time()
        
        # Load assessment model
        self.assessment_model = self._load_model("assessment_model_lite.onnx", session_options)
        
        # Load enhancement models
        self.color_model = self._load_model("color_correction_lite.onnx", session_options)
        self.deblur_model = self._load_model("deblur_lite.onnx", session_options)
        self.lowlight_model = self._load_model("lowlight_lite.onnx", session_options)
        self.denoise_model = self._load_model("denoise_lite.onnx", session_options)
        
        print(f"Models loaded in {time.time() - start_time:.2f} seconds")
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_name, session_options):
        """Load an ONNX model with error handling"""
        model_path = os.path.join(self.model_dir, model_name)
        
        try:
            if os.path.exists(model_path):
                return ort.InferenceSession(
                    model_path, 
                    sess_options=session_options,
                    providers=self.execution_providers
                )
            else:
                print(f"Warning: Model {model_path} not found.")
                return None
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image from file path"""
        try:
            # Open with PIL for consistent processing
            image = Image.open(image_path).convert('RGB')
            
            # Save original dimensions for later resizing
            original_size = image.size
            
            # Transform image for model input
            img_tensor = self.transform(image).unsqueeze(0)
            
            return img_tensor, original_size
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None, None
    
    def _run_onnx_inference(self, model, input_data, input_name=None):
        """Run inference with an ONNX model"""
        if model is None:
            print("Model not loaded")
            return None
        
        try:
            # Get input name if not provided
            if input_name is None:
                input_name = model.get_inputs()[0].name
            
            # Convert input to appropriate format if needed
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.cpu().numpy()
            
            # Run inference
            outputs = model.run(None, {input_name: input_data})
            return outputs
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def assess_image(self, image_tensor, ref_tensor=None):
        """Assess image quality and degradation type using ONNX model"""
        try:
            # Convert image tensor to numpy for ONNX inference
            img_np = image_tensor.cpu().numpy()
            
            # Create input dict based on whether reference is provided
            inputs = {"input": img_np}
            if ref_tensor is not None:
                inputs["reference"] = ref_tensor.cpu().numpy()
            
            # Run assessment model
            outputs = self._run_onnx_inference(self.assessment_model, img_np)
            
            if outputs is None or len(outputs) < 2:
                print("Assessment model returned invalid output")
                return 0.5, 0  # Default to mid-quality and color correction
            
            # Extract quality score and degradation type
            quality_score = outputs[0][0][0]  # Assuming first output is quality
            degradation_logits = outputs[1][0]  # Assuming second output is degradation
            degradation_type = np.argmax(degradation_logits)
            
            return quality_score, degradation_type
        except Exception as e:
            print(f"Assessment error: {e}")
            return 0.5, 0  # Default values on error
    
    def enhance_image(self, image_tensor, degradation_type):
        """Apply appropriate enhancement based on degradation type"""
        try:
            # Convert to numpy for ONNX inference
            img_np = image_tensor.cpu().numpy()
            
            # Select appropriate model based on degradation type
            if degradation_type == 0:  # Color cast
                model = self.color_model
                model_name = "color correction"
            elif degradation_type == 1:  # Blur
                model = self.deblur_model
                model_name = "deblur"
            elif degradation_type == 2:  # Low light
                model = self.lowlight_model
                model_name = "low light"
            else:  # Noise or default
                model = self.denoise_model
                model_name = "denoising"
            
            print(f"Applying {model_name} enhancement...")
            
            # Run enhancement
            outputs = self._run_onnx_inference(model, img_np)
            if outputs is None or len(outputs) == 0:
                print(f"Enhancement model ({model_name}) failed")
                return image_tensor  # Return original on failure
            
            # Convert output back to tensor
            enhanced = torch.from_numpy(outputs[0])
            
            return enhanced
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image_tensor  # Return original on error
    
    def postprocess_image(self, enhanced_tensor, original_size):
        """Convert tensor to displayable image and resize to original dimensions"""
        try:
            # Remove batch dimension and rearrange to HWC
            enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Scale to 0-255 range
            enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
            
            # Resize to original dimensions
            if original_size:
                enhanced_np = cv2.resize(enhanced_np, (original_size[0], original_size[1]))
            
            return enhanced_np
        except Exception as e:
            print(f"Postprocessing error: {e}")
            return None
    
    def process_image(self, input_path, output_path=None, ref_path=None):
        """Process a single image through the complete pipeline"""
        start_time = time.time()
        
        print(f"Processing image: {input_path}")
        
        # Preprocess input image
        image_tensor, original_size = self.preprocess_image(input_path)
        if image_tensor is None:
            return None
        
        # Preprocess reference image if provided
        ref_tensor = None
        if ref_path:
            ref_tensor, _ = self.preprocess_image(ref_path)
        
        # Assess image quality and degradation type
        quality, degradation_type = self.assess_image(image_tensor, ref_tensor)
        degradation_names = ["color cast", "blur", "low light", "noise"]
        print(f"Assessment: Quality score: {quality:.2f}, Degradation: {degradation_names[degradation_type]}")
        
        # Enhance image
        enhanced_tensor = self.enhance_image(image_tensor, degradation_type)
        
        # Postprocess
        enhanced_image = self.postprocess_image(enhanced_tensor, original_size)
        
        # Save or return
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
            print(f"Enhanced image saved to {output_path}")
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return enhanced_image
    
    def process_directory(self, input_dir, output_dir=None):
        """Process all images in a directory"""
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
                
                # Process each image
                self.process_image(input_path, output_path)
                count += 1
        
        print(f"Processed {count} images")

def main():
    parser = argparse.ArgumentParser(description="Underwater Image Enhancement for Raspberry Pi")
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output", type=str, help="Output path or directory (optional)")
    parser.add_argument("--reference", type=str, help="Reference image path (optional)")
    parser.add_argument("--models", type=str, default="./models", help="Directory containing ONNX models")
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = UnderwaterImageEnhancer(model_dir=args.models)
    
    # Process input (file or directory)
    if os.path.isdir(args.input):
        enhancer.process_directory(args.input, args.output)
    else:
        enhancer.process_image(args.input, args.output, args.reference)

if __name__ == "__main__":
    main()

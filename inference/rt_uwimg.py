import os
import time
import cv2
import psutil
import csv
from datetime import datetime
from picamera2 import Picamera2
from underwater_image_enhancer import UnderwaterImageEnhancer

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_cpu_temp():
    """Return CPU temperature in Celsius."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_milli = int(f.read())
        return temp_milli / 1000.0
    except:
        return None

def get_power_usage():
    """Stub for power usage; integrate with real HAT sensor if available."""
    # Return None or 0 if unavailable, or code for INA219, etc. if you have.
    return None

def capture_and_enhance(
    model_dir="./models",
    input_dir="input_images",
    output_dir="output_images",
    csv_path="results.csv",
    frame_width=640,
    frame_height=480,
    display=True
):
    # Ensure required folders exist
    ensure_dir(input_dir)
    ensure_dir(output_dir)

    enhancer = UnderwaterImageEnhancer(model_dir=model_dir)

    # Prepare CSV log
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame",
        "Input Filename",
        "Input Quality Score",
        "Degradation Type",
        "Enhanced Filename",
        "Enhancement Applied",
        "Processing Time (s)",
        "Enhanced Quality Score",
        "CPU (%)",
        "RAM (%)",
        "CPU Temp (C)",
        "Power Usage (W)"
    ])

    # Setup the Pi camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (frame_width, frame_height)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow camera to warm up
    print("\n--- Underwater Image Capture & Enhancement ---")
    print("Press Enter to capture/enhance. Type 'exit' to quit.")

    frame_count = 0
    try:
        while True:
            user_input = input("Press Enter to capture (or type 'exit' to quit): ").strip().lower()
            if user_input == "exit":
                break

            # Resource stats before capture
            cpu_before = psutil.cpu_percent(interval=None)
            ram_before = psutil.virtual_memory().percent
            temp_before = get_cpu_temp()
            power_before = get_power_usage()

            # Time the processing
            start_time = time.time()

            # Capture frame from Pi camera
            frame = picam2.capture_array()  # (H, W, 3), RGB format
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            raw_fname = f"input_{timestamp}_{frame_count:04d}.jpg"
            raw_fpath = os.path.join(input_dir, raw_fname)
            cv2.imwrite(raw_fpath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Saved input image: {raw_fpath}")

            # Enhance the image and save result
            degradation_names = ["color cast", "blur", "low light", "noise"]
            quality, degradation_type = 0, 0  # defaults
            enhancement_method = ""
            enhanced_quality = ""
            try:
                # Preprocess
                image_tensor, original_size = enhancer.preprocess_image(raw_fpath)
                if image_tensor is not None:
                    # Assess
                    quality, degradation_type = enhancer.assess_image(image_tensor)
                    enhancement_method = degradation_names[degradation_type]
                    print(f"Assessment: Quality score: {quality:.2f}, Degradation: {enhancement_method}")

                    # Enhance
                    enhanced_tensor = enhancer.enhance_image(image_tensor, degradation_type)
                    enhanced_img = enhancer.postprocess_image(enhanced_tensor, original_size)

                    enh_fname = f"enhanced_{timestamp}_{frame_count:04d}.jpg"
                    enh_fpath = os.path.join(output_dir, enh_fname)
                    if enhanced_img is not None:
                        cv2.imwrite(enh_fpath, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
                        print(f"Saved enhanced image: {enh_fpath}")

                        # Assess enhanced image quality
                        enhanced_tensor2, _ = enhancer.preprocess_image(enh_fpath)
                        if enhanced_tensor2 is not None:
                            enhanced_quality, _ = enhancer.assess_image(enhanced_tensor2)
                            print(f"Enhanced image quality score: {enhanced_quality:.2f}")

                        if display:
                            cv2.imshow("Enhanced", enhanced_img)
                            print("Press any key in the image window to close.")
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                    else:
                        print("Enhancement failed for this frame.")
                        enh_fname, enh_fpath = "", ""
                else:
                    print("Could not process input image (preprocessing failed).")
            except Exception as e:
                print(f"Error during processing: {e}")
                enh_fname, enh_fpath = "", ""

            processing_time = time.time() - start_time

            # Resource stats after
            cpu_after = psutil.cpu_percent(interval=None)
            ram_after = psutil.virtual_memory().percent
            temp_after = get_cpu_temp()
            power_after = get_power_usage()

            # Average CPU and RAM over operation
            cpu_avg = (cpu_before + cpu_after) / 2.0
            ram_avg = (ram_before + ram_after) / 2.0

            # Write to CSV
            csv_writer.writerow([
                frame_count,
                raw_fname,
                f"{quality:.2f}",
                enhancement_method,
                enh_fname,
                enhancement_method,
                f"{processing_time:.2f}",
                f"{enhanced_quality:.2f}" if enhanced_quality != "" else "",
                f"{cpu_avg:.1f}",
                f"{ram_avg:.1f}",
                f"{temp_after:.1f}" if temp_after is not None else "",
                f"{power_after:.2f}" if power_after is not None else "",
            ])
            csv_file.flush()

            frame_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        try:
            picam2.stop()
            print("Camera stopped. Exiting.")
        except Exception as e:
            print(f"Camera stop error (probably already stopped): {e}")
        csv_file.close()
        print(f"Results saved in {csv_file.name}")

if __name__ == "__main__":
    capture_and_enhance(
        model_dir="./models",
        input_dir="input_images",
        output_dir="output_images",
        csv_path="results.csv",
        frame_width=640,
        frame_height=480,
        display=True
    )

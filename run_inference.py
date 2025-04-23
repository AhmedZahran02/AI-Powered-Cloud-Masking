import argparse
from pathlib import Path
import torch
import torch.profiler
from torchvision import transforms # type: ignore
import numpy as np
import pandas as pd
from src.models.unet import UNet
from src.models.slim_unet import SlimUNet
from src.rle_encoder_decoder import rle_encode
from src.visualization import plot_image_and_mask
from src.utils import load_and_normalize_tiff
from skimage.transform import resize
from tqdm import tqdm  # Import tqdm
import logging
from torchinfo import summary # type: ignore
from ptflops import get_model_complexity_info # type: ignore
from datetime import datetime
import os

logging.basicConfig(
    filename="outputs\\logs\\model_logs.txt",
    filemode="w",  # Overwrite each run
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def log_model_stats(model, input_size=(4, IMG_SIZE, IMG_SIZE)):
    logging.info("Model Summary:")
    model_stats = summary(model, input_size=(1, *input_size), verbose=0)
    logging.info(str(model_stats))

    # Log FLOPs and params using ptflops
    macs, params = get_model_complexity_info(model, input_res=input_size, as_strings=True, print_per_layer_stat=False)
    logging.info(f"Total Parameters: {params}")
    logging.info(f"Multiply-Accumulate Operations (MACs): {macs}")

def load_model(model_path):
    model = SlimUNet(n_channels=4, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def predict(model, image_tensor, threshold):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        output = model(image_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (prob > threshold).astype(np.uint8)
        
    return mask

def main(test_folder, output_csv, model_path, threshold=0.5, visualize=False):
    test_folder = Path(test_folder)
    prediction_dir = Path("outputs/predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    log_model_stats(model)
    submissions = []

    # Configure the profiler with more comprehensive settings
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,  # Skip the first step (warm-up)
            warmup=1,  # Number of warm-up steps
            active=3,  # Number of active steps to record
            repeat=1  # Number of profiling cycles
        ),
        profile_memory=True,
        record_shapes=True,
        with_stack=True,  # Enable stack tracing
        with_flops=True,  # Estimate FLOPs
        with_modules=True  # Record model hierarchy
    ) as profiler:
        for i, image_file in enumerate(tqdm(sorted(test_folder.glob("*.tif")), desc="Processing Images", unit="image")):
            image = load_and_normalize_tiff(image_file)

            if image.shape[0] != 4:
                raise ValueError(f"Image must have 4 channels (RGB + IR), found {image.shape[0]} channels.")

            if image.shape[1] != IMG_SIZE or image.shape[2] != IMG_SIZE:
                image = resize(image.transpose(1, 2, 0), (IMG_SIZE, IMG_SIZE), order=1, mode='reflect', preserve_range=True)
                image = image.transpose(2, 0, 1)

            # Normalize each band
            for i in range(image.shape[0]):
                band = image[i]
                if band.max() > band.min():
                    image[i] = (band - band.min()) / (band.max() - band.min())
                else:
                    image[i] = 0

            image_tensor = torch.tensor(image).float()  # (C, H, W)

            mask = predict(model, image_tensor, threshold)        
            mask_resized = resize(mask, (512, 512), order=0, mode='reflect', preserve_range=True)
            mask_resized = (mask_resized > 0.5).astype(np.uint8) 

            # RLE Encoding
            rle = rle_encode(mask_resized)
            submissions.append({
                "id": image_file.stem,
                "segmentation": rle
            })

            if visualize:
                vis_path = prediction_dir / f"{image_file.stem}_prediction.png"
                plot_image_and_mask(image, mask_resized, title="Prediction Visualization", save_path=vis_path)

            # Step the profiler
            profiler.step()

        df = pd.DataFrame(submissions)
        df.to_csv(output_csv, index=False)
        print(f"Saved submission to {output_csv}")

        # Generate comprehensive profiling reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("outputs", "logs", f"pytorch_report_{timestamp}.txt")

        with open(report_path, "w") as report_file:
            # Generate comprehensive profiling reports
            report_file.write("\n=== PyTorch Profiler Report ===\n")
            report_file.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_file.write(f"Processed {len(submissions)} images\n\n")

            # Get key averages and total stats
            key_averages = profiler.key_averages(group_by_input_shape=True)
            total_stats = key_averages.total_average()

            # 1. Write key averages table
            report_file.write("\nKey Averages Table:\n")
            report_file.write(key_averages.table(
                sort_by="self_cuda_time_total",
                row_limit=20,
                header="CUDA Time (ms)",
                top_level_events_only=True
            ) + "\n")

            # 2. Write memory statistics
            report_file.write("\nMemory Statistics:\n")
            report_file.write(key_averages.table(
                sort_by="self_cuda_memory_usage",
                row_limit=20,
                header="CUDA Memory Usage (MB)"
            ) + "\n")

            # 3. Write aggregated statistics
            report_file.write("\nAggregate Statistics:\n")
            if hasattr(total_stats, 'cpu_time_total'):
                report_file.write(f"Total CPU time: {total_stats.cpu_time_total / 1000:.2f} ms\n")

            if hasattr(total_stats, 'self_cuda_time_total'):
                report_file.write(f"Total CUDA time: {total_stats.self_cuda_time_total / 1000:.2f} ms\n")
            elif hasattr(total_stats, 'cuda_time_total'):
                report_file.write(f"Total CUDA time: {total_stats.cuda_time_total / 1000:.2f} ms\n")

            if hasattr(total_stats, 'cuda_memory_usage'):
                report_file.write(f"Peak CUDA memory usage: {total_stats.cuda_memory_usage / (1024 * 1024):.2f} MB\n")

            # 4. Write event statistics
            report_file.write("\nDetailed Event Statistics:\n")
            for event in key_averages:
                report_file.write(f"\n{event.key}:\n")
                report_file.write(f"  Calls: {event.count}\n")
                if hasattr(event, 'cpu_time_total'):
                    report_file.write(f"  CPU time: {event.cpu_time_total / 1000:.3f}ms\n")
                if hasattr(event, 'self_cuda_time_total'):
                    report_file.write(f"  CUDA time: {event.self_cuda_time_total / 1000:.3f}ms\n")
                if hasattr(event, 'cuda_memory_usage'):
                    report_file.write(f"  Memory: {event.cuda_memory_usage / (1024 * 1024):.2f}MB\n")

            # 5. Note about chrome trace
            trace_file = os.path.join("outputs", "logs", f"pytorch_trace_{timestamp}.json")
            profiler.export_chrome_trace(trace_file)
            report_file.write(f"\nChrome trace saved to: {trace_file}\n")
            report_file.write("Open in chrome://tracing/ for visualization\n")

        print(f"\nProfiler report saved to: {report_path}")
        print(f"Chrome trace saved to: {trace_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_folder", type=str, required=True, help="Path to folder with .tif images")
    parser.add_argument("--model_path", type=str, default="outputs\\models\\Unet_model.pth", help="Path to trained model")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Path to output CSV")
    parser.add_argument("--threshold", type=float, default=0.35, help="Threshold for binarizing mask")
    parser.add_argument("--visualize", action="store_true", help="Save prediction visualizations")

    args = parser.parse_args()
    main(args.test_folder, args.output_csv, args.model_path, args.threshold, args.visualize)


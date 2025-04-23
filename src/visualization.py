import matplotlib.pyplot as plt
from pathlib import Path

def plot_image_and_mask(image, mask, title, save_path=None):
    image_rgb = image[:3].transpose(1, 2, 0)
    image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min() + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Mask")
    fig.suptitle(title)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def plot_image_and_mask_and_prediction(image, mask, prediction, title,visualize=False):
    
    image_rgb = image[:3].transpose(1, 2, 0)
    image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min() + 1e-6)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].axis('off')
    axes[0].set_title(f"Image")

    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')
    axes[1].set_title(f"Ground Truth Mask")

    axes[2].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    axes[2].axis('off')
    axes[2].set_title(f"Predicted Mask")
    
    fig.suptitle(title)
    
    save_path = Path("output") / f"{title}_prediction.png"
    if visualize == True:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()   
    
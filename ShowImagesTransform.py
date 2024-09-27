from torchvision.transforms import v2 as transformsV2
import torch
from PIL import Image
import matplotlib.pyplot as plt


def rescale_0_1(image):
    """Rescale pixel values to range [0, 1] for visualization purposes only."""
    min_val = image.min()
    max_val = image.max()
    rescaled_image = (image-min_val)/abs(max_val-min_val)
    return rescaled_image

def plot_transformation_for_image_in_batch(image_path):
    """
    Plots all the transformation that have been applied to batch on a sample image.

    Args:
    - image path (str): Path to the image file.
    """

    transforms = [
        transformsV2.RandomHorizontalFlip(p=1),
        transformsV2.RandomVerticalFlip(p=1),
        transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=1),
        transformsV2.RandomAutocontrast(p=1),  
        transformsV2.ColorJitter(brightness=0.25, contrast=0.20, saturation=0.20, hue=0.1)]
    
    transform_names = [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomAdjustSharpness",
        "RandomAutocontrast",
        "ColorJitter",
    ]   

    toTensor = transformsV2.Compose([
        transformsV2.Resize((224, 224)),
        transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
        transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    ])

    image = Image.open(image_path)
    image = toTensor(image)
    image_ready = image.permute(1, 2, 0)    
    _, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    axes[0].imshow(image_ready)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    for i, transform in enumerate(transforms):
        ax = axes[i+1]
        transformed_image = transform(image_ready.permute(2, 0, 1)).permute(1, 2, 0)
        if i == 6:
            transformed_image = rescale_0_1(transformed_image)
        
        ax.imshow(transformed_image)
        ax.set_title(f"{transform_names[i]}", fontweight="bold")
        ax.axis("off")
    
    plt.suptitle("Possible transformations on a sample image", fontsize=20, fontweight="bold", color="red")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    plt.tight_layout()
    plt.show()



def plot_misclassified_images(image_paths, true_labels, predicted_labels):
    """
    Plots misclassified images with their true and predicted labels.

    Args:
    - image_paths (list of str): List of file paths to the misclassified images.
    - true_labels (list of int): List of true labels for the images.
    - predicted_labels (list of int): List of predicted labels for the images.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Misclassified Examples", fontsize=20, fontweight="bold", color="red")
    for i, ax in enumerate(axes):
        img = Image.open(image_paths[i])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"True: {true_labels[i]} (hotdog), Predicted: {predicted_labels[i]} (not hotdog)",  fontweight="bold")

    plt.tight_layout()
    plt.show()

image_paths = [
    "Exercises/Project_1/run_FirstRun_Adam_Scheduler_Yes/Results/misclassified_52_true_lable_0_predicted_label_1.png",
    "Exercises/Project_1/run_FirstRun_Adam_Scheduler_Yes/Results/misclassified_62_true_lable_0_predicted_label_1.png",
    "Exercises/Project_1/run_FirstRun_Adam_Scheduler_Yes/Results/misclassified_103_true_lable_0_predicted_label_1.png"
]

true_labels = [0, 0, 0]
predicted_labels = [1, 1, 1]
plot_misclassified_images(image_paths, true_labels, predicted_labels)


image_path = "Exercises\\data\\hotdog_nothotdog\\train\\hotdog\\hotdog (310).jpg"
plot_transformation_for_image_in_batch(image_path)

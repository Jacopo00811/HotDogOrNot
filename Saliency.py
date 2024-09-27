import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as transformsV2
from NetworkFineTuning import *

test_transform = transformsV2.Compose([
    transformsV2.Resize((224, 224)),
    transformsV2.ToImage(),
    transformsV2.ToDtype(torch.float32, scale=True)
])

hyperparameters = { 
    "backbone": "mobilenet_v3_large",
    "number of classes": 2,
}

def add_noise(image, noise_level=0.1):
    noise = torch.randn_like(image) * noise_level
    return image + noise

def saliency(model, image):
    image.requires_grad_()

    output = model(image.unsqueeze(0))
    output_idx = output.argmax()
    output_max = output[0, output_idx]

    output_max.backward()
    
    saliency_map, _ = torch.max(image.grad.data.abs(), dim=0)
    
    return saliency_map

def plot_vanilla_saliency_maps(model, images):
    fig, ax = plt.subplots(2, len(images), figsize=(10, 8))

    for i, image in enumerate(images):
        ax[0, i].imshow(image.permute(1, 2, 0).detach().numpy())
        ax[0, i].axis('off')
        ax[0, i].set_title(f'Original Image {i+1}', fontweight="bold")
        
        saliency_map = saliency(model, image)
        ax[1, i].imshow(saliency_map.detach().numpy(), cmap=plt.cm.hot)
        ax[1, i].axis('off')
        ax[1, i].set_title(f'Saliency Map Image {i+1}', fontweight="bold")
    plt.suptitle("Vanilla saliency maps", fontsize=20, fontweight="bold", color="red")
    plt.tight_layout()
    plt.show()

def smoothgrad_saliency(model, image, target_class, n_samples=50, noise_level=0.1):
    image.requires_grad_()
    smooth_grad = torch.zeros_like(image)

    for _ in range(n_samples):
        noisy_image = add_noise(image, noise_level)
        noisy_image.requires_grad_()
        noisy_image.retain_grad()  # Retain gradients for non-leaf tensor

        output = model(noisy_image.unsqueeze(0))
        output[0, target_class].backward()

        smooth_grad += noisy_image.grad.abs()

    smooth_grad /= n_samples
    smooth_grad = smooth_grad.mean(dim=0)

    return smooth_grad

def plot_smoothgrad_saliency_maps(model, images, noise_levels, n_samples=50):
    fig, ax = plt.subplots(len(images), len(noise_levels) + 1, figsize=(15, 8))

    for i, image in enumerate(images):
        output = model(image.unsqueeze(0))
        target_class = output.argmax(dim=1).item()

        ax[i, 0].imshow(image.permute(1, 2, 0).detach().numpy())
        ax[i, 0].axis('off')
        ax[i, 0].set_title(f'Original Image {i+1}', fontweight="bold")

        for j, noise_level in enumerate(noise_levels):
            smooth_grad = smoothgrad_saliency(model, image, target_class, n_samples=n_samples, noise_level=noise_level)
            ax[i, j+1].imshow(smooth_grad.detach().numpy(), cmap=plt.cm.hot, aspect='auto')
            ax[i, j+1].axis('off')
            ax[i, j+1].set_title(f'SmoothGrad (noise {noise_level*100}%) Image {i+1}', fontweight="bold")
    plt.suptitle("SmoothGrad saliency maps for different noise levels", fontsize=20, fontweight="bold", color="red")
    plt.tight_layout()
    plt.show()



model = MultiModel("mobilenet_v3_large", hyperparameters=hyperparameters, load_pretrained=False)
model_weights_path = "Exercises\\Project_1\\run_FirstRun_Adam_Scheduler_Yes\\FirstRun_accuracy_0.939.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=True), strict=True)
model.eval()

image1 = Image.open("Exercises\\data\\hotdog_nothotdog\\train\\hotdog\\hotdog (89).jpg")
image1 = test_transform(image1)

image2 = Image.open("Exercises\\data\\hotdog_nothotdog\\train\\hotdog\\hotdog (310).jpg")
image2 = test_transform(image2)

images = [image1, image2]

noise_levels = [0.2, 0.1]


plot_vanilla_saliency_maps(model, images)
plot_smoothgrad_saliency_maps(model, images, noise_levels)

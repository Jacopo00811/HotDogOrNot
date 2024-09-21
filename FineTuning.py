from NetworkFineTuning import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")
print("Current working directory:", os.getcwd())

hyperparameters = {
    "number of classes": 2,
    "device": device,
    "epochs": [20, 20, 15],
    "batch size": 32,
    "learning rate": [0.001, 0.0001, 1e-05],
    "image size": (224, 224),
    "backbone": "mobilenet_v3_large",
    "torch home": "TorchvisionModels",
    "network name": "FirstRun",
    "momentum": 0.9,
    "weight decay": 0.0005,
    "optimizer": "Adam",
    "scheduler": "Yes",
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08,
    "step size": [5, 5, 3],
    "gamma": [0.8, 0.9, 0.7],
}

train_transform = transformsV2.Compose([transformsV2.Resize(hyperparameters["image size"]),
                                        transformsV2.RandomVerticalFlip(p=0.5),
                                        transformsV2.RandomHorizontalFlip(
                                            p=0.5),
                                        transformsV2.RandomAdjustSharpness(
                                            sharpness_factor=2, p=0.5),
                                        transformsV2.RandomAutocontrast(p=0.5),
                                        transformsV2.ColorJitter(
                                            brightness=0.25, saturation=0.20),
                                        # Replace deprecated ToTensor()
                                        transformsV2.ToImage(),
                                        transformsV2.ToDtype(torch.float32, scale=True)])
test_transform = transformsV2.Compose([transformsV2.Resize(hyperparameters["image size"]),
                                       # Replace deprecated ToTensor()
                                       transformsV2.ToImage(),
                                       transformsV2.ToDtype(torch.float32, scale=True)])


trainset = Hotdog_NotHotdog(train=True, transforms=train_transform)

# Define the sizes for train and validation splits
train_size = int(0.8 * len(trainset))  # 80% for training
val_size = len(trainset) - train_size   # 20% for validation

# Split the train dataset into train and val
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
# TODO: Check if this is correct (I think it applies the transofrmations before splitting)
val_dataset.dataset.set_transforms(test_transform)
testset = Hotdog_NotHotdog(train=False, transforms=test_transform)

train_loader = DataLoader(
    train_dataset, batch_size=hyperparameters["batch size"], shuffle=True)
val_loader = DataLoader(
    val_dataset, batch_size=hyperparameters["batch size"], shuffle=False)
test_loader = DataLoader(testset, batch_size=hyperparameters["batch size"],
                         shuffle=False)


os.environ['TORCH_HOME'] = hyperparameters["torch home"]
os.makedirs(hyperparameters["torch home"], exist_ok=True)

# Define the loss function
loss_function = nn.CrossEntropyLoss()

# Empty memory before start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

automatic_fine_tune(hyperparameters, hyperparameters["backbone"], device,
                    loss_function, train_loader, val_loader, test_loader, hyperparameters["optimizer"], hyperparameters["scheduler"])

import torch.optim as optim
from torchvision import transforms
from smp_training_functions import *

# Define transformations

# # transforms for the arsenic dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.534], std=[0.131]),
# ])

# transforms for the initial taxol dataset tiled->6
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.439], std=[0.061]),
])


def train_model(args):
    config = create_config(args)  # Create experiment-specific directories for saving logs and models
    setup_logging(args, config["log_file"])

    save_hyperparameters(args, config)

    ## if training with k-fold cross validation##
    if args.use_cross_validation:
        args.config = config

        # pull the entire training dataset for creating the k-folds
        dataset = CustomImageDataset(
            data_dir=args.train_dir,
            no_of_classes=args.num_classes,
            transform=transform,
        )

        # the k in this function is for k-fold cross validation, this is NOT the k parameter in args.
        # The k parameter in args is for the k-NearestNeightbors evaluation.
        # do not use args.k in this function
        train_with_cross_validation(dataset=dataset, model_fn=lambda: get_model(args), args=args, k=5)
        return  # skip the rest of this script if training with k-fold cross validation

    ## if NOT training with k-fold cross validation, then ##

    # Load datasets & initialize dataloaders
    train_loader_singles, validation_loader_singles, train_loader_pairs, validation_loader_pairs = get_dataloaders(args, transform=transform)

    # load model based on args input
    model = get_model(args)

    if args.weights_path:
        model = load_pretrained_weights(model, weight_path=args.weights_path)
    else:
        model = load_pretrained_weights(model, use_torchvision=True)  # ImageNet weights

    model.to(args.device)

    # set optimizer function
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) if args.use_scheduler else None

    # Pack training parameters
    train_params = {
        "model": model,
        "train_loader_singles": train_loader_singles,
        "validation_loader_singles": validation_loader_singles,
        "train_loader_pairs": train_loader_pairs,
        "validation_loader_pairs": validation_loader_pairs,
        "device": args.device,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "config": config,
        "args": args
    }

    if args.loss_function == "contrastive":
        train_contrastive_model(train_params=train_params)

    else:  # args.loss_function == "cross_entropy" or "earth_movers"
        train_classification_model(train_params=train_params)


if __name__ == '__main__':
    the_args = parse_args()
    train_model(the_args)

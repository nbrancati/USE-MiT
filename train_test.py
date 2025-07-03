import os, cv2, argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as album
from albumentations.pytorch import ToTensorV2
from metric import iou, pix_acc
from augmentation import *

img_size = 256

def get_training_augmentation_tr():
    """Training augmentations with flips and rotation."""
    train_transform = [
        album.Resize(img_size, img_size),
        album.OneOf([
            album.HorizontalFlip(p=1),
            album.VerticalFlip(p=1),
            album.RandomRotate90(p=1),
        ], p=0.75),
        ToTensorV2(),
    ]
    return album.Compose(train_transform)

def get_training_augmentation_ori():
    """Basic preprocessing: resize and convert to tensor."""
    return album.Compose([
        album.Resize(img_size, img_size),
        ToTensorV2(),
    ])

class CustomDataset(Dataset):
    """Custom dataset for loading image-mask pairs."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        file_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, file_name)
        mask_path = os.path.join(self.masks_dir, file_name)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY) / 255

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'].float(), sample['mask']

        return image, mask, file_name



def train_model(criterion, dataset, dataset_tr, dataset_test, kf, k_folds, device, num_epochs, lr, bs):
    """Main training loop with validation and logging."""
    model = smp.USE_MiT(
        encoder_name="mit_b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    )
    model.cuda(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    num_sample = len(dataset)
    train_indices = list(range(num_sample))
    train_sampler = SubsetRandomSampler(train_indices)

    combined_dataset = torch.utils.data.ConcatDataset([dataset, dataset_tr])
    train_sampler_tr = [i + len(dataset) for i in train_sampler.indices]
    combined_indices = np.append(train_sampler.indices, train_sampler_tr)
    train_sampler.indices = combined_indices

    train_loader = DataLoader(combined_dataset, batch_size=bs, sampler=train_sampler)
    test_loader = DataLoader(dataset_test)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, _ in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).long().squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            output_binary = outputs[:, 1, :, :]
            loss = criterion(output_binary, targets.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.sampler)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_loss_total = 0.0
        n_classes = 2
        n = len(test_loader.sampler)
        pixel_acc, iou_smp, prec_smp, rec_smp, spec_smp, f1_smp = [0.0] * 6
        class_iou = [0.0] * n_classes
        all_preds, all_targets, all_file_names = [], [], []

        with torch.no_grad():
            for val_inputs, val_targets, file_names in tqdm(test_loader):
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device).long().squeeze()
                batch_size = val_inputs.size(0)
                val_outputs = model(val_inputs)
                output = torch.argmax(val_outputs, dim=1).squeeze()

                if batch_size == 1:
                    output = output.unsqueeze(0)
                    val_targets = val_targets.unsqueeze(0)

                class_iou += iou(output, val_targets, batch_size, n_classes) * (batch_size / n)
                pixel_acc += pix_acc(output, val_targets, batch_size) * (batch_size / n)

                output_binary = val_outputs[:, 1, :, :]
                val_loss = criterion(output_binary, val_targets.float())
                val_loss_total += val_loss.item() * val_inputs.size(0)

                all_preds.append(output.cpu())
                all_targets.append(val_targets.cpu())
                all_file_names.extend(file_names)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        tp, fp, fn, tn = smp.metrics.get_stats(all_preds, all_targets, mode='binary', threshold=0.5)
        iou_smp += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        prec_smp = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        rec_smp = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        f1_smp = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        val_epoch_loss = val_loss_total / n
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {val_epoch_loss:.4f}')
        print(f"Accuracy: {pixel_acc:.3f}, Precision: {prec_smp:.3f}, Recall: {rec_smp:.3f}, Dice: {f1_smp:.3f}, IOU: {iou_smp:.3f}")


    torch.save(model.state_dict(), 'model.pth')

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = CustomDataset(root_dir=args.data_dir, transform=get_training_augmentation_ori())
    dataset_tr = CustomDataset(root_dir=args.data_dir, transform=get_training_augmentation_tr())
    dataset_test = CustomDataset(root_dir=args.test_dir, transform=get_training_augmentation_ori())

    criterion = nn.BCEWithLogitsLoss()

    kf = KFold(n_splits=args.folds, shuffle=True)
    train_model(criterion, dataset, dataset_tr, dataset_test, kf, args.folds, device, args.num_epoch, args.learning_rate, args.batch)

if __name__ == '__main__':
    DATA_DIR = "train"
    TEST_DIR = "test"

    parser = argparse.ArgumentParser(description='Train U-Net model for segmentation')
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--test_dir', default=TEST_DIR)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--gpu_list', default="0")
    args = parser.parse_args()
    main(args)

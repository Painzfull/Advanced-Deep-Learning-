import torch 
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time 
import torch.utils.data
import cv2
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import directed_hausdorff


#%%
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, photo_paths, mask_paths, transform=None, device=torch.device('cuda')):
        self.photo_paths = photo_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.device = device

    def __getitem__(self, idx):
        photo = Image.open(self.photo_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            photo = self.transform(photo).to(self.device)
            mask = self.transform(mask).to(self.device)

        return photo, mask

    def __len__(self):
        return len(self.photo_paths)

def load_data(photo_dir, mask_dir, test_size=0.3):
    photo_files = [os.path.join(photo_dir, img) for img in os.listdir(photo_dir)]
    mask_files = [os.path.join(mask_dir, img) for img in os.listdir(mask_dir)]

    photo_files.sort()
    mask_files.sort()

    # Create train and test datasets while preserving pairs
    train_photo_paths, test_photo_paths, train_mask_paths, test_mask_paths = train_test_split(
        photo_files, mask_files, test_size=test_size, random_state=42
    )

    # Create datasets
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CustomDataset(train_photo_paths, train_mask_paths, transform=transform, device=device)
    test_dataset = CustomDataset(test_photo_paths, test_mask_paths, transform=transform, device=device)

    print("Training Dataset Sample Count:", len(train_dataset))
    print("Test Dataset Sample Count:", len(test_dataset))

    return train_dataset, test_dataset

photo_dir = "C:/Users/Froggremann/Desktop/PROJECTS/Çalışılmış_Verisetleri/Boyun_karşı_veriseti/Fotoğraflar"
mask_dir = "C:/Users/Froggremann/Desktop/PROJECTS/Çalışılmış_Verisetleri/Boyun_karşı_veriseti/Maskeler"

train_dataset, test_dataset = load_data(photo_dir, mask_dir, test_size=0.3)

batch_size = 1
learning_rate = 0.0001
num_epochs = 100
num_classes = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for i, (photo, mask) in enumerate(train_loader):
    print(f"Photo Tensor Shape: {photo.shape}")
    print(f"Mask Tensor Shape: {mask.shape}")

    if i == 0:
        example_photo_pixels = photo[0].squeeze().cpu().numpy()
        example_mask_pixels = mask[0].squeeze().cpu().numpy()
        
        print("Example Photo pixel values:\n", example_photo_pixels)
        print("Example Mask pixel values:\n", example_mask_pixels)
        break



#%%
import matplotlib.pyplot as plt

def plot_image_mask_pair(photo, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Photo
    axes[0].imshow(photo.squeeze(), cmap='gray')  # Assuming photo is grayscale
    axes[0].set_title('Photo')
    axes[0].axis('off')

    # Mask
    axes[1].imshow(mask.squeeze(), cmap='gray')  # Assuming mask is grayscale
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Örnek olarak ilk fotoğraf ve maskeyi çizdirelim
example_photo, example_mask = train_dataset[100]  # Örnek olarak train setinden ilk öğeyi alıyoruz
plot_image_mask_pair(example_photo.cpu().numpy(), example_mask.cpu().numpy())



#%%
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        
        up6 = self.up6(conv5)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(conv8)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(merge9)
        
        out = self.out(conv9)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Cihaz seçimi (GPU öncelikli, CPU kullanımı)

model = UNet(in_channels=1, out_channels=num_classes)  # Model oluşturma (num_classes'e uygun olarak)

model = model.to(device)  # Modeli seçilen cihaza gönderme

# Eğitim veya tahminler için modeli kullanmaya başlayabilirsiniz


# Örnek giriş verisi
sample_input = torch.randn(1, 1, 512, 512).to(device)

# Modelin ilerletilmesi
model_output = model(sample_input)

print(model_output.shape)  # Model çıkış şeklini kontrol etmek için




#%%
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)

#%%

smooth = 100

def dice_coef(y_true, y_pred):
    y_truef = torch.flatten(y_true)
    y_predf = torch.flatten(y_pred)
    intersection = torch.sum(y_truef * y_predf)
    return ((2 * intersection + smooth) / (torch.sum(y_truef) + torch.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    sum_ = torch.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    return -iou(y_true, y_pred)


import numpy as np
from scipy.spatial.distance import cdist
import torch

def hausdorff_distance(true_mask, predicted_mask, pixel_scale=0.5):
    """
    Gerçek maske sınırından tahmin edilen maske sınırına olan Hausdorff uzaklığını hesaplar.

    Parametreler:
        true_mask: Gerçek maske (PyTorch tensor).
        predicted_mask: Tahmin edilen maske (PyTorch tensor).
        pixel_scale: 1 pikselin gerçek dünya değerini temsil eder.

    Dönüş Değeri:
        Max ve min mesafe farkları (mm cinsinden).
    """
    def find_boundary(mask):
        return np.column_stack(np.where(mask))

    # PyTorch tensorlerini NumPy array'lerine dönüştürün
    true_mask_np = true_mask.cpu().numpy().astype(bool)
    predicted_mask_np = predicted_mask.cpu().numpy().astype(bool)

    # Maskelerin sınırlarını bulun
    true_boundary = find_boundary(true_mask_np)
    predicted_boundary = find_boundary(predicted_mask_np)

    # Mesafeleri hesaplayın
    distances_true_to_pred = cdist(true_boundary, predicted_boundary)
    distances_pred_to_true = cdist(predicted_boundary, true_boundary)

    # Max ve min mesafe farklarını hesaplayın
    if len(distances_true_to_pred) > 0 and len(distances_pred_to_true) > 0:
        max_distance_diff = max(np.max(np.min(distances_true_to_pred, axis=0)),
                                np.max(np.min(distances_pred_to_true, axis=0)))
        min_distance_diff = min(np.min(np.min(distances_true_to_pred, axis=0)),
                                np.min(np.min(distances_pred_to_true, axis=0)))
    else:
        max_distance_diff = 0.0
        min_distance_diff = 0.0

    # Max ve min mesafe farklarını mm cinsine çevirin
    max_distance_diff_mm = max_distance_diff * pixel_scale
    min_distance_diff_mm = min_distance_diff * pixel_scale

    return max_distance_diff_mm, min_distance_diff_mm

        
def visualize_best_worst_masks(loader, is_train):
    dice_scores = []

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        threshold = 0.5
        binary_preds = (torch.sigmoid(output) > threshold).float()

        dice = dice_coef(binary_preds, labels)
        hausdorff_dist = hausdorff_distance(binary_preds, labels)

        dice_scores.append((dice.item(), hausdorff_dist, images, labels, binary_preds))

    dice_scores.sort(key=lambda x: x[0], reverse= True)

    title = "Train Set" if is_train else "Test Set"

    all_scores = dice_scores[:]
    best_scores = all_scores[:5]
    worst_scores = all_scores[-5:]
    average_scores = all_scores[len(all_scores) // 2 - 2: len(all_scores) // 2 + 3]

    for scores, title_text in [(best_scores, "Top 5 Best"), (worst_scores, "Top 5 Worst "), (average_scores, "Average")]:
        for idx, (dice_score, hausdorff_dist, image, label, prediction) in enumerate(scores):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
            axes[0].set_title('Input Image')

            axes[1].imshow(label.squeeze().cpu().numpy(), cmap='PuBu')  # Label rengini değiştirelim
            axes[1].set_title('Ground Truth Mask')

            axes[2].imshow(prediction.squeeze().cpu().numpy(), cmap='seismic')  # Prediction rengini değiştirelim
            axes[2].set_title('Predicted Mask')

            mask_diff_1 = label.squeeze().cpu().numpy() - prediction.squeeze().cpu().numpy()
            mask_diff_2 = prediction.squeeze().cpu().numpy() - label.squeeze().cpu().numpy()
            mask_diff_3 = label.squeeze().cpu().numpy() - mask_diff_1
            mask_diff_4 = prediction.squeeze().cpu().numpy() - mask_diff_2

            axes[3].imshow(mask_diff_1, cmap='coolwarm', alpha=0.5)  # Mask difference rengini değiştirelim
            axes[3].imshow(mask_diff_2, cmap='coolwarm', alpha=0.5)
            axes[3].imshow(mask_diff_3, cmap='coolwarm', alpha=0.5)
            axes[3].imshow(mask_diff_4, cmap='coolwarm', alpha=0.5)
            axes[3].set_title('Mask Difference')

            max_dist = torch.max(torch.tensor(hausdorff_dist))
            min_dist = torch.min(torch.tensor(hausdorff_dist))
            axes[3].text(0.5, -0.15, f"Max Hausdorff Dist: {max_dist:.2f}", ha='center', transform=axes[3].transAxes)
            axes[3].text(0.5, -0.2, f"Min Hausdorff Dist: {min_dist:.2f}", ha='center', transform=axes[3].transAxes)

            plt.suptitle(f"{title} {title_text} Dice Scores: {dice_score}")

            plt.show()

            
import numpy as np

def calculate_mask_difference(mask1, mask2, pixel_scale=0.5):
    # İki maske arasındaki farkı hesapla
    mask_difference = np.abs(mask1.cpu().numpy() - mask2.cpu().numpy())

    # Farkın içindeki en küçük ve en büyük değerleri bul
    min_difference = np.min(mask_difference) * pixel_scale
    max_difference = np.max(mask_difference) * pixel_scale

    return min_difference, max_difference

def train(model, train_loader, test_loader, optimizer, num_epochs):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_dice = 0.0
        total_jaccard = 0.0
        total_hausdorff = []
        total_min_diff = []  # Min farkları sakla
        total_max_diff = []  # Max farkları sakla

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            threshold = 0.5
            binary_preds = (torch.sigmoid(output) > threshold).float()

            dice = dice_coef(binary_preds, labels)
            jaccard = iou(binary_preds, labels)
            hausdorff_dist, _ = hausdorff_distance(binary_preds, labels)  # Min ve Max farkları kullanmayacağız
            min_diff, max_diff = calculate_mask_difference(binary_preds, labels)  # Maskeler arasındaki farkı hesapla

            total_dice += dice.item() * images.size(0)
            total_jaccard += jaccard.item() * images.size(0)
            total_hausdorff.append(hausdorff_dist)
            total_min_diff.append(min_diff)  # Her bir min farkını listeye ekle
            total_max_diff.append(max_diff)  # Her bir max farkını listeye ekle

            pbar.set_postfix({'Loss': running_loss / (i + 1), 
                              'Dice': total_dice / ((i + 1) * images.size(0)), 
                              'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

        avg_hausdorff = torch.mean(torch.tensor(total_hausdorff))
        min_hausdorff = torch.min(torch.tensor(total_hausdorff))
        max_hausdorff = torch.max(torch.tensor(total_hausdorff))
        min_diff = min(total_min_diff)  # Min farkların en küçüğünü bul
        max_diff = max(total_max_diff)  # Max farkların en büyüğünü bul

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Hausdorff Distance: {avg_hausdorff}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Min Hausdorff Distance: {min_hausdorff}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Max Hausdorff Distance: {max_hausdorff}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Min Mask Difference: {min_diff}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Max Mask Difference: {max_diff}")

        # Her epoch sonunda test seti üzerinde değerlendirme yap
        test(model, test_loader, epoch, num_epochs)

    # Eğitim ve test sürecinin sonunda maskeleri görselleştir
    visualize_best_worst_masks(train_loader, is_train=True)
    visualize_best_worst_masks(test_loader, is_train=False)
            
def test(model, test_loader, epoch, num_epochs):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Örnek bir kayıp fonksiyonu, kullanımınıza göre değiştirebilirsiniz

    total_dice = 0.0
    total_jaccard = 0.0
    total_hausdorff = []
    total_min_diff = []
    total_max_diff = []
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            threshold = 0.5
            binary_preds = (torch.sigmoid(output) > threshold).float()

            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)

            dice = dice_coef(binary_preds, labels)
            jaccard = iou(binary_preds, labels)
            hausdorff_dist, _ = hausdorff_distance(binary_preds, labels)
            min_diff, max_diff = calculate_mask_difference(binary_preds, labels)

            total_dice += dice.item() * images.size(0)
            total_jaccard += jaccard.item() * images.size(0)
            total_hausdorff.append(hausdorff_dist)
            total_min_diff.append(min_diff)
            total_max_diff.append(max_diff)

            pbar.set_postfix({'Loss': total_loss / ((i + 1) * images.size(0)),  
                              'Dice': total_dice / ((i + 1) * images.size(0)), 
                              'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

        avg_hausdorff = torch.mean(torch.tensor(total_hausdorff))
        min_hausdorff = torch.min(torch.tensor(total_hausdorff))
        max_hausdorff = torch.max(torch.tensor(total_hausdorff))
        min_diff = min(total_min_diff)
        max_diff = max(total_max_diff)

        print(f"Epoch [{epoch}/{num_epochs}] - Average Hausdorff Distance: {avg_hausdorff}")
        print(f"Epoch [{epoch}/{num_epochs}] - Min Hausdorff Distance: {min_hausdorff}")
        print(f"Epoch [{epoch}/{num_epochs}] - Max Hausdorff Distance: {max_hausdorff}")
        print(f"Epoch [{epoch}/{num_epochs}] - Min Mask Difference: {min_diff}")
        print(f"Epoch [{epoch}/{num_epochs}] - Max Mask Difference: {max_diff}")

        
        # if current_epoch == total_epochs - 1:
        #     visualize_closest_to_average(test_loader, is_train=False)
        #     visualize_best_worst_masks(test_loader, is_train=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = num_epochs  # num_epochs değerini burada tanımladığınızı varsayalım
train(model, train_loader, test_loader, optimizer, num_epochs)

# visualize_best_worst_masks(train_loader, is_train=True)
# visualize_best_worst_masks(test_loader, is_train=False)
#%%

# smooth = 100

# def dice_coef(y_true, y_pred):
#     y_truef = torch.flatten(y_true)
#     y_predf = torch.flatten(y_pred)
#     intersection = torch.sum(y_truef * y_predf)
#     return ((2 * intersection + smooth) / (torch.sum(y_truef) + torch.sum(y_predf) + smooth))

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

# def iou(y_true, y_pred):
#     intersection = torch.sum(y_true * y_pred)
#     sum_ = torch.sum(y_true + y_pred)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return jac

# def jac_distance(y_true, y_pred):
#     return -iou(y_true, y_pred)


# import numpy as np
# from scipy.spatial.distance import cdist
# import torch

# def hausdorff_distance(true_mask, predicted_mask, pixel_scale=0.5):
#     """
#     Gerçek maske sınırından tahmin edilen maske sınırına olan Hausdorff uzaklığını hesaplar.

#     Parametreler:
#         true_mask: Gerçek maske (PyTorch tensor).
#         predicted_mask: Tahmin edilen maske (PyTorch tensor).
#         pixel_scale: 1 pikselin gerçek dünya değerini temsil eder.

#     Dönüş Değeri:
#         Max ve min mesafe farkları (mm cinsinden).
#     """
#     def find_boundary(mask):
#         return np.column_stack(np.where(mask))

#     # PyTorch tensorlerini NumPy array'lerine dönüştürün
#     true_mask_np = true_mask.cpu().numpy().astype(bool)
#     predicted_mask_np = predicted_mask.cpu().numpy().astype(bool)

#     # Maskelerin sınırlarını bulun
#     true_boundary = find_boundary(true_mask_np)
#     predicted_boundary = find_boundary(predicted_mask_np)

#     # Mesafeleri hesaplayın
#     distances_true_to_pred = cdist(true_boundary, predicted_boundary)
#     distances_pred_to_true = cdist(predicted_boundary, true_boundary)

#     # Max ve min mesafe farklarını hesaplayın
#     if len(distances_true_to_pred) > 0 and len(distances_pred_to_true) > 0:
#         max_distance_diff = max(np.max(np.min(distances_true_to_pred, axis=0)),
#                                 np.max(np.min(distances_pred_to_true, axis=0)))
#         min_distance_diff = min(np.min(np.min(distances_true_to_pred, axis=0)),
#                                 np.min(np.min(distances_pred_to_true, axis=0)))
#     else:
#         max_distance_diff = 0.0
#         min_distance_diff = 0.0

#     # Max ve min mesafe farklarını mm cinsine çevirin
#     max_distance_diff_mm = max_distance_diff * pixel_scale
#     min_distance_diff_mm = min_distance_diff * pixel_scale

#     return max_distance_diff_mm, min_distance_diff_mm


# def visualize_closest_to_average(loader, is_train):
#     dice_scores = []

#     for i, (images, labels) in enumerate(loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         output = model(images)
#         threshold = 0.5
#         binary_preds = (torch.sigmoid(output) > threshold).float()

#         dice = dice_coef(binary_preds, labels)
#         dice_scores.append((dice.item(), images, labels, binary_preds))

#     dice_scores.sort(key=lambda x: x[0])

#     closest_to_average = dice_scores[:]

#     if is_train:
#         title = "Train Set"
#     else:
#         title = "Test Set"

#     closest_to_average.sort(key=lambda x: x[0])  # Tüm skorları küçükten büyüğe sırala

#     # En yakın 5 skoru al
#     closest_to_average = closest_to_average[:5]

#     for dice_score, image, label, prediction in closest_to_average:
#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#         axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
#         axes[0].set_title('Input Image')
#         axes[1].imshow(label.squeeze().cpu().numpy(), cmap='gray')
#         axes[1].set_title('Ground Truth Mask')
#         axes[2].imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
#         axes[2].set_title('Predicted Mask')
#         plt.suptitle(f"{title} - Closest to Average Dice Score: {dice_score}")
#         plt.show()
 
        
# def visualize_best_worst_masks(loader, is_train):
#     dice_scores = []

#     for i, (images, labels) in enumerate(loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         output = model(images)
#         threshold = 0.5
#         binary_preds = (torch.sigmoid(output) > threshold).float()

#         dice = dice_coef(binary_preds, labels)
#         hausdorff_dist = hausdorff_distance(binary_preds, labels)

#         dice_scores.append((dice.item(), hausdorff_dist, images, labels, binary_preds))

#     dice_scores.sort(key=lambda x: x[0], reverse=is_train)

#     title = "Train Set" if is_train else "Test Set"

#     all_scores = dice_scores[:]
#     best_scores = all_scores[:5]
#     worst_scores = all_scores[-5:]
#     average_scores = all_scores[len(all_scores) // 2 - 2: len(all_scores) // 2 + 3]

#     for scores, title_text in [(best_scores, "Top 5 Best"), (worst_scores, "Top 5 Worst"), (average_scores, "Average")]:
#         for idx, (dice_score, hausdorff_dist, image, label, prediction) in enumerate(scores):
#             fig, axes = plt.subplots(1, 4, figsize=(16, 4))

#             axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
#             axes[0].set_title('Input Image')

#             axes[1].imshow(label.squeeze().cpu().numpy(), cmap='viridis')
#             axes[1].set_title('Ground Truth Mask')

#             axes[2].imshow(prediction.squeeze().cpu().numpy(), cmap='plasma')
#             axes[2].set_title('Predicted Mask')

#             mask_diff_1 = label.squeeze().cpu().numpy() - prediction.squeeze().cpu().numpy()
#             mask_diff_2 = prediction.squeeze().cpu().numpy() - label.squeeze().cpu().numpy()
#             mask_diff_3 = label.squeeze().cpu().numpy() - mask_diff_1
#             mask_diff_4 = prediction.squeeze().cpu().numpy() - mask_diff_2

#             axes[3].imshow(mask_diff_1, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_2, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_3, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_4, cmap='spring', alpha=0.5)
#             axes[3].set_title('Mask Difference')

#             max_dist = torch.max(torch.tensor(hausdorff_dist))
#             min_dist = torch.min(torch.tensor(hausdorff_dist))
#             axes[3].text(0.5, -0.15, f"Max Hausdorff Dist: {max_dist:.2f}", ha='center', transform=axes[3].transAxes)
#             axes[3].text(0.5, -0.2, f"Min Hausdorff Dist: {min_dist:.2f}", ha='center', transform=axes[3].transAxes)

#             plt.suptitle(f"{title} {title_text} Dice Scores: {dice_score}")

#             plt.show()
            
# import numpy as np

# def calculate_mask_difference(mask1, mask2, pixel_scale=0.5):
#     # İki maske arasındaki farkı hesapla
#     mask_difference = np.abs(mask1.cpu().numpy() - mask2.cpu().numpy())

#     # Farkın içindeki en küçük ve en büyük değerleri bul
#     min_difference = np.min(mask_difference) * pixel_scale
#     max_difference = np.max(mask_difference) * pixel_scale

#     return min_difference, max_difference

# def train(model, train_loader, optimizer, num_epochs):
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()

#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         total_dice = 0.0
#         total_jaccard = 0.0
#         total_hausdorff = []
#         total_min_diff = []  # Min farkları sakla
#         total_max_diff = []  # Max farkları sakla

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
#         for i, (images, labels) in enumerate(pbar):
#             images = images.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             output = model(images)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             threshold = 0.5
#             binary_preds = (torch.sigmoid(output) > threshold).float()

#             dice = dice_coef(binary_preds, labels)
#             jaccard = iou(binary_preds, labels)
#             hausdorff_dist, _ = hausdorff_distance(binary_preds, labels)  # Min ve Max farkları kullanmayacağız
#             min_diff, max_diff = calculate_mask_difference(binary_preds, labels)  # Maskeler arasındaki farkı hesapla

#             total_dice += dice.item() * images.size(0)
#             total_jaccard += jaccard.item() * images.size(0)
#             total_hausdorff.append(hausdorff_dist)
#             total_min_diff.append(min_diff)  # Her bir min farkını listeye ekle
#             total_max_diff.append(max_diff)  # Her bir max farkını listeye ekle

#             pbar.set_postfix({'Loss': running_loss / (i + 1), 
#                               'Dice': total_dice / ((i + 1) * images.size(0)), 
#                               'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

#         avg_hausdorff = torch.mean(torch.tensor(total_hausdorff))
#         min_hausdorff = torch.min(torch.tensor(total_hausdorff))
#         max_hausdorff = torch.max(torch.tensor(total_hausdorff))
#         min_diff = min(total_min_diff)  # Min farkların en küçüğünü bul
#         max_diff = max(total_max_diff)  # Max farkların en büyüğünü bul

#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Hausdorff Distance: {avg_hausdorff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Min Hausdorff Distance: {min_hausdorff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Max Hausdorff Distance: {max_hausdorff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Min Mask Difference: {min_diff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Max Mask Difference: {max_diff}")
        
#         if epoch == num_epochs - 1:
#             visualize_closest_to_average(train_loader, is_train=True)
#             visualize_best_worst_masks(train_loader, is_train=True)
            
# def test(model, test_loader, current_epoch, total_epochs):
#     model.to(device)
#     total_dice = 0.0
#     total_jaccard = 0.0
#     total_hausdorff = []
#     total_min_diff = []  # Min farkları sakla
#     total_max_diff = []  # Max farkları sakla

#     with torch.no_grad():
#         pbar = tqdm(test_loader, desc="Testing")
#         for i, (images, labels) in enumerate(pbar):
#             images = images.to(device)
#             labels = labels.to(device)

#             output = model(images)

#             threshold = 0.5
#             binary_preds = (torch.sigmoid(output) > threshold).float()

#             dice = dice_coef(binary_preds, labels)
#             jaccard = iou(binary_preds, labels)
#             hausdorff_dist, _ = hausdorff_distance(binary_preds, labels)  # Min ve Max farkları kullanmayacağız
#             min_diff, max_diff = calculate_mask_difference(binary_preds, labels)  # Maskeler arasındaki farkı hesapla

#             total_dice += dice.item() * images.size(0)
#             total_jaccard += jaccard.item() * images.size(0)
#             total_hausdorff.append(hausdorff_dist)
#             total_min_diff.append(min_diff)  # Her bir min farkını listeye ekle
#             total_max_diff.append(max_diff)  # Her bir max farkını listeye ekle

#             pbar.set_postfix({'Dice': total_dice / ((i + 1) * images.size(0)), 
#                               'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

#         avg_hausdorff = torch.mean(torch.tensor(total_hausdorff))
#         min_hausdorff = torch.min(torch.tensor(total_hausdorff))
#         max_hausdorff = torch.max(torch.tensor(total_hausdorff))
#         min_diff = min(total_min_diff)  # Min farkların en küçüğünü bul
#         max_diff = max(total_max_diff)  # Max farkların en büyüğünü bul

#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Average Hausdorff Distance: {avg_hausdorff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Min Hausdorff Distance: {min_hausdorff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Max Hausdorff Distance: {max_hausdorff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Min Mask Difference: {min_diff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Max Mask Difference: {max_diff}")
        
#         if current_epoch == total_epochs - 1:
#             visualize_closest_to_average(test_loader, is_train=False)
#             visualize_best_worst_masks(test_loader, is_train=False)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # num_epochs = num_epochs  # num_epochs değerini burada tanımladığınızı varsayalım
# # train(model, train_loader, optimizer, num_epochs)
# # test(model, test_loader, num_epochs - 1, num_epochs)

# visualize_best_worst_masks(train_loader, is_train=True)
# visualize_best_worst_masks(test_loader, is_train=False)
# #%%

# import torch
# from scipy.spatial.distance import directed_hausdorff
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# smooth = 100

# def dice_coef(y_true, y_pred):
#     y_truef = torch.flatten(y_true)
#     y_predf = torch.flatten(y_pred)
#     intersection = torch.sum(y_truef * y_predf)
#     return ((2 * intersection + smooth) / (torch.sum(y_truef) + torch.sum(y_predf) + smooth))

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

# def iou(y_true, y_pred):
#     intersection = torch.sum(y_true * y_pred)
#     sum_ = torch.sum(y_true + y_pred)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return jac

# def jac_distance(y_true, y_pred):
#     return -iou(y_true, y_pred)


# import numpy as np
# from scipy.spatial.distance import cdist
# import torch

# def hausdorff_distance(true_mask, predicted_mask, pixel_scale=0.5):
#     """
#     Gerçek maske sınırından tahmin edilen maske sınırına olan Hausdorff uzaklığını hesaplar.

#     Parametreler:
#         true_mask: Gerçek maske (PyTorch tensor).
#         predicted_mask: Tahmin edilen maske (PyTorch tensor).
#         pixel_scale: 1 pikselin gerçek dünya değerini temsil eder.

#     Dönüş Değeri:
#         Max ve min mesafe farkları (mm cinsinden).
#     """
#     def find_boundary(mask):
#         return np.column_stack(np.where(mask))

#     # PyTorch tensorlerini NumPy array'lerine dönüştürün
#     true_mask_np = true_mask.cpu().numpy().astype(bool)
#     predicted_mask_np = predicted_mask.cpu().numpy().astype(bool)

#     # Maskelerin sınırlarını bulun
#     true_boundary = find_boundary(true_mask_np)
#     predicted_boundary = find_boundary(predicted_mask_np)

#     # Mesafeleri hesaplayın
#     distances_true_to_pred = cdist(true_boundary, predicted_boundary)
#     distances_pred_to_true = cdist(predicted_boundary, true_boundary)

#     # Max ve min mesafe farklarını hesaplayın
#     if len(distances_true_to_pred) > 0 and len(distances_pred_to_true) > 0:
#         max_distance_diff = max(np.max(np.min(distances_true_to_pred, axis=0)),
#                                 np.max(np.min(distances_pred_to_true, axis=0)))
#         min_distance_diff = min(np.min(np.min(distances_true_to_pred, axis=0)),
#                                 np.min(np.min(distances_pred_to_true, axis=0)))
#     else:
#         max_distance_diff = 0.0
#         min_distance_diff = 0.0

#     # Max ve min mesafe farklarını mm cinsine çevirin
#     max_distance_diff_mm = max_distance_diff * pixel_scale
#     min_distance_diff_mm = min_distance_diff * pixel_scale

#     return max_distance_diff_mm, min_distance_diff_mm






# # def hausdorff_distance(y_true, y_pred, pixel_scale=0.5):
# #     y_true_np = y_true.cpu().numpy()
# #     y_pred_np = y_pred.cpu().numpy()

# #     # Tensorlar 4 boyutlu olabilir, ancak Hausdorff mesafesi 2 boyutlu kabul eder.
# #     # Dolayısıyla gerektiğinde tensorları 2 boyuta düşürebiliriz.
# #     y_true_np = y_true_np.squeeze() if len(y_true_np.shape) == 4 else y_true_np
# #     y_pred_np = y_pred_np.squeeze() if len(y_pred_np.shape) == 4 else y_pred_np

# #     # Ölçeklendirme işlemi (0.5 mm cinsinden)
# #     y_true_np_scaled = y_true_np * pixel_scale
# #     y_pred_np_scaled = y_pred_np * pixel_scale

# #     hausdorff_dist = directed_hausdorff(y_true_np_scaled, y_pred_np_scaled)[0]
# #     return hausdorff_dist

# # from scipy.spatial.distance import cdist


# # def hausdorff_distance(y_true, y_pred, pixel_scale=0.5):
# #     def create_coordinate_array(mask):
# #         mask_np = mask.cpu().numpy()
# #         mask_np = mask_np.squeeze() if len(mask_np.shape) == 4 else mask_np
# #         return np.column_stack(np.where(mask_np > 0))

# #     true_coords = create_coordinate_array(y_true)
# #     pred_coords = create_coordinate_array(y_pred)

# #     if len(true_coords) == 0 or len(pred_coords) == 0:
# #         return 0.0, 0.0

# #     distances = cdist(true_coords, pred_coords)
# #     min_dist = np.min(distances) * pixel_scale
# #     max_dist = np.max(distances) * pixel_scale

# #     return min_dist, max_dist


# # def hausdorff_distance(y_true, y_pred, pixel_scale=0.5):
# #     y_true_np = y_true.cpu().numpy()
# #     y_pred_np = y_pred.cpu().numpy()

# #     y_true_np = y_true_np.squeeze() if len(y_true_np.shape) == 4 else y_true_np
# #     y_pred_np = y_pred_np.squeeze() if len(y_pred_np.shape) == 4 else y_pred_np

# #     true_coords = np.column_stack(np.where(y_true_np > 0))
# #     pred_coords = np.column_stack(np.where(y_pred_np > 0))

# #     if len(true_coords) == 0 or len(pred_coords) == 0:
# #         # Boş diziler varsa uygun bir değer döndür
# #         return 0.0, 0.0

# #     distances = cdist(true_coords, pred_coords)

# #     min_dist = np.min(distances) * pixel_scale
# #     max_dist = np.max(distances) * pixel_scale

# #     return min_dist, max_dist






# def visualize_closest_to_average(loader, is_train):
#     dice_scores = []

#     for i, (images, labels) in enumerate(loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         output = model(images)
#         threshold = 0.5
#         binary_preds = (torch.sigmoid(output) > threshold).float()

#         dice = dice_coef(binary_preds, labels)
#         dice_scores.append((dice.item(), images, labels, binary_preds))

#     dice_scores.sort(key=lambda x: x[0])

#     closest_to_average = dice_scores[:]

#     if is_train:
#         title = "Train Set"
#     else:
#         title = "Test Set"

#     closest_to_average.sort(key=lambda x: x[0])  # Tüm skorları küçükten büyüğe sırala

#     # En yakın 5 skoru al
#     closest_to_average = closest_to_average[:5]

#     for dice_score, image, label, prediction in closest_to_average:
#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#         axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
#         axes[0].set_title('Input Image')
#         axes[1].imshow(label.squeeze().cpu().numpy(), cmap='gray')
#         axes[1].set_title('Ground Truth Mask')
#         axes[2].imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
#         axes[2].set_title('Predicted Mask')
#         plt.suptitle(f"{title} - Closest to Average Dice Score: {dice_score}")
#         plt.show()
 
        
# def visualize_best_worst_masks(loader, is_train):
#     dice_scores = []

#     for i, (images, labels) in enumerate(loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         output = model(images)
#         threshold = 0.5
#         binary_preds = (torch.sigmoid(output) > threshold).float()

#         dice = dice_coef(binary_preds, labels)
#         hausdorff_dist = hausdorff_distance(binary_preds, labels)

#         dice_scores.append((dice.item(), hausdorff_dist, images, labels, binary_preds))

#     dice_scores.sort(key=lambda x: x[0], reverse=is_train)

#     title = "Train Set" if is_train else "Test Set"

#     all_scores = dice_scores[:]
#     best_scores = all_scores[:5]
#     worst_scores = all_scores[-5:]
#     average_scores = all_scores[len(all_scores) // 2 - 2: len(all_scores) // 2 + 3]

#     for scores, title_text in [(best_scores, "Top 5 Best"), (worst_scores, "Top 5 Worst"), (average_scores, "Average")]:
#         for idx, (dice_score, hausdorff_dist, image, label, prediction) in enumerate(scores):
#             fig, axes = plt.subplots(1, 4, figsize=(16, 4))

#             axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
#             axes[0].set_title('Input Image')

#             axes[1].imshow(label.squeeze().cpu().numpy(), cmap='viridis')
#             axes[1].set_title('Ground Truth Mask')

#             axes[2].imshow(prediction.squeeze().cpu().numpy(), cmap='plasma')
#             axes[2].set_title('Predicted Mask')

#             mask_diff_1 = label.squeeze().cpu().numpy() - prediction.squeeze().cpu().numpy()
#             mask_diff_2 = prediction.squeeze().cpu().numpy() - label.squeeze().cpu().numpy()
#             mask_diff_3 = label.squeeze().cpu().numpy() - mask_diff_1
#             mask_diff_4 = prediction.squeeze().cpu().numpy() - mask_diff_2

#             axes[3].imshow(mask_diff_1, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_2, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_3, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_4, cmap='spring', alpha=0.5)
#             axes[3].set_title('Mask Difference')

#             max_dist = torch.max(torch.tensor(hausdorff_dist))
#             min_dist = torch.min(torch.tensor(hausdorff_dist))
#             axes[3].text(0.5, -0.15, f"Max Hausdorff Dist: {max_dist:.2f}", ha='center', transform=axes[3].transAxes)
#             axes[3].text(0.5, -0.2, f"Min Hausdorff Dist: {min_dist:.2f}", ha='center', transform=axes[3].transAxes)

#             plt.suptitle(f"{title} {title_text} Dice Scores: {dice_score}")

#             plt.show()
            
# import numpy as np

# def calculate_mask_difference(mask1, mask2, pixel_scale=0.5):
#     # İki maske arasındaki farkı hesapla
#     mask_difference = np.abs(mask1.cpu().numpy() - mask2.cpu().numpy())

#     # Farkın içindeki en küçük ve en büyük değerleri bul
#     min_difference = np.min(mask_difference) * pixel_scale
#     max_difference = np.max(mask_difference) * pixel_scale

#     return min_difference, max_difference


# def train(model, train_loader, optimizer, num_epochs):
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         total_dice = 0.0
#         total_jaccard = 0.0
#         total_hausdorff = []
#         total_min_diff = []  # Min farkları sakla
#         total_max_diff = []  # Max farkları sakla

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
#         for i, (images, labels) in enumerate(pbar):
#             images = images.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             output = model(images)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             threshold = 0.5
#             binary_preds = (torch.sigmoid(output) > threshold).float()

#             dice = dice_coef(binary_preds, labels)
#             jaccard = iou(binary_preds, labels)
#             hausdorff_dist = hausdorff_distance(binary_preds, labels)
#             min_diff, max_diff = calculate_mask_difference(binary_preds, labels)  # Maskeler arasındaki farkı hesapla

#             total_dice += dice.item() * images.size(0)
#             total_jaccard += jaccard.item() * images.size(0)
#             total_hausdorff.append(hausdorff_dist)
#             total_min_diff.append(min_diff)  # Her bir min farkını listeye ekle
#             total_max_diff.append(max_diff)  # Her bir max farkını listeye ekle

#             pbar.set_postfix({'Loss': running_loss / (i + 1), 
#                               'Dice': total_dice / ((i + 1) * images.size(0)), 
#                               'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

#         avg_hausdorff = torch.mean(torch.tensor(total_hausdorff))
#         min_hausdorff = torch.min(torch.tensor(total_hausdorff))
#         max_hausdorff = torch.max(torch.tensor(total_hausdorff))
#         min_diff = min(total_min_diff)  # Min farkların en küçüğünü bul
#         max_diff = max(total_max_diff)  # Max farkların en büyüğünü bul

#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Hausdorff Distance: {avg_hausdorff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Min Hausdorff Distance: {min_hausdorff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Max Hausdorff Distance: {max_hausdorff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Min Mask Difference: {min_diff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Max Mask Difference: {max_diff}")
        
#         if epoch == num_epochs - 1:
#             visualize_closest_to_average(train_loader, is_train=True)
#             visualize_best_worst_masks(train_loader, is_train=True)


# def test(model, test_loader, current_epoch, total_epochs):
#     model.to(device)
#     total_dice = 0.0
#     total_jaccard = 0.0
#     total_hausdorff = []
#     total_min_diff = []  # Min farkları sakla
#     total_max_diff = []  # Max farkları sakla

#     with torch.no_grad():
#         pbar = tqdm(test_loader, desc="Testing")
#         for i, (images, labels) in enumerate(pbar):
#             images = images.to(device)
#             labels = labels.to(device)

#             output = model(images)

#             threshold = 0.5
#             binary_preds = (torch.sigmoid(output) > threshold).float()

#             dice = dice_coef(binary_preds, labels)
#             jaccard = iou(binary_preds, labels)
#             hausdorff_dist = hausdorff_distance(binary_preds, labels)
#             min_diff, max_diff = calculate_mask_difference(binary_preds, labels)  # Maskeler arasındaki farkı hesapla

#             total_dice += dice.item() * images.size(0)
#             total_jaccard += jaccard.item() * images.size(0)
#             total_hausdorff.append(hausdorff_dist)
#             total_min_diff.append(min_diff)  # Her bir min farkını listeye ekle
#             total_max_diff.append(max_diff)  # Her bir max farkını listeye ekle

#             pbar.set_postfix({'Dice': total_dice / ((i + 1) * images.size(0)), 
#                               'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

#         avg_hausdorff = torch.mean(torch.tensor(total_hausdorff))
#         min_hausdorff = torch.min(torch.tensor(total_hausdorff))
#         max_hausdorff = torch.max(torch.tensor(total_hausdorff))
#         min_diff = min(total_min_diff)  # Min farkların en küçüğünü bul
#         max_diff = max(total_max_diff)  # Max farkların en büyüğünü bul

#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Average Hausdorff Distance: {avg_hausdorff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Min Hausdorff Distance: {min_hausdorff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Max Hausdorff Distance: {max_hausdorff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Min Mask Difference: {min_diff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Max Mask Difference: {max_diff}")
        
#         if current_epoch == total_epochs - 1:
#             visualize_closest_to_average(test_loader, is_train=False)
#             visualize_best_worst_masks(test_loader, is_train=False)
            
            

            
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_epochs = num_epochs  # num_epochs değerini burada tanımladığınızı varsayalım
# train(model, train_loader, optimizer, num_epochs)
# test(model, test_loader, num_epochs - 1, num_epochs)
# # visualize_best_worst_masks(test_loader, is_train=True)        
# #%%

# import torch
# from scipy.spatial.distance import directed_hausdorff
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# smooth = 100

# def dice_coef(y_true, y_pred):
#     y_truef = torch.flatten(y_true)
#     y_predf = torch.flatten(y_pred)
#     intersection = torch.sum(y_truef * y_predf)
#     return ((2 * intersection + smooth) / (torch.sum(y_truef) + torch.sum(y_predf) + smooth))

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

# def iou(y_true, y_pred):
#     intersection = torch.sum(y_true * y_pred)
#     sum_ = torch.sum(y_true + y_pred)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return jac

# def jac_distance(y_true, y_pred):
#     return -iou(y_true, y_pred)

# # def hausdorff_distance(y_true, y_pred, pixel_scale=0.5):
# #     y_true_np = y_true.cpu().numpy()
# #     y_pred_np = y_pred.cpu().numpy()

# #     # Tensorlar 4 boyutlu olabilir, ancak Hausdorff mesafesi 2 boyutlu kabul eder.
# #     # Dolayısıyla gerektiğinde tensorları 2 boyuta düşürebiliriz.
# #     y_true_np = y_true_np.squeeze() if len(y_true_np.shape) == 4 else y_true_np
# #     y_pred_np = y_pred_np.squeeze() if len(y_pred_np.shape) == 4 else y_pred_np

# #     # Ölçeklendirme işlemi (0.5 mm cinsinden)
# #     y_true_np_scaled = y_true_np * pixel_scale
# #     y_pred_np_scaled = y_pred_np * pixel_scale

# #     hausdorff_dist = directed_hausdorff(y_true_np_scaled, y_pred_np_scaled)[0]
# #     return hausdorff_dist

# from scipy.spatial.distance import cdist

# def hausdorff_distance(y_true, y_pred, pixel_scale=0.5):
#     y_true_np = y_true.cpu().numpy()
#     y_pred_np = y_pred.cpu().numpy()

#     y_true_np = y_true_np.squeeze() if len(y_true_np.shape) == 4 else y_true_np
#     y_pred_np = y_pred_np.squeeze() if len(y_pred_np.shape) == 4 else y_pred_np

#     true_coords = np.column_stack(np.where(y_true_np > 0))
#     pred_coords = np.column_stack(np.where(y_pred_np > 0))

#     if len(true_coords) == 0 or len(pred_coords) == 0:
#         # Boş diziler varsa uygun bir değer döndür
#         return 0.0, 0.0

#     distances = cdist(true_coords, pred_coords)

#     min_dist = np.min(distances) * pixel_scale
#     max_dist = np.max(distances) * pixel_scale

#     return min_dist, max_dist




# def visualize_closest_to_average(loader, is_train):
#     dice_scores = []

#     for i, (images, labels) in enumerate(loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         output = model(images)
#         threshold = 0.5
#         binary_preds = (torch.sigmoid(output) > threshold).float()

#         dice = dice_coef(binary_preds, labels)
#         dice_scores.append((dice.item(), images, labels, binary_preds))

#     dice_scores.sort(key=lambda x: x[0])

#     closest_to_average = dice_scores[:]

#     if is_train:
#         title = "Train Set"
#     else:
#         title = "Test Set"

#     closest_to_average.sort(key=lambda x: x[0])  # Tüm skorları küçükten büyüğe sırala

#     # En yakın 5 skoru al
#     closest_to_average = closest_to_average[:5]

#     for dice_score, image, label, prediction in closest_to_average:
#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#         axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
#         axes[0].set_title('Input Image')
#         axes[1].imshow(label.squeeze().cpu().numpy(), cmap='gray')
#         axes[1].set_title('Ground Truth Mask')
#         axes[2].imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
#         axes[2].set_title('Predicted Mask')
#         plt.suptitle(f"{title} - Closest to Average Dice Score: {dice_score}")
#         plt.show()


# # import torch
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Diğer fonksiyonlarınızı ve değişkenlerinizi buraya ekleyin.

# # def visualize_best_worst_masks(loader, is_train):
# #     dice_scores = []

# #     for i, (images, labels) in enumerate(loader):
# #         images = images.to(device)
# #         labels = labels.to(device)

# #         output = model(images)
# #         threshold = 0.5
# #         binary_preds = (torch.sigmoid(output) > threshold).float()

# #         dice = dice_coef(binary_preds, labels)
# #         hausdorff_dist = hausdorff_distance(binary_preds, labels)

# #         dice_scores.append((dice.item(), hausdorff_dist, images, labels, binary_preds))

# #     dice_scores.sort(key=lambda x: x[0], reverse=is_train)

# #     title = "Train Set" if is_train else "Test Set"

# #     all_scores = dice_scores[:]
# #     best_scores = all_scores[:5]
# #     worst_scores = all_scores[-5:]
# #     average_scores = all_scores[len(all_scores) // 2 - 2: len(all_scores) // 2 + 3]

# #     for scores, title_text in [(best_scores, "Top 5 Best"), (worst_scores, "Top 5 Worst"), (average_scores, "Average")]:
# #         for idx, (dice_score, hausdorff_dist, image, label, prediction) in enumerate(scores):
# #             fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# #             # Gerçek maske ve tahmin edilen maskeyi birleştirme
# #             combined_masks = torch.cat([labels, binary_preds], dim=1)
# #             combined_masks = combined_masks.squeeze().cpu().numpy()

# #             # Farkları hesaplama
# #             mask_diff = torch.abs(labels - binary_preds)
# #             mask_diff = mask_diff.squeeze().cpu().numpy()

# #             axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
# #             axes[0].set_title('Input Image')

# #             # Gerçek maske mavi, tahmin edilen maske sarı renkte
# #             axes[1].imshow(combined_masks, cmap='viridis', alpha=0.5)
# #             axes[1].set_title('Ground Truth vs Predicted Mask')

# #             # Farklı kısımları farklı bir renkte gösterme
# #             axes[2].imshow(mask_diff, cmap='spring')
# #             axes[2].set_title('Mask Difference')

# #             # Max ve min Hausdorff mesafelerini gösterme
# #             max_dist = torch.max(torch.tensor(hausdorff_dist))
# #             min_dist = torch.min(torch.tensor(hausdorff_dist))
# #             axes[2].text(0.5, -0.1, f"Max Hausdorff Dist: {max_dist:.2f}", ha='center', transform=axes[2].transAxes)
# #             axes[2].text(0.5, -0.15, f"Min Hausdorff Dist: {min_dist:.2f}", ha='center', transform=axes[2].transAxes)

# #             # Renk bilgilendirmesi ekleme
# #             plt.colorbar(plt.imshow(mask_diff, cmap='spring'), ax=axes[2], fraction=0.046, pad=0.04)
# #             plt.suptitle(f"{title} Dice Scores: {dice_score}")

# #             # Sağ üst köşeye renklerin açıklamasını ekleme
# #             axes[1].text(0.95, 0.95, 'Ground Truth Mask: Blue\nPredicted Mask: Yellow\nMask Difference: Other', color='black', ha='right', va='top', transform=axes[1].transAxes, bbox=dict(facecolor='white', alpha=0.5))
# #             plt.show()
# def visualize_best_worst_masks(loader, is_train):
#     dice_scores = []

#     for i, (images, labels) in enumerate(loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         output = model(images)
#         threshold = 0.5
#         binary_preds = (torch.sigmoid(output) > threshold).float()

#         dice = dice_coef(binary_preds, labels)
#         hausdorff_dist = hausdorff_distance(binary_preds, labels)

#         dice_scores.append((dice.item(), hausdorff_dist, images, labels, binary_preds))

#     dice_scores.sort(key=lambda x: x[0], reverse=is_train)

#     title = "Train Set" if is_train else "Test Set"

#     all_scores = dice_scores[:]
#     best_scores = all_scores[:5]
#     worst_scores = all_scores[-5:]
#     average_scores = all_scores[len(all_scores) // 2 - 2: len(all_scores) // 2 + 3]

#     for scores, title_text in [(best_scores, "Top 5 Best"), (worst_scores, "Top 5 Worst"), (average_scores, "Average")]:
#         for idx, (dice_score, hausdorff_dist, image, label, prediction) in enumerate(scores):
#             fig, axes = plt.subplots(1, 4, figsize=(16, 4))

#             axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
#             axes[0].set_title('Input Image')

#             axes[1].imshow(label.squeeze().cpu().numpy(), cmap='viridis')
#             axes[1].set_title('Ground Truth Mask')

#             axes[2].imshow(prediction.squeeze().cpu().numpy(), cmap='plasma')
#             axes[2].set_title('Predicted Mask')

#             mask_diff_1 = label.squeeze().cpu().numpy() - prediction.squeeze().cpu().numpy()
#             mask_diff_2 = prediction.squeeze().cpu().numpy() - label.squeeze().cpu().numpy()
#             mask_diff_3 = label.squeeze().cpu().numpy() - mask_diff_1
#             mask_diff_4 = prediction.squeeze().cpu().numpy() - mask_diff_2

#             axes[3].imshow(mask_diff_1, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_2, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_3, cmap='spring', alpha=0.5)
#             axes[3].imshow(mask_diff_4, cmap='spring', alpha=0.5)
#             axes[3].set_title('Mask Difference')

#             max_dist = torch.max(torch.tensor(hausdorff_dist))
#             min_dist = torch.min(torch.tensor(hausdorff_dist))
#             axes[3].text(0.5, -0.1, f"Max Hausdorff Dist: {max_dist:.2f}", ha='center', transform=axes[3].transAxes)
#             axes[3].text(0.5, -0.15, f"Min Hausdorff Dist: {min_dist:.2f}", ha='center', transform=axes[3].transAxes)

#             plt.suptitle(f"{title} {title_text} Dice Scores: {dice_score}")

#             plt.show()


        
# def train(model, train_loader, optimizer, num_epochs):
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         total_dice = 0.0
#         total_jaccard = 0.0
#         total_hausdorff = []  # Listeyi tanımla

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
#         for i, (images, labels) in enumerate(pbar):
#             images = images.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             output = model(images)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             threshold = 0.5
#             binary_preds = (torch.sigmoid(output) > threshold).float()

#             dice = dice_coef(binary_preds, labels)
#             jaccard = iou(binary_preds, labels)
#             hausdorff_dist = hausdorff_distance(binary_preds, labels)

#             total_dice += dice.item() * images.size(0)
#             total_jaccard += jaccard.item() * images.size(0)
#             total_hausdorff.append(hausdorff_dist)  # Her bir Hausdorff mesafesini listeye ekle

#             pbar.set_postfix({'Loss': running_loss / (i + 1), 
#                               'Dice': total_dice / ((i + 1) * images.size(0)), 
#                               'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

#         avg_hausdorff = torch.mean(torch.tensor(total_hausdorff))
#         min_hausdorff = torch.min(torch.tensor(total_hausdorff))
#         max_hausdorff = torch.max(torch.tensor(total_hausdorff))

#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Hausdorff Distance: {avg_hausdorff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Min Hausdorff Distance: {min_hausdorff}")
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Max Hausdorff Distance: {max_hausdorff}")

#         if epoch == num_epochs - 1:
#             visualize_closest_to_average(train_loader, is_train=True)
#             visualize_best_worst_masks(train_loader, is_train=True)

        
# def test(model, test_loader, current_epoch, total_epochs):
#     model.to(device)
#     total_dice = 0.0
#     total_jaccard = 0.0
#     total_hausdorff = []

#     with torch.no_grad():
#         pbar = tqdm(test_loader, desc="Testing")
#         for i, (images, labels) in enumerate(pbar):
#             images = images.to(device)
#             labels = labels.to(device)

#             output = model(images)

#             threshold = 0.5
#             binary_preds = (torch.sigmoid(output) > threshold).float()

#             dice = dice_coef(binary_preds, labels)
#             jaccard = iou(binary_preds, labels)
#             hausdorff_dist = hausdorff_distance(binary_preds, labels)
#             total_hausdorff.append(hausdorff_dist)  # Her bir Hausdorff mesafesini listeye ekle

#             total_dice += dice.item() * images.size(0)
#             total_jaccard += jaccard.item() * images.size(0)

#             pbar.set_postfix({'Dice': total_dice / ((i + 1) * images.size(0)), 
#                               'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

#         avg_hausdorff = torch.mean(torch.tensor(total_hausdorff))
#         min_hausdorff = torch.min(torch.tensor(total_hausdorff))
#         max_hausdorff = torch.max(torch.tensor(total_hausdorff))

#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Average Hausdorff Distance: {avg_hausdorff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Min Hausdorff Distance: {min_hausdorff}")
#         print(f"Epoch [{current_epoch + 1}/{total_epochs}] - Max Hausdorff Distance: {max_hausdorff}")

#         if current_epoch == total_epochs - 1:
#             visualize_closest_to_average(test_loader, is_train=False)
#             visualize_best_worst_masks(test_loader, is_train=False)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_epochs = num_epochs  # num_epochs değerini burada tanımladığınızı varsayalım
# train(model, train_loader, optimizer, num_epochs)
# test(model, test_loader, num_epochs - 1, num_epochs)
# # visualize_best_worst_masks(test_loader, is_train=False)

# #%%

# torch.save(model.state_dict(), 'egitilmis_model.pth')

# #%%

# model.load_state_dict(torch.load('egitilmis_model.pth'))
# model.eval()

# def visualize_best_worst_masks(loader, is_train):
#     dice_scores = []

#     for i, (images, labels) in enumerate(loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         output = model(images)
#         threshold = 0.5
#         binary_preds = (torch.sigmoid(output) > threshold).float()

#         dice = dice_coef(binary_preds, labels)
#         hausdorff_dist = hausdorff_distance(binary_preds, labels)

#         dice_scores.append((dice.item(), hausdorff_dist, images, labels, binary_preds))

#     dice_scores.sort(key=lambda x: x[0], reverse=is_train)

#     title = "Train Set" if is_train else "Test Set"

#     all_scores = dice_scores[:]
#     best_scores = all_scores[:5]
#     worst_scores = all_scores[-5:]
#     average_scores = all_scores[len(all_scores) // 2 - 2: len(all_scores) // 2 + 3]

#     for scores, title_text in [(best_scores, "Top 5 Best"), (worst_scores, "Top 5 Worst"), (average_scores, "Average")]:
#         for idx, (dice_score, hausdorff_dist, image, label, prediction) in enumerate(scores):
#             fig, axes = plt.subplots(1, 3, figsize=(12, 4))

#             # Gerçek maske ve tahmin edilen maskeyi birleştirme
#             combined_masks = torch.cat([labels, binary_preds], dim=1)
#             combined_masks = combined_masks.squeeze().cpu().numpy()

#             # Farkları hesaplama
#             mask_diff = torch.abs(labels - binary_preds)
#             mask_diff = mask_diff.squeeze().cpu().numpy()

#             axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
#             axes[0].set_title('Input Image')

#             # Gerçek maske mavi, tahmin edilen maske sarı renkte
#             axes[1].imshow(combined_masks, cmap='viridis', alpha=0.5)
#             axes[1].set_title('Ground Truth vs Predicted Mask')

#             # Farklı kısımları farklı bir renkte gösterme
#             axes[2].imshow(mask_diff, cmap='spring')
#             axes[2].set_title('Mask Difference')

#             # Max ve min Hausdorff mesafelerini gösterme
#             max_dist = torch.max(torch.tensor(hausdorff_dist))
#             min_dist = torch.min(torch.tensor(hausdorff_dist))
#             axes[2].text(0.5, -0.1, f"Max Hausdorff Dist: {max_dist:.2f}", ha='center', transform=axes[2].transAxes)
#             axes[2].text(0.5, -0.15, f"Min Hausdorff Dist: {min_dist:.2f}", ha='center', transform=axes[2].transAxes)

#             # Renk bilgilendirmesi ekleme
#             plt.colorbar(plt.imshow(mask_diff, cmap='spring'), ax=axes[2], fraction=0.046, pad=0.04)
#             plt.suptitle(f"{title} Dice Scores: {dice_score}")

#             # Sağ üst köşeye renklerin açıklamasını ekleme
#             axes[1].text(0.95, 0.95, 'Ground Truth Mask: Blue\nPredicted Mask: Yellow\nMask Difference: Other', color='black', ha='right', va='top', transform=axes[1].transAxes, bbox=dict(facecolor='white', alpha=0.5))
#             plt.show()

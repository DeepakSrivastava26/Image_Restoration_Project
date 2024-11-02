import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision.models as models
from torch.utils.data import Dataset,DataLoader, random_split
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import cv2

#Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.global_avg_pool(x))
        max_out = self.fc(self.global_max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
#Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

#CBAM Module
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

#Self Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        query = self.query_conv(x).view(batch_size, C, -1)
        key = self.key_conv(x).view(batch_size, C, -1)
        value = self.value_conv(x).view(batch_size, C, -1)


        scores = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)

        out = torch.bmm(value, scores.transpose(1, 2))
        out = out.view(batch_size, C, width, height)

        return self.gamma * out + x
    
#Model
class DenoisingNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(DenoisingNet, self).__init__()
        #Conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(dropout_rate)

        #Conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(dropout_rate)

        #Conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.CBAM3 = CBAM(32,32,7)

        #Conv4
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,dilation = 2, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.CBAM4 = CBAM(64,64,7)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.selfattn1 = SelfAttention(64)
        self.selfattn2 = SelfAttention(64)

        #Deconv1
        self.conv1Trans = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3,dilation = 2, stride=1, padding=0)
        self.bn1Trans = nn.BatchNorm2d(32)
        self.dropout1Trans = nn.Dropout(dropout_rate)
        self.CBAMTrans1 = CBAM(32,32,7)

        #Deconv2
        self.conv2Trans = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.bn2Trans = nn.BatchNorm2d(32)
        self.dropout2Trans = nn.Dropout(dropout_rate)
        self.CBAMTrans2 = CBAM(32,32,7)

        #Deconv3
        self.conv3Trans = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn3Trans = nn.BatchNorm2d(16)
        self.dropout3Trans = nn.Dropout(dropout_rate)

        #Deconv4
        self.conv4Trans = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=7, stride=1, padding=0)
        self.bn4Trans = nn.BatchNorm2d(3)
        self.dropout4Trans = nn.Dropout(dropout_rate)

        #Deconv5
        self.convoutput = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.maxUnpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.PReLU = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    # Encoder
      x = self.PReLU(self.bn1(self.conv1(x)))
      x = self.dropout1(x)
      conv_1 = x.clone()
      x, idx_1 = self.maxPool(x)

      x = self.PReLU(self.bn2(self.conv2(x)))
      x = self.dropout2(x)
      conv_2 = x.clone()
      x, idx_2 = self.maxPool(x)

      x = self.PReLU(self.bn3(self.conv3(x)))
      x = self.dropout3(x)
      x = self.CBAM3(x)
      conv_3 = x.clone()
      x, idx_3 = self.maxPool(x)

      x = self.PReLU(self.bn4(self.conv4(x)))
      x = self.dropout4(x)
      x = self.CBAM4(x)
      conv_4 = x.clone()
      x, idx_4 = self.maxPool(x)

      x = self.selfattn1(x)

    # Decoder
      x = self.maxUnpool(x, idx_4,output_size = conv_4.size())
      conv_4 = self.selfattn2(conv_4)
      concat = torch.concat((x,conv_4.clone()),dim = 1)
      x = self.PReLU(self.bn1Trans(self.conv1Trans(concat)))
      x = self.dropout1Trans(x)
      x = self.CBAMTrans1(x)

      x = self.maxUnpool(x, idx_3,output_size = conv_3.size())
      concat = torch.concat((x,conv_3.clone()),dim = 1)
      x = self.PReLU(self.bn2Trans(self.conv2Trans(concat)))
      x = self.dropout2Trans(x)
      x = self.CBAMTrans2(x)

      x = self.maxUnpool(x, idx_2,output_size = conv_2.size())
      concat = torch.concat((x,conv_2.clone()),dim = 1)
      x = self.PReLU(self.bn3Trans(self.conv3Trans(concat)))
      x = self.dropout3Trans(x)

      x = self.maxUnpool(x, idx_1,output_size = conv_1.size())
      concat = torch.concat((x,conv_1.clone()),dim = 1)
      x = self.PReLU(self.bn4Trans(self.conv4Trans(concat)))
      x = self.dropout4Trans(x)

      x = self.sigmoid(self.convoutput(x))

      return x

#Bilateral Filteration
class BilateralFilterTransform:
    def __init__(self, d=9, sigmaColor=75, sigmaSpace=75):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def __call__(self, img):
        img_np = img.permute(1, 2, 0).numpy()

        filtered_img = cv2.bilateralFilter(img_np, self.d, self.sigmaColor, self.sigmaSpace)

        filtered_img_tensor = torch.tensor(filtered_img).permute(2, 0, 1)
        return filtered_img_tensor

class CustomImageDataset(Dataset):
    def __init__(self, img_folder, label_folder, noisy_transform=None,clean_transform =None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.noisetransform = noisy_transform
        self.cleantransform = clean_transform
        self.image_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.label_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels must match."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = Image.open(self.label_paths[idx])

        if self.noisetransform:
            image = self.noisetransform(image)
        if self.cleantransform:
            label = self.cleantransform(label)

        return image, label

img_folder = r"Noisy Images Directory"
label_folder = r"Clean Images Directory"

transform_noisy = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.size(0) == 1 else x),
    BilateralFilterTransform(d=9, sigmaColor=100, sigmaSpace=100)
])

transform_clean = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.size(0) == 1 else x)
])

test_dataset = CustomImageDataset(img_folder, label_folder, noisy_transform=transform_noisy,clean_transform = transform_clean)
print("Data Loaded Successfully!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#PSNR function
def calculate_psnr(original, compressed):
    original = original.clamp(0, 1)
    compressed = compressed.clamp(0, 1)

    mse = torch.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 1.0
    psnr = 10 * torch.log10(max_pixel**2 / mse)
    return psnr

#SSIM function
def calculate_ssim(img1, img2,win_size = 3):
    img1 = img1.cpu().detach().numpy().squeeze()
    img2 = img2.cpu().detach().numpy().squeeze()

    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    ssim_values = []
    for c in range(img1.shape[0]):
        channel_ssim = ssim(img1[c], img2[c], data_range=img2[c].max() - img2[c].min(), win_size=win_size)
        ssim_values.append(channel_ssim)
    return np.mean(ssim_values)

def test_model(model,criterion):
    model.eval()
    psnr_list = []
    ssim_list = []
    recon_img_list = []
    avg_psnr = 0
    avg_ssim = 0
    val_loss = 0
    with torch.no_grad():
        for img,label in test_dataset:
            img,label = img.to(device),label.to(device)
            recon_img = model(img.unsqueeze(dim = 0))
            loss = criterion(recon_img.squeeze(),label)

            psnr = calculate_psnr(label.unsqueeze(dim = 0),recon_img)
            SSIM = calculate_ssim(recon_img,label.unsqueeze(dim = 0))
            psnr_list.append(psnr)
            ssim_list.append(SSIM)
            recon_img_list.append(recon_img)
            avg_psnr += psnr
            avg_ssim += SSIM
            val_loss += loss.item()

    avg_psnr /= len(test_dataset)
    avg_ssim /= len(test_dataset)
    return avg_psnr,avg_ssim,val_loss,psnr_list,ssim_list,recon_img_list

model_path = r"Model Weights Directory"
DNet = DenoisingNet(dropout_rate=0.4)
print("Model Loaded Successfully!")
DNet.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
criterion = nn.L1Loss()

avg_psnr,avg_ssim,loss,psnr_list,ssim_list,recon_img_list = test_model(DNet,criterion)
print(f"Average PSNR: {avg_psnr:.4f} || Average SSIM: {avg_ssim:.4f}")

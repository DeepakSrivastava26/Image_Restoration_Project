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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
    

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)  # Apply channel attention
        x = x * self.spatial_attention(x)  # Apply spatial attention
        return x
    

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

        # Generate query, key, and value matrices
        query = self.query_conv(x).view(batch_size, C, -1)  # (B, C, W*H)
        key = self.key_conv(x).view(batch_size, C, -1)      # (B, C, W*H)
        value = self.value_conv(x).view(batch_size, C, -1)  # (B, C, W*H)

        # Compute attention scores
        scores = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)  # (B, W*H, W*H)

        # Compute the attention-weighted values
        out = torch.bmm(value, scores.transpose(1, 2))  # (B, C, W*H)
        out = out.view(batch_size, C, width, height)    # Reshape to (B, C, W, H)

        # Apply residual connection and scaling
        return self.gamma * out + x
    

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ConvolutionalAutoEncoder, self).__init__()
        #Conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.CBAM1 = CBAM(16,16,7)

        #Conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.CBAM2 = CBAM(32,32,7)

        #Conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.CBAM3 = CBAM(32,32,7)

        #Conv4
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,dilation = 2, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.CBAM4 = CBAM(64,64,7)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.selfattn = SelfAttention(64)

        #Deconv1
        self.conv1Trans = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3,dilation = 2, stride=1, padding=0)
        self.bn1Trans = nn.BatchNorm2d(32)
        self.dropout1Trans = nn.Dropout(dropout_rate)

        #Deconv2
        self.conv2Trans = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn2Trans = nn.BatchNorm2d(32)
        self.dropout2Trans = nn.Dropout(dropout_rate)

        #Deconv3
        self.conv3Trans = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.bn3Trans = nn.BatchNorm2d(16)
        self.dropout3Trans = nn.Dropout(dropout_rate)

        #Deconv4
        self.conv4Trans = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=0)

        #Deconv5
        self.convoutput = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.maxUnpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    # Encoder
      x = self.ReLU(self.bn1(self.conv1(x)))
      x = self.dropout1(x)
      x = self.CBAM1(x)
      conv_1 = x.clone()
      x, idx_1 = self.maxPool(x)

      x = self.ReLU(self.bn2(self.conv2(x)))
      x = self.dropout2(x)
      x = self.CBAM2(x)
      conv_2 = x.clone()
      x, idx_2 = self.maxPool(x)

      x = self.ReLU(self.bn3(self.conv3(x)))
      x = self.dropout3(x)
      x = self.CBAM3(x)
      conv_3 = x.clone()
      x, idx_3 = self.maxPool(x)

      x = self.ReLU(self.bn4(self.conv4(x)))
      x = self.dropout4(x)
      x = self.CBAM4(x)
      conv_4 = x.clone()
      x, idx_4 = self.maxPool(x)

      x = self.selfattn(x)

    # Decoder
      x = self.maxUnpool(x, idx_4,output_size = conv_4.size())
      concat = torch.concat((x,conv_4.clone()),dim = 1)
      x = self.ReLU(self.bn1Trans(self.conv1Trans(concat)))
      x = self.dropout1Trans(x)

      x = self.maxUnpool(x, idx_3,output_size = conv_3.size())
      concat = torch.concat((x,conv_3.clone()),dim = 1)
      x = self.ReLU(self.bn2Trans(self.conv2Trans(concat)))
      x = self.dropout2Trans(x)

      x = self.maxUnpool(x, idx_2,output_size = conv_2.size())
      concat = torch.concat((x,conv_2.clone()),dim = 1)
      x = self.ReLU(self.bn3Trans(self.conv3Trans(concat)))
      x = self.dropout3Trans(x)

      x = self.maxUnpool(x, idx_1,output_size = conv_1.size())
      concat = torch.concat((x,conv_1.clone()),dim = 1)
      x = self.ReLU(self.conv4Trans(concat))
      x = self.sigmoid(self.convoutput(x))

      return x
    

class CustomImageDataset(Dataset):
    def __init__(self, img_folder, label_folder, transform=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.label_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Ensure the lengths match
        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels must match."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = Image.open(self.label_paths[idx])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
    
# Example usage
img_folder = r"/data2/Gadha/Denoising_DIV2K_1/test/input"  # Replace with your image folder path
label_folder = r"/data2/Gadha/Denoising_DIV2K_1/test/target"  # Replace with your label folder path



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.size(0) == 1 else x)  # Expand to 3 channels if grayscale
])

# Create the dataset
test_dataset = CustomImageDataset(img_folder, label_folder, transform=transform)

def calculate_psnr(original, compressed):
    # Ensure the images are in the range [0, 1]
    original = original.clamp(0, 1)  # Clamping to handle potential out-of-bounds values
    compressed = compressed.clamp(0, 1)

    mse = torch.mean((original - compressed) ** 2)
    if mse == 0:  # If there's no difference, PSNR is infinite
        return float('inf')

    max_pixel = 1.0  # For normalized images
    psnr = 10 * torch.log10(max_pixel**2 / mse)
    return psnr

def calculate_ssim(img1, img2,win_size = 3):
    # Ensure the images are in the correct format
    img1 = img1.cpu().detach().numpy().squeeze()  # Convert to NumPy and remove batch/dimensions if necessary
    img2 = img2.cpu().detach().numpy().squeeze()

    # Normalize the images to the range [0, 1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    ssim_values = []
    for c in range(img1.shape[0]):  # Iterate over each channel
        channel_ssim = ssim(img1[c], img2[c], data_range=img2[c].max() - img2[c].min(), win_size=win_size)
        ssim_values.append(channel_ssim)

    # Return the average SSIM across channels
    return np.mean(ssim_values)

def test_model(model,criterion):
    model.eval()
    psnr_list = []
    ssim_list = []
    reconstructed_imgs = []
    avg_psnr = 0
    avg_ssim = 0
    val_loss = 0
    with torch.no_grad():
        for img,label in test_dataset:
            #img,label = img.to(device),label.to(device)
            recon_img = model(img.unsqueeze(dim = 0))
            loss = criterion(recon_img.squeeze(),label)
            reconstructed_imgs.append(recon_img)
            psnr = calculate_psnr(label.unsqueeze(dim = 0),recon_img)
            SSIM = calculate_ssim(recon_img,label.unsqueeze(dim = 0))
            psnr_list.append(psnr)
            ssim_list.append(SSIM)
            avg_psnr += psnr
            avg_ssim += SSIM
            val_loss += loss.item()

    avg_psnr /= len(test_dataset)
    avg_ssim /= len(test_dataset)
    return avg_psnr,avg_ssim,val_loss,psnr_list,ssim_list,reconstructed_imgs

model = ConvolutionalAutoEncoder(dropout_rate = 0.4)
model.load_state_dict(torch.load(r'/data2/Gadha/ConvAE_weights.pth',map_location=torch.device('cpu')))
print(model)

criterion = nn.L1Loss()

psnr, ssim_ , loss , psnr_list, ssim_list, reconstructed_images = test_model(model, criterion)
print(f'avg_psnr = {psnr} , avg_ssim = {ssim_} , avg_loss = {loss}')
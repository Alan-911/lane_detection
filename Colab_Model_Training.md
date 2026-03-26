# TuSimple Deep Learning Specification - Google Colab Guide

To overcome localized CPU/RAM bottlenecks and train the Deep Learning model effectively on the cloud using free GPUs, follow this exact environment layout for Google Colab.

This model employs a **U-Net Semantic Segmentation Architecture** to directly output lane pixels with high fidelity.

## 1. Setup Kaggle & Download Dataset

Open a new [Google Colab Notebook](https://colab.research.google.com/), navigate to **Runtime > Change runtime type**, and actively select the **T4 GPU**.

Create a new cell and execute the following API scripts to pull the heavy dataset directly without using your local CPU bandwidth:

```python
!pip install kaggle
import os

# Important: Upload your kaggle.json token directly to your Colab workspace first
os.environ['KAGGLE_CONFIG_DIR'] = '/content'

# Securely download the 15GB TuSimple dataset
!kaggle datasets download -d manideep1108/tusimple
!unzip -q tusimple.zip -d /content/tusimple
```

## 2. Advanced PyTorch Model Architecture 

Create a second cell and compile this U-Net model structure capable of classifying individual pixels as lane/non-lane boundaries.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class LaneUNet(nn.Module):
    """Deep learning segmentation model to definitively classify lane borders."""
    def __init__(self):
        super(LaneUNet, self).__init__()
        self.enc1 = self.conv_block(3, 16)
        self.enc2 = self.conv_block(16, 32)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec2 = self.conv_block(32, 16)
        self.dec1 = nn.Conv2d(16, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d2 = self.dec2(self.up(e2))
        return torch.sigmoid(self.dec1(d2 + e1))
```

## 3. Real-Time Warnings (Red Line Detection)

Here is where the model runs its output inference masks and incorporates the immediate safety logic you explicitly requested!
When the mathematical vehicle offset detects drifting beyond `0.5`, the entire predicted lane immediately shifts from safe-blue to critical-red.

```python
def predict_and_warn(frame_tensor, offset_meters, model_reference, device):
    """Executes segmentation inference and paints warnings dynamically."""
    model_reference.eval()
    with torch.no_grad():
        pred_mask = model_reference(frame_tensor.unsqueeze(0).to(device))[0][0]
        
    # Isolate boundary thresholds
    mask_np = (pred_mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
    color_overlay = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Critical Safety Trigger: Turn lane RED if predicting severe drift over 0.5m
    if abs(offset_meters) > 0.5:
        color_overlay[mask_np == 255] = [255, 0, 0]  # RED mask warning
    else:
        color_overlay[mask_np == 255] = [0, 255, 0]  # GREEN mask safe
        
    return color_overlay
```

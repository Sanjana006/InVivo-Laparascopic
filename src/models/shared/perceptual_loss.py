import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss extractor based on VGG16.
    Used for Cyclic Perceptual-Consistency Loss in Cycle-Dehaze.
    
    The paper extracts features from the 2nd and 5th pooling layers of VGG16.
    In torchvision's VGG16 `features` sequential model:
      - MaxPool2d (2nd pooling) is at index 9
      - MaxPool2d (5th pooling) is at index 30
    """
    def __init__(self, device: str):
        super().__init__()
        # Load pre-trained VGG16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # We only need up to the 5th pooling layer (index 30)
        # We split the network into two parts to get both features
        self.block1 = nn.Sequential(*[vgg[i] for i in range(10)]).to(device)  # up to index 9
        self.block2 = nn.Sequential(*[vgg[i] for i in range(10, 31)]).to(device) # index 10 to 30
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.eval()
        
        # ImageNet normalization parameters (required for VGG)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
        
        self.l2_loss = nn.MSELoss()

    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        Calculates L2 loss between the VGG16 features of input_img and target_img.
        Images should be RGB tensors in [0, 1].
        """
        # Normalize images for VGG
        input_norm = (input_img - self.mean) / self.std
        target_norm = (target_img - self.mean) / self.std
        
        # Extract features for input
        in_feat2 = self.block1(input_norm)
        in_feat5 = self.block2(in_feat2)
        
        # Extract features for target
        tgt_feat2 = self.block1(target_norm)
        tgt_feat5 = self.block2(tgt_feat2)
        
        # Calculate L2 (MSE) loss for both layers
        loss2 = self.l2_loss(in_feat2, tgt_feat2)
        loss5 = self.l2_loss(in_feat5, tgt_feat5)
        
        return loss2 + loss5

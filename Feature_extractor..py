import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import inspect
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleGAN3Encoder(nn.Module):
    def __init__(self, stylegan_pkl_path="/content/stylegan3-r-ffhq-1024x1024.pkl"):
        super(StyleGAN3Encoder, self).__init__()
        # Load StyleGAN3 model
        with open(stylegan_pkl_path, 'rb') as f:
            G = pickle.load(f)['G_ema'].to(device).eval()
        self.G = G
        self.mapping = G.mapping
        self.synthesis = G.synthesis
        self.w_dim = G.w_dim  # 512
        self.num_ws = G.num_ws
        print(f"Model w_dim: {self.w_dim}, num_ws: {self.num_ws}")
        print("Synthesis blocks:")
        for name, block in self.synthesis.named_children():
            print(f" - {name}: {type(block)}")
      
        self.input_block = getattr(self.synthesis, 'input', None)
        self.input_accepts_w = False
        if self.input_block and callable(self.input_block):
            sig = inspect.signature(self.input_block.forward)
            self.input_accepts_w = 'w' in sig.parameters
            print(f"'input' block accepts 'w': {self.input_accepts_w}")
       
        self.ws_adjust = nn.Identity().to(device)
        self.expected_w_dim = self.w_dim

    def forward(self, x):
        batch_size = x.shape[0]
        # Generate random z and map to ws
        z = torch.randn(batch_size, self.w_dim, device=device)
        ws = self.mapping(z, None)  # [batch, num_ws, 512]
        print(f"ws shape: {ws.shape}")
        # Initialize synthesis
        synthesis_idx = 0
        if self.input_block and self.input_accepts_w:
            ws_slice = ws[:, synthesis_idx:synthesis_idx+1].squeeze(1)  # [batch, 512]
            synthesis_input = self.input_block(ws_slice)
            synthesis_idx += 1
        else:
            synthesis_input = torch.randn(batch_size, 512, 4, 4, device=device)
        # Process synthesis blocks
        for name, block in self.synthesis.named_children():
            if name == 'input':
                continue
            if callable(block):
                sig = inspect.signature(block.forward)
                if 'w' in sig.parameters:
                    ws_slice = ws[:, synthesis_idx:synthesis_idx+1].squeeze(1)  # [batch, 512]
                    print(f"ws_slice shape for {name}: {ws_slice.shape}")
                    try:
                        synthesis_input = block(synthesis_input, ws_slice)
                    except RuntimeError as e:
                        print(f"Error in block {name}: {e}. Skipping to next block.")
                        synthesis_idx += 1
                        continue
                    synthesis_idx += 1
                    print(f"Block {name}, output shape: {synthesis_input.shape}")
                    if synthesis_input.shape[2] >= 256:
                        features = synthesis_input
                        break
                else:
                    synthesis_input = block(synthesis_input)
        if 'features' not in locals():
            features = synthesis_input
        # Resize to 256x256 if needed
        if features.shape[2] != 256:
            features = F.interpolate(features, size=(256, 256), mode='bilinear', align_corners=True)
        return features.half()

class ResNet101Extractor(nn.Module):
    def __init__(self):
        super(ResNet101Extractor, self).__init__()
        resnet = models.resnet101(weights='IMAGENET1K_V1')
        for block in resnet.layer3:
            block.conv1.stride = (1, 1)
            block.conv2.stride = (1, 1)
            block.conv2.dilation = (2, 2)
            block.conv2.padding = (2, 2)
            block.conv3.stride = (1, 1)
            if block.downsample:
                block.downsample[0].stride = (1, 1)
        for block in resnet.layer4:
            block.conv1.stride = (1, 1)
            block.conv2.stride = (1, 1)
            block.conv2.dilation = (4, 4)
            block.conv2.padding = (4, 4)
            block.conv3.stride = (1, 1)
            if block.downsample:
                block.downsample[0].stride = (1, 1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        print(f"Input to ResNet: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        conv4_x = self.layer3(x)
        conv5_x = self.layer4(conv4_x)
        return conv4_x, conv5_x

class FeatureFusionModule(nn.Module):
    def __init__(self, gan_channels=512, resnet_channels=2048, out_channels=512):
        super(FeatureFusionModule, self).__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(gan_channels + resnet_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, gan_features, resnet_features):
        print(f"gan_features shape: {gan_features.shape}")
        print(f"resnet_features shape before interpolation: {resnet_features.shape}")
        resnet_features = F.interpolate(resnet_features, size=gan_features.shape[2:], mode='bilinear', align_corners=True)
        print(f"resnet_features shape after interpolation: {resnet_features.shape}")
        fused = torch.cat((gan_features, resnet_features), dim=1)
        print(f"fused shape: {fused.shape}")
        fused = self.compress(fused)
        ca = self.channel_attention(fused)
        sa = self.spatial_attention(fused)
        fused = fused * ca * sa
        return fused

class ManipFeatureExtractor(nn.Module):
    def __init__(self, stylegan_pkl_path="/content/stylegan3-r-ffhq-1024x1024.pkl"):
        super(ManipFeatureExtractor, self).__init__()
        self.gan_encoder = StyleGAN3Encoder(stylegan_pkl_path).to(device)
        # Determine gan_channels dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256).to(device)
            gan_features = self.gan_encoder(dummy_input)
            gan_channels = gan_features.shape[1]
            print(f"Determined gan_channels: {gan_channels}")
        self.resnet_extractor = ResNet101Extractor().to(device).half()
        self.fusion = FeatureFusionModule(gan_channels=gan_channels, resnet_channels=2048, out_channels=512).to(device).half()

    def forward(self, x):
        print(f"Input to ManipFeatureExtractor: {x.shape}, dtype: {x.dtype}")
        gan_features = self.gan_encoder(x)
        print(f"gan_features after StyleGAN3Encoder: {gan_features.shape}, dtype: {gan_features.dtype}")
        x_half = x.half()
        _, conv5_x = self.resnet_extractor(x_half)
        print(f"conv5_x from ResNet101Extractor: {conv5_x.shape}, dtype: {conv5_x.dtype}")
        fused_features = self.fusion(gan_features, conv5_x)
        print(f"fused_features after FeatureFusionModule: {fused_features.shape}, dtype: {fused_features.dtype}")
        return fused_features

"""
ResNet-UNet implementation for semantic segmentation

ResNetをエンコーダーとして使用し、U-Net形式のデコーダーを持つモデル。
pretrainedの使用/不使用を選択可能。

Architecture:
    - Encoder: ResNet34/ResNet50 (torchvision)
    - Decoder: U-Net style upsampling with skip connections
    - Output: Single channel segmentation mask

Usage:
    # Pretrainedを使用
    model = ResNetUNet(backbone='resnet34', pretrained=True, n_classes=1)
    
    # Pretrainedを使用しない（ランダム初期化）
    model = ResNetUNet(backbone='resnet34', pretrained=False, n_classes=1)
    
    # ResNet50を使用
    model = ResNetUNet(backbone='resnet50', pretrained=True, n_classes=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """
    畳み込みブロック: Conv -> BN -> ReLU
    デコーダーで使用
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    デコーダーブロック: Upsample -> Concat -> ConvBlock
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, 
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock(
            in_channels // 2 + skip_channels, 
            out_channels
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # スキップ接続のサイズを調整（必要に応じて）
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ResNetUNet(nn.Module):
    """
    ResNet-UNet for semantic segmentation
    
    Args:
        backbone (str): ResNetのバージョン ('resnet34' or 'resnet50')
        pretrained (bool): ImageNet pretrainedを使用するか
        n_classes (int): 出力クラス数（デフォルト1: バイナリセグメンテーション）
        freeze_backbone (bool): バックボーンを固定するか（転移学習用）
    """
    def __init__(self, backbone='resnet34', pretrained=True, n_classes=1, freeze_backbone=False):
        super().__init__()
        
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.n_classes = n_classes
        
        # ResNetバックボーンを読み込み
        if backbone == 'resnet34':
            if pretrained:
                # torchvision 0.13以降の新しいweights API対応
                try:
                    resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                except:
                    # 古いバージョン対応
                    resnet = models.resnet34(pretrained=True)
            else:
                resnet = models.resnet34(pretrained=False)
            encoder_channels = [64, 64, 128, 256, 512]
            
        elif backbone == 'resnet50':
            if pretrained:
                try:
                    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                except:
                    resnet = models.resnet50(pretrained=True)
            else:
                resnet = models.resnet50(pretrained=False)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet34' or 'resnet50'")
        
        # エンコーダー（ResNetの各ステージ）
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64 or 256 channels
        self.encoder3 = resnet.layer2  # 128 or 512 channels
        self.encoder4 = resnet.layer3  # 256 or 1024 channels
        self.encoder5 = resnet.layer4  # 512 or 2048 channels
        
        # バックボーンを固定する場合
        if freeze_backbone:
            for param in [self.encoder1, self.pool1, self.encoder2, 
                         self.encoder3, self.encoder4, self.encoder5]:
                for p in param.parameters():
                    p.requires_grad = False
        
        # ボトルネック
        self.bottleneck = ConvBlock(encoder_channels[4], encoder_channels[4])
        
        # デコーダー（U-Net形式のアップサンプリング）
        self.decoder4 = DecoderBlock(
            encoder_channels[4], encoder_channels[3], encoder_channels[3]
        )
        self.decoder3 = DecoderBlock(
            encoder_channels[3], encoder_channels[2], encoder_channels[2]
        )
        self.decoder2 = DecoderBlock(
            encoder_channels[2], encoder_channels[1], encoder_channels[1]
        )
        self.decoder1 = DecoderBlock(
            encoder_channels[1], encoder_channels[0], encoder_channels[0]
        )
        
        # 最終アップサンプリング（元の解像度に戻す）
        self.final_upsample = nn.ConvTranspose2d(
            encoder_channels[0], encoder_channels[0] // 2,
            kernel_size=2, stride=2
        )
        
        # 出力層
        self.out_conv = nn.Conv2d(encoder_channels[0] // 2, n_classes, 1)
        
        # モデル情報を表示
        print(f"ResNetUNet initialized:")
        print(f"  - Backbone: {backbone}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Output classes: {n_classes}")
        print(f"  - Freeze backbone: {freeze_backbone}")
    
    def forward(self, x):
        # エンコーダー（特徴抽出）
        # Stage 1
        e1 = self.encoder1(x)       # 64 channels, 1/2 size
        
        # Stage 2
        e2 = self.pool1(e1)         
        e2 = self.encoder2(e2)      # 64/256 channels, 1/4 size
        
        # Stage 3
        e3 = self.encoder3(e2)      # 128/512 channels, 1/8 size
        
        # Stage 4
        e4 = self.encoder4(e3)      # 256/1024 channels, 1/16 size
        
        # Stage 5 (bottleneck)
        e5 = self.encoder5(e4)      # 512/2048 channels, 1/32 size
        b = self.bottleneck(e5)
        
        # デコーダー（U-Net形式のアップサンプリング + スキップ接続）
        d4 = self.decoder4(b, e4)   # 1/16 size
        d3 = self.decoder3(d4, e3)  # 1/8 size
        d2 = self.decoder2(d3, e2)  # 1/4 size
        d1 = self.decoder1(d2, e1)  # 1/2 size
        
        # 最終アップサンプリング
        out = self.final_upsample(d1)  # 元のサイズ
        out = self.out_conv(out)
        
        return out
    
    def get_model_info(self):
        """モデル情報を返す"""
        return {
            'backbone': self.backbone_name,
            'pretrained': self.pretrained,
            'n_classes': self.n_classes,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# テスト用コード
if __name__ == "__main__":
    # モデルのテスト
    print("="*60)
    print("ResNet-UNet Model Test")
    print("="*60)
    
    # Pretrained使用
    print("\n1. ResNet34 with pretrained weights:")
    model_pretrained = ResNetUNet(backbone='resnet34', pretrained=True, n_classes=1)
    info = model_pretrained.get_model_info()
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    # Pretrained不使用
    print("\n2. ResNet34 without pretrained weights:")
    model_scratch = ResNetUNet(backbone='resnet34', pretrained=False, n_classes=1)
    info = model_scratch.get_model_info()
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    # フォワードパステスト
    print("\n3. Forward pass test:")
    x = torch.randn(2, 3, 384, 512)  # (B, C, H, W)
    with torch.no_grad():
        out = model_pretrained(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # ResNet50のテスト
    print("\n4. ResNet50 with pretrained weights:")
    model_resnet50 = ResNetUNet(backbone='resnet50', pretrained=True, n_classes=1)
    info = model_resnet50.get_model_info()
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

import torch
import torch.nn as nn
from easydict import EasyDict

 

########################################################################
# 1. DENSE 2D BACKBONE
########################################################################
class RadarDenseBEVBackbone(nn.Module):
    def __init__(self, cfg):
        super(RadarDenseBEVBackbone, self).__init__()
        self.cfg = cfg

        # Hybrid Normalization: Handle the massive numerical range of radar
        self.input_norm = nn.BatchNorm2d(10)

        # Block 1: Keep spatial resolution (256x107)
        self.block1 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Block 2: Downsample by 2x (128x53)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Block 3: Downsample by 2x (64x27)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # TO_BEV FPN Aggregation
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(size=(256, 107), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(size=(256, 107), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

 

        self.dropout = nn.Dropout2d(p=cfg.MODEL.BACKBONE.DROPOUT_P)

 

    def forward(self, dict_item):
        x = dict_item['bev_input']
        x = self.input_norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        out1 = self.up1(x1)
        out2 = self.up2(x2)
        out3 = self.up3(x3)

        bev_feat = torch.cat([out1, out2, out3], dim=1)
        bev_feat = self.dropout(bev_feat)

        dict_item['bev_feat'] = bev_feat
        return dict_item

 

########################################################################
# 2. POLAR CENTERPOINT HEAD (Anchor-Free)
########################################################################
class PolarCenterHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=7):
        super(PolarCenterHead, self).__init__()

        self.num_classes = num_classes

        # Shared feature reduction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 1. Heatmap Head: Predicts object centers
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        # 2. Regression Head: Predicts w_idx, l_idx, and sub-pixel offsets (dx, dy)
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=1) 
        )

        # Initialize biases strategically for the heatmap to prevent early instability
        self.heatmap_head[-1].bias.data.fill_(-2.19) # Initialize prob to ~0.1

 

    def forward(self, dict_item):
        x = dict_item['bev_feat']
        x = self.shared_conv(x)

        # (Batch, Classes, 256, 107)
        pred_hm = self.heatmap_head(x)

        # (Batch, 4, 256, 107)
        pred_reg = self.reg_head(x)

        dict_item['pred_hm'] = pred_hm
        dict_item['pred_reg'] = pred_reg

        return dict_item

 

 

########################################################################
# 3. EXECUTION & TESTING WITH ACTUAL DATA
########################################################################
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # IMPORT YOUR DATASET SCRIPT
    # Make sure your previous script is saved as 'dataset.py' in the same folder
    from dataset_v1_heatmap import KRadarBEVDataset, radar_detection_collate_fn

    # Configuration
    cfg = {
        'MODEL': {
            'BACKBONE': {'DROPOUT_P': 0.3},
        }
    }
    cfg = EasyDict(cfg)

    # 1. Load the Actual Dataset
    COMPRESSED_DATA_DIR = "/media/terra2024/NVME-Storage/K-Radar-GMM-sinc"
    GT_DATA_DIR = "/media/terra2024/NVME-Storage/K-Radar-GT-Polar/"

    dataset = KRadarBEVDataset(
        compressed_dir=COMPRESSED_DATA_DIR, 
        gt_dir=GT_DATA_DIR, 
        sequences=[1] # Adjust sequence list as needed
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=radar_detection_collate_fn
    )

 

    # 2. Instantiate Network Components
    # Important: Set num_classes to 7 to match your CLASS_MAP in dataset.py
    backbone = RadarDenseBEVBackbone(cfg=cfg).cuda()
    center_head = PolarCenterHead(in_channels=768, num_classes=7).cuda()

    # 3. Process the first real batch
    for batch_idx, (bev_batch, boxes_batch, target_hm, target_reg, reg_mask) in enumerate(dataloader):
        print("\n=== STARTING END-TO-END TEST ON ACTUAL DATA ===")

        # Move inputs to GPU
        bev_batch = bev_batch.cuda()
        target_hm = target_hm.cuda()
        target_reg = target_reg.cuda()
        reg_mask = reg_mask.cuda()

        # Forward Pass
        data_dict = {'bev_input': bev_batch}
        data_dict = backbone(data_dict)
        data_dict = center_head(data_dict)

        # Print Backbone Outputs
        print("--- Backbone Output ---")
        print(f"BEV Feature Shape:     {data_dict['bev_feat'].shape}") 

        # Print Head Outputs
        print("\n--- CenterPoint Head Output ---")
        print(f"Predicted Heatmap:     {data_dict['pred_hm'].shape}") 
        print(f"Predicted Regression:  {data_dict['pred_reg'].shape}")  

        # Print Generated Targets (from Dataloader)
        print("\n--- Ground Truth Targets ---")
        print(f"Target Heatmap:        {target_hm.shape}")  
        print(f"Target Regression:     {target_reg.shape}") 
        print(f"Regression Mask:       {reg_mask.shape}")   

        # Let's verify that the network predictions match the target dimensions perfectly
        assert data_dict['pred_hm'].shape == target_hm.shape, "Heatmap shape mismatch!"
        assert data_dict['pred_reg'].shape == target_reg.shape, "Regression shape mismatch!"
        print("\nSUCCESS: Network output shapes perfectly match Ground Truth target shapes!")

        break # Only test the first batch
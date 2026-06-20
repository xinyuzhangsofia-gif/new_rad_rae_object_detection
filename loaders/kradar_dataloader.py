from torch.utils.data import DataLoader
from dataset import KRadarDataset
dataset = KRadarDataset("/home/local/xinyu/KRadar/1/radar_tesseract")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch in dataloader:
    print(batch['rad'].shape)
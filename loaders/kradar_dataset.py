from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import glob
import os

class KRadarDataset(Dataset):
    def __init__(self, radar_folder):
        self.files = sorted(glob.glob(os.path.join(radar_folder, "*.mat")))

    def __len__(self):
        return len(self.files)
    
    def _drea2rae(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=0)  

    def _drea2rad(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=2)  
    
    def _drea2aed(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=1)  

    def __getitem__(self, idx):
        drea = loadmat(self.files[idx])['arrDREA']  
        drea = np.asarray(drea)
        return {
            "rae": self._drea2rae(drea),
            "rad": self._drea2rad(drea),
            "aed": self._drea2aed(drea)
        }

# if __name__ == "__main__":
#     dataset = KRadarDataset("/home/local/xinyu/KRadar/1/radar_tesseract")
#     data = dataset[0]
#     print(dataset[0]['rae'].shape)



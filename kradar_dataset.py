from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import glob
import os

class KRadarDataset(Dataset):
    def __init__(self, radar_folder):
        self.files = sorted(glob.glob(os.path.join(radar_folder, "*.mat")))

        self.idx_to_file ={}
        for f in self.files:
            fname = os.path.basename(f)
            tesseract_idx = fname.split('_')[1].split('.')[0]
            self.idx_to_file[tesseract_idx] = f

    def __len__(self):
        return len(self.files)
    
    def _drea2rea(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=0)  

    def _drea2rad(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=2)  
    
    def _drea2aed(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=1)
    
    def _rea2ra(self,rea:np.ndarray):
        return np.sum(rea,axis=1)
    
    def _rea2re(self,rea:np.array):
        return np.sum(rea, axis=2)

    
    def _load_one_file(self, file_path):
        drea = loadmat(file_path)['arrDREA']  
        drea = np.asarray(drea)
        rea = self._drea2rea(drea)

        return {
            "rea": rea,
            "rad": self._drea2rad(drea),
            "aed": self._drea2aed(drea),
            "ra_map":self._rea2ra(rea),
            "re_map":self._rea2re(rea)
        }
    
    def __getitem__(self, idx):
        return self._load_one_file(self.files[idx])
    
    def get_by_tesseract_idx(self,tesseract_idx):
        if tesseract_idx not in self.idx_to_file:
            raise KeyError(f"tesseract_idx {tesseract_idx} not found in dataset")
        file_path=self.idx_to_file[tesseract_idx]
        return self._load_one_file(file_path)
    


# if __name__ == "__main__":
#     dataset = KRadarDataset("/home/local/xinyu/KRadar/1/radar_tesseract")
#     data = dataset[0]
#     print(dataset[0]['rae'].shape)



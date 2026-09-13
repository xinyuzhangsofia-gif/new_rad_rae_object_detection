"""Load raw K-Radar ``arrDREA`` MAT tensors for visualization tools."""

import glob
import os

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset


class KRadarDataset(Dataset):
    """Return the three standard projections of each raw DREA tensor."""

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

    def _load_one_file(self, file_path):
        drea = np.asarray(loadmat(file_path)["arrDREA"])
        return {
            "rae": self._drea2rae(drea),
            "rad": self._drea2rad(drea),
            "aed": self._drea2aed(drea),
        }

    def __getitem__(self, index):
        return self._load_one_file(self.files[index])


class KRadarSensorDataset(KRadarDataset):
    """Add RA/RE maps and frame-index lookup for sensor visualization."""

    def __init__(self, radar_folder):
        super().__init__(radar_folder)
        self.idx_to_file = {}
        for file_path in self.files:
            filename = os.path.basename(file_path)
            tesseract_idx = filename.split("_")[1].split(".")[0]
            self.idx_to_file[tesseract_idx] = file_path

    def _load_one_file(self, file_path):
        projections = super()._load_one_file(file_path)
        rae = projections["rae"]
        return {
            "rea": rae,
            "rad": projections["rad"],
            "aed": projections["aed"],
            "ra_map": np.sum(rae, axis=1),
            "re_map": np.sum(rae, axis=2),
        }

    def get_by_tesseract_idx(self, tesseract_idx):
        if tesseract_idx not in self.idx_to_file:
            raise KeyError(
                f"tesseract_idx {tesseract_idx} not found in dataset"
            )
        return self._load_one_file(self.idx_to_file[tesseract_idx])


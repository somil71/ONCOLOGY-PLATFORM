import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

class MultimodalOncologyDataset(Dataset):
    """
    High-throughput Dataset for Multimodal Oncology (WSI, Genomics, Clinical).
    Designed to prevent GPU starvation by pre-loading embeddings into RAM or using MemMaps.
    Includes dynamic 'Structural Missingness' masking to train Evidential Networks.
    """
    def __init__(self, 
                 metadata_csv: str, 
                 wsi_emb_dir: str, 
                 genomic_csv: str, 
                 clinical_csv: str,
                 max_wsi_patches: int = 1000,
                 is_train: bool = True,
                 missing_rate: float = 0.2):
        """
        Args:
            metadata_csv: Path to CSV containing Patient IDs and Survival labels.
            wsi_emb_dir: Directory containing pre-computed WSI embeddings (.npy files).
            genomic_csv: Path to pre-processed RNA-Seq/Mutation data.
            clinical_csv: Path to pre-processed normalized clinical tabular data.
            max_wsi_patches: Pad/Truncate WSI bags to this size for batched training.
            is_train: If True, applies random missingness to modalities for robustness.
            missing_rate: Probability of dropping a modality (Genomics or Clinical) during training.
        """
        super().__init__()
        self.is_train = is_train
        self.missing_rate = missing_rate
        self.max_patches = max_wsi_patches
        self.wsi_emb_dir = wsi_emb_dir
        
        # 1. Load Metadata (Patient ID, Survival Time, Event Status)
        # Expected columns: ['patient_id', 'survival_months', 'event_status']
        self.metadata = pd.read_csv(metadata_csv)
        self.patient_ids = self.metadata['patient_id'].values
        
        # Convert survival data to fast numpy arrays
        self.survival_time = self.metadata['survival_months'].values.astype(np.float32)
        self.event_status = self.metadata['event_status'].values.astype(np.float32)

        # 2. Pre-load Genomics and Clinical data into RAM
        # Since these are tabular/vector data, they easily fit in memory for fast __getitem__
        print("Loading Genomic and Clinical data into memory...")
        df_genomic = pd.read_csv(genomic_csv).set_index('patient_id')
        df_clinical = pd.read_csv(clinical_csv).set_index('patient_id')
        
        # Realign to match metadata patient order exactly
        df_genomic = df_genomic.loc[self.patient_ids]
        df_clinical = df_clinical.loc[self.patient_ids]
        
        self.genomic_data = torch.from_numpy(df_genomic.values.astype(np.float32))
        self.clinical_data = torch.from_numpy(df_clinical.values.astype(np.float32))
        
        print(f"Dataset Initialized: {len(self.patient_ids)} patients.")
        print(f"Genomic dims: {self.genomic_data.shape[1]} | Clinical dims: {self.clinical_data.shape[1]}")

    def __len__(self):
        return len(self.patient_ids)

    def _load_wsi_bag(self, patient_id: str) -> torch.Tensor:
        """
        Loads the pre-computed WSI embeddings.
        To avoid slow disk I/O, ensure these are stored on a fast NVMe SSD.
        """
        emb_path = os.path.join(self.wsi_emb_dir, f"{patient_id}.npy")
        try:
            # Using mmap_mode='r' can save RAM if files are massive, but standard load is faster if RAM allows.
            wsi_emb = np.load(emb_path) 
        except FileNotFoundError:
            # Fallback for missing WSI: return zero tensor
            return torch.zeros((self.max_patches, 768), dtype=torch.float32), False

        # Pad or Truncate to max_patches
        num_patches = wsi_emb.shape[0]
        if num_patches >= self.max_patches:
            # Random sampling during training for augmentation, deterministic during val
            if self.is_train:
                indices = np.random.choice(num_patches, self.max_patches, replace=False)
                wsi_emb = wsi_emb[indices]
            else:
                wsi_emb = wsi_emb[:self.max_patches]
            valid_mask = torch.ones(self.max_patches, dtype=torch.bool)
        else:
            # Pad with zeros
            pad_len = self.max_patches - num_patches
            wsi_emb = np.pad(wsi_emb, ((0, pad_len), (0, 0)), mode='constant')
            valid_mask = torch.cat([
                torch.ones(num_patches, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
            
        return torch.from_numpy(wsi_emb).float(), valid_mask, True

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        
        # Load modalities
        wsi_feat, wsi_pad_mask, wsi_present = self._load_wsi_bag(pid)
        rna_feat = self.genomic_data[idx]
        cli_feat = self.clinical_data[idx]
        
        time = self.survival_time[idx]
        event = self.event_status[idx]

        # Missing Modality Mask: [WSI, Genomics, Clinical]
        # 1.0 means present, 0.0 means missing
        missing_mask = torch.ones(3, dtype=torch.float32)
        if not wsi_present:
            missing_mask[0] = 0.0
            
        # During training, simulate "Structural Missingness" to teach the Evidential Network
        if self.is_train:
            if random.random() < self.missing_rate:
                missing_mask[1] = 0.0 # Drop Genomics
                rna_feat = torch.zeros_like(rna_feat)
                
            if random.random() < (self.missing_rate / 2): # Clinical is rarely missing
                missing_mask[2] = 0.0 # Drop Clinical
                cli_feat = torch.zeros_like(cli_feat)

        return {
            'patient_id': pid,
            'wsi': wsi_feat,             # [max_patches, 768]
            'wsi_pad_mask': wsi_pad_mask,# [max_patches] boolean
            'rna': rna_feat,             # [num_genes]
            'cli': cli_feat,             # [num_clinical_vars]
            'missing_mask': missing_mask,# [3]
            'time': time,                # scalar
            'event': event               # scalar
        }

# ==========================================
# Fast DataLoader Initialization Factory
# ==========================================
def create_dataloaders(cfg, metadata_train, metadata_val, wsi_dir, genomic_csv, clinical_csv):
    """
    Constructs high-throughput dataloaders with prefetching and pinning.
    """
    train_ds = MultimodalOncologyDataset(
        metadata_csv=metadata_train,
        wsi_emb_dir=wsi_dir,
        genomic_csv=genomic_csv,
        clinical_csv=clinical_csv,
        max_wsi_patches=cfg.get('max_wsi_patches', 1000),
        is_train=True,
        missing_rate=cfg.get('missing_rate', 0.2)
    )
    
    val_ds = MultimodalOncologyDataset(
        metadata_csv=metadata_val,
        wsi_emb_dir=wsi_dir,
        genomic_csv=genomic_csv,
        clinical_csv=clinical_csv,
        max_wsi_patches=cfg.get('max_wsi_patches', 1000),
        is_train=False
    )
    
    # Mirroring your GenMol DataLoader optimization
    # Use max available workers and aggressive prefetch factor
    num_workers = min(8, os.cpu_count() or 1)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg['batch_size'], 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True
    )
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=cfg['batch_size'] * 2, # No grads, larger batch size
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    return train_dl, val_dl

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torchaudio
import torchaudio.transforms as T_audio
import torchvision.transforms as T_img

class MultimodalHealthDataset(Dataset):
    def __init__(self, root_dir, img_size=(64, 64), audio_sr=16000, audio_dur=2.0):
        self.root_dir = root_dir
        self.audio_sr = audio_sr
        self.n_samples = int(audio_sr * audio_dur)
        self.img_size = img_size
        
        # Load CSV
        csv_path = os.path.join(root_dir, 'tabular.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到表格文件: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"📄 数据集加载完成: {len(self.df)} 条样本, 列名: {self.df.columns.tolist()}")

        # --- 1. 智能列名匹配逻辑 ---
        all_cols = self.df.columns.tolist()
        target_mapping = {
            'weight': ['weight', 'weight(kg)', 'Weight', 'Weight(kg)'],
            'height': ['height', 'height(cm)', 'Height', 'Height(cm)'],
            'age': ['age', 'Age']
        }
        
        selected_cols = []
        for standard_name, possible_names in target_mapping.items():
            found_col = None
            for name in possible_names:
                if name in all_cols:
                    found_col = name
                    break
            if found_col:
                selected_cols.append(found_col)
            else:
                print(f"⚠️ 警告: 未找到 '{standard_name}' 对应的列 (尝试了: {possible_names})")
        
        if len(selected_cols) < 3:
            raise ValueError(f"错误: 表格中缺少必要的数值列。需要3列，但只找到了 {len(selected_cols)} 列: {selected_cols}")
        
        self.tab_cols = selected_cols
        print(f"✅ 已选定表格列用于训练: {self.tab_cols}")

        # --- 2. 预计算归一化参数 (强制 float32) ---
        tab_data_np = self.df[self.tab_cols].values.astype(np.float32)
        self.tab_mean = np.mean(tab_data_np, axis=0).astype(np.float32)
        self.tab_std = (np.std(tab_data_np, axis=0) + 1e-6).astype(np.float32)

        # --- 3. 定义变换 ---
        # 图像变换
        self.img_transform = T_img.Compose([
            T_img.Resize(img_size),
            T_img.ToTensor(), # 输出为 [0,1] 的 float32
            T_img.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 音频变换 (MelSpectrogram)
        self.mel_spec_transform = T_audio.MelSpectrogram(
            sample_rate=audio_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=64,
            f_min=0.0,
            f_max=audio_sr/2.0
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row['user_id'])
        label = int(row['label'])
        
        # --- 处理图像 ---
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            path = os.path.join(self.root_dir, 'images', f"{uid}{ext}")
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.img_transform(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                img_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
        else:
            # 占位符
            img_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])

        # --- 处理音频 ---
        audio_path = None
        for ext in ['.wav', '.mp3', '.flac']:
            path = os.path.join(self.root_dir, 'audios', f"{uid}{ext}")
            if os.path.exists(path):
                audio_path = path
                break
        
        if audio_path:
            try:
                waveform, sr = torchaudio.load(audio_path)
                # 重采样如果需要
                if sr != self.audio_sr:
                    resampler = T_audio.Resample(orig_freq=sr, new_freq=self.audio_sr)
                    waveform = resampler(waveform)
                
                # 填充或截断
                if waveform.shape[1] < self.n_samples:
                    pad_len = self.n_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                else:
                    waveform = waveform[:, :self.n_samples]
                
                # compute mel spectrogram; output shape [C, F, T] where C is channels
                mel_spec = self.mel_spec_transform(waveform)  # [C, F, T]
                # if multichannel, collapse to mono by averaging
                if mel_spec.dim() == 3 and mel_spec.size(0) > 1:
                    mel_spec = mel_spec.mean(dim=0, keepdim=False)  # [F, T]
                else:
                    mel_spec = mel_spec.squeeze(0)  # [F, T]

                # 对数压缩和归一化 (可选，视模型需求而定，这里做简单的标准化)
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
                audio_tensor = mel_spec.unsqueeze(0) # [1, F, T]
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")
                audio_tensor = torch.zeros(1, 64, int(self.n_samples / 512) + 1)
        else:
            # 占位符
            audio_tensor = torch.zeros(1, 64, int(self.n_samples / 512) + 1)

        # --- 处理表格数据 (关键修复：强制 float32) ---
        try:
            raw_vals = row[self.tab_cols].values.astype(np.float32)
            norm_vals = (raw_vals - self.tab_mean) / self.tab_std
            tab_tensor = torch.tensor(norm_vals, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing tabular data for {uid}: {e}")
            tab_tensor = torch.zeros(len(self.tab_cols), dtype=torch.float32)

        return {
            'img': img_tensor,      # [3, H, W], float32
            'audio': audio_tensor,  # [1, F, T], float32
            'tabular': tab_tensor,  # [3], float32
            'label': torch.tensor(label, dtype=torch.long)
        }
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchaudio
import torchaudio.transforms as T_audio
import torchvision.transforms as T_img
import yaml
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.modules.fusion_transformer import FusionTransformer
from src.heads.classifier_head import ClassifierHead

def load_config(config_path='config/model_config.yaml'):
    if not os.path.exists(config_path):
        print(f"❌ 配置文件 {config_path} 不存在！")
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class SingleSamplePreprocessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_size = (64, 64)
        self.audio_sr = 16000
        self.audio_dur = 2.0
        self.n_samples = int(self.audio_sr * self.audio_dur)
        
        # 1. 加载 CSV 并计算归一化参数 (与训练集完全一致)
        csv_path = os.path.join(root_dir, 'tabular.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到表格文件: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # 智能列名匹配 (复制自 Dataset 代码)
        all_cols = self.df.columns.tolist()
        target_mapping = {
            'weight': ['weight', 'weight(kg)', 'Weight', 'Weight(kg)'],
            'height': ['height', 'height(cm)', 'Height', 'Height(cm)'],
            'age': ['age', 'Age']
        }
        
        self.tab_cols = []
        for standard_name, possible_names in target_mapping.items():
            found_col = None
            for name in possible_names:
                if name in all_cols:
                    found_col = name
                    break
            if found_col:
                self.tab_cols.append(found_col)
        
        if len(self.tab_cols) < 3:
            raise ValueError(f"错误: 表格中缺少必要的数值列。找到: {self.tab_cols}")
        
        # 预计算 Mean 和 Std (强制 float32)
        tab_data_np = self.df[self.tab_cols].values.astype(np.float32)
        self.tab_mean = np.mean(tab_data_np, axis=0).astype(np.float32)
        self.tab_std = (np.std(tab_data_np, axis=0) + 1e-6).astype(np.float32)
        
        print(f"✅ 初始化完成: 表格列={self.tab_cols}, Mean={self.tab_mean}, Std={self.tab_std}")

        # 2. 定义变换
        self.img_transform = T_img.Compose([
            T_img.Resize(self.img_size),
            T_img.ToTensor(),
            T_img.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mel_spec_transform = T_audio.MelSpectrogram(
            sample_rate=self.audio_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=64,
            f_min=0.0,
            f_max=self.audio_sr/2.0
        )

    def process(self, user_id):
        """输入 user_id (字符串), 返回 tensors"""
        uid = str(user_id)
        
        # --- 处理图像 ---
        img_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
        img_found = False
        for ext in ['.jpg', '.png', '.jpeg']:
            path = os.path.join(self.root_dir, 'images', f"{uid}{ext}")
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.img_transform(img)
                    img_found = True
                    break
                except Exception as e:
                    print(f"⚠️ 图像读取失败 {path}: {e}")
        
        if not img_found:
            print(f"⚠️ 未找到图像文件 (user_id: {uid}), 使用零填充。")

        # --- 处理音频 ---
        audio_tensor = torch.zeros(1, 64, int(self.n_samples / 512) + 1)
        audio_found = False
        for ext in ['.wav', '.mp3', '.flac']:
            path = os.path.join(self.root_dir, 'audios', f"{uid}{ext}")
            if os.path.exists(path):
                try:
                    waveform, sr = torchaudio.load(path)
                    # 重采样
                    if sr != self.audio_sr:
                        resampler = T_audio.Resample(orig_freq=sr, new_freq=self.audio_sr)
                        waveform = resampler(waveform)
                    
                    # 填充或截断
                    if waveform.shape[1] < self.n_samples:
                        pad_len = self.n_samples - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                    else:
                        waveform = waveform[:, :self.n_samples]
                    
                    # Mel Spectrogram
                    mel_spec = self.mel_spec_transform(waveform)
                    if mel_spec.dim() == 3 and mel_spec.size(0) > 1:
                        mel_spec = mel_spec.mean(dim=0, keepdim=False)
                    else:
                        mel_spec = mel_spec.squeeze(0)
                    
                    # 标准化 (关键步骤)
                    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
                    audio_tensor = mel_spec.unsqueeze(0)
                    audio_found = True
                    break
                except Exception as e:
                    print(f"⚠️ 音频读取失败 {path}: {e}")
        
        if not audio_found:
            print(f"⚠️ 未找到音频文件 (user_id: {uid}), 使用零填充。")

        # --- 处理表格 ---
        tab_tensor = torch.zeros(len(self.tab_cols), dtype=torch.float32)
        row = self.df[self.df['user_id'].astype(str) == uid]
        
        if not row.empty:
            try:
                raw_vals = row[self.tab_cols].values[0].astype(np.float32)
                norm_vals = (raw_vals - self.tab_mean) / self.tab_std
                tab_tensor = torch.tensor(norm_vals, dtype=torch.float32)
            except Exception as e:
                print(f"⚠️ 表格数据处理失败: {e}")
        else:
            print(f"⚠️ 未在 CSV 中找到 user_id: {uid}, 使用零填充。")

        return img_tensor, audio_tensor, tab_tensor

    def process_with_tab_values(self, user_id: str, height_cm: float, weight_kg: float, age: float = 0.0):
        """
        与 `process()` 相同的图像/音频处理流程，但表格特征不再从 CSV 查行，
        而是直接使用表单传入的 height/weight/age，并使用训练集一致的 mean/std 归一化。
        """
        uid = str(user_id)

        # --- 处理图像 ---
        img_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
        img_found = False
        for ext in ['.jpg', '.png', '.jpeg']:
            path = os.path.join(self.root_dir, 'images', f"{uid}{ext}")
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.img_transform(img)
                    img_found = True
                    break
                except Exception as e:
                    print(f"⚠️ 图像读取失败 {path}: {e}")

        if not img_found:
            print(f"⚠️ 未找到图像文件 (user_id: {uid}), 使用零填充。")

        # --- 处理音频 ---
        audio_tensor = torch.zeros(1, 64, int(self.n_samples / 512) + 1)
        audio_found = False
        for ext in ['.wav', '.mp3', '.flac']:
            path = os.path.join(self.root_dir, 'audios', f"{uid}{ext}")
            if os.path.exists(path):
                try:
                    waveform, sr = torchaudio.load(path)
                    if sr != self.audio_sr:
                        resampler = T_audio.Resample(orig_freq=sr, new_freq=self.audio_sr)
                        waveform = resampler(waveform)

                    if waveform.shape[1] < self.n_samples:
                        pad_len = self.n_samples - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                    else:
                        waveform = waveform[:, :self.n_samples]

                    mel_spec = self.mel_spec_transform(waveform)
                    if mel_spec.dim() == 3 and mel_spec.size(0) > 1:
                        mel_spec = mel_spec.mean(dim=0, keepdim=False)
                    else:
                        mel_spec = mel_spec.squeeze(0)

                    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
                    audio_tensor = mel_spec.unsqueeze(0)
                    audio_found = True
                    break
                except Exception as e:
                    print(f"⚠️ 音频读取失败 {path}: {e}")

        if not audio_found:
            print(f"⚠️ 未找到音频文件 (user_id: {uid}), 使用零填充。")

        # --- 处理表格：直接用表单值 ---
        tab_tensor = torch.zeros(len(self.tab_cols), dtype=torch.float32)
        try:
            mapping = {
                'height': float(height_cm),
                'weight': float(weight_kg),
                'age': float(age),
            }
            raw_vals: List[float] = []
            for col in self.tab_cols:
                c = col.lower()
                if 'height' in c:
                    raw_vals.append(mapping['height'])
                elif 'weight' in c:
                    raw_vals.append(mapping['weight'])
                elif c == 'age' or 'age' in c:
                    raw_vals.append(mapping['age'])
                else:
                    raw_vals.append(0.0)

            raw_np = np.array(raw_vals, dtype=np.float32)
            norm_vals = (raw_np - self.tab_mean) / self.tab_std
            tab_tensor = torch.tensor(norm_vals, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️ 表格数据处理失败（使用表单值）: {e}")

        return img_tensor, audio_tensor, tab_tensor

def get_class_name(pred_idx, class_map=None):
    if class_map and isinstance(class_map, dict):
        return class_map.get(pred_idx, f"Class_{pred_idx}")
    return f"Class_{pred_idx}"


_SRGA_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_paths(base_dir: str) -> Tuple[str, str, str]:
    """
    base_dir: `main/SRGA` 目录（绝对路径）
    返回: (data_dir, config_path, checkpoint_path)
    """
    config_path = os.path.join(base_dir, "config", "model_config.yaml")
    data_dir = os.path.join(base_dir, "temp")

    checkpoint_candidates = [
        os.path.join(base_dir, "checkpoints", "best_model.pth"),
        os.path.join(base_dir, "checkpoints", "best_model_f1.pth"),
        os.path.join(base_dir, "best_model.pth"),
        os.path.join(base_dir, "best_model_f1.pth"),
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if os.path.exists(p)), checkpoint_candidates[0])
    return data_dir, config_path, checkpoint_path


def _load_srga_model_once(base_dir: str, device: torch.device):
    cache_key = f"{base_dir}::{device.type}"
    cached = _SRGA_MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    data_dir, config_path, checkpoint_path = _resolve_paths(base_dir)

    cfg: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"未找到模型权重文件：期望存在 `main/SRGA/checkpoints/best_model.pth`（或 best_model_f1.pth）。当前解析到: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not cfg:
        cfg = checkpoint.get("config", {}) or {}

    fusion_dim = int(cfg.get("fusion_dim", 256))
    num_heads = int(cfg.get("num_heads", 4))
    depth = int(cfg.get("depth", 4))

    model = FusionTransformer(fusion_dim=fusion_dim, num_heads=num_heads, depth=depth).to(device)
    classifier = ClassifierHead(input_dim=fusion_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.eval()
    classifier.eval()

    class_map = checkpoint.get("class_mapping", None)

    pack = {
        "model": model,
        "classifier": classifier,
        "class_map": class_map,
        "data_dir": data_dir,
        "checkpoint_path": checkpoint_path,
    }
    _SRGA_MODEL_CACHE[cache_key] = pack
    return pack


def run_srga_inference(
    *,
    base_dir: str,
    user_id: str,
    height_cm: float,
    weight_kg: float,
    age: float = 0.0,
) -> Dict[str, Any]:
    """
    Web 端推理入口：图像/音频从 `main/SRGA/temp` 读取，表格特征用表单 height/weight/age。
    返回预测标签、置信度、概率分布等。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = _load_srga_model_once(base_dir, device)

    # 关键运行信息（便于与 Web 端联动排查）
    print(f"📥 加载模型: {pack.get('checkpoint_path')}")
    print(f"📂 数据目录: {pack.get('data_dir')}")
    print(f" 👤userid: {str(user_id)}")

    preprocessor = SingleSamplePreprocessor(root_dir=pack["data_dir"])
    img_t, audio_t, tab_t = preprocessor.process_with_tab_values(user_id, height_cm, weight_kg, age)

    img_t = img_t.unsqueeze(0).to(device)
    audio_t = audio_t.unsqueeze(0).to(device)
    tab_t = tab_t.unsqueeze(0).to(device)

    with torch.no_grad():
        fused_feat = pack["model"](img_t, audio_t, tab_t)
        logits = pack["classifier"](fused_feat)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, predicted = torch.max(probs, 0)

    pred_idx = int(predicted.item())
    conf_score = float(confidence.item())
    class_map = pack.get("class_map", None)
    pred_label = get_class_name(pred_idx, class_map)

    print(f"📊 预测结果: 🟢 {pred_label}")

    prob_list: List[Dict[str, Any]] = []
    probs_np = probs.detach().cpu().numpy().tolist()
    for i, p in enumerate(probs_np):
        prob_list.append(
            {
                "idx": i,
                "label": get_class_name(i, class_map),
                "prob": float(p),
                "prob_percent": float(p) * 100.0,
            }
        )
    prob_list.sort(key=lambda x: x["prob"], reverse=True)

    return {
        "user_id": str(user_id),
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "confidence": conf_score,
        "confidence_percent": conf_score * 100.0,
        "probs": prob_list,
        "device": device.type,
        "checkpoint_path": pack.get("checkpoint_path"),
    }



# def main():
#     import yaml # 动态导入以防万一
    
#     # 1. 配置
#     config_path = 'config/model_config.yaml'
#     data_dir = 'data/health_dataset' # 请根据实际路径修改，或者通过输入指定
    
#     # 尝试从 config 读取 data_dir
#     cfg = None
#     if os.path.exists(config_path):
#         with open(config_path, 'r') as f:
#             cfg = yaml.safe_load(f)
#             if 'data_dir' in cfg:
#                 data_dir = cfg['data_dir']
    
#     if not os.path.exists(data_dir):
#         data_dir = input(f"默认数据路径 '{data_dir}' 不存在，请输入数据集根目录路径: ").strip()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"🚀 设备: {device}")

#     # 2. 初始化预处理器
#     try:
#         preprocessor = SingleSamplePreprocessor(root_dir=data_dir)
#     except Exception as e:
#         print(f"❌ 初始化预处理器失败: {e}")
#         return

#     # 3. 加载模型
#     checkpoint_path = 'checkpoints/best_model.pth'
#     if not os.path.exists(checkpoint_path):
#         # 尝试查找其他可能
#         if os.path.exists('checkpoints/best_model_f1.pth'):
#             checkpoint_path = 'checkpoints/best_model_f1.pth'
#         else:
#             print(f"❌ 未找到模型文件: {checkpoint_path}")
#             return

#     print(f"📥 加载模型: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     # 确保 cfg 有必要的键，如果没有则从 checkpoint 猜或使用默认值
#     if cfg is None:
#         cfg = checkpoint.get('config', {}) # 尝试从 checkpoint 恢复配置
        
#     # 必要的默认值防止报错
#     fusion_dim = cfg.get('fusion_dim', 256)
#     num_heads = cfg.get('num_heads', 4)
#     depth = cfg.get('depth', 4)

#     model = FusionTransformer(
#         fusion_dim=fusion_dim,
#         num_heads=num_heads,
#         depth=depth
#     ).to(device)
    
#     classifier = ClassifierHead(input_dim=fusion_dim).to(device)
    
#     model.load_state_dict(checkpoint['model_state_dict'])
#     classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
#     model.eval()
#     classifier.eval()
    
#     class_map = checkpoint.get('class_mapping', None)

#     print("\n" + "="*60)
#     print("🩺 健康评估模型 - 单样本实战测试")
#     print("="*60)
#     print(f"📂 数据目录: {data_dir}")
#     print("📝 使用说明:")
#     print("   输入 user_id (即文件名，不含后缀)，程序将自动查找:")
#     print(f"   - {data_dir}/images/{{user_id}}.*")
#     print(f"   - {data_dir}/audios/{{user_id}}.*")
#     print(f"   - {data_dir}/tabular.csv 中的对应行")
#     print("="*60)

#     while True:
#         uid = input("\n👤 请输入 User ID: ").strip()
#         if uid.lower() == 'q':
#             print("👋 退出测试。")
#             break
#         if not uid:
#             continue

#         try:
#             print("⏳ 正在预处理数据...")
#             img_t, audio_t, tab_t = preprocessor.process(uid)
            
#             # 移动到设备
#             img_t = img_t.unsqueeze(0).to(device)      # [1, 3, 64, 64]
#             audio_t = audio_t.unsqueeze(0).to(device)  # [1, 1, 64, T]
#             tab_t = tab_t.unsqueeze(0).to(device)      # [1, 3]

#             # 推理
#             with torch.no_grad():
#                 fused_feat = model(img_t, audio_t, tab_t)
#                 logits = classifier(fused_feat)
                
#                 probs = torch.softmax(logits, dim=1)
#                 confidence, predicted = torch.max(probs, 1)
                
#                 pred_idx = predicted.item()
#                 conf_score = confidence.item()
#                 pred_label = get_class_name(pred_idx, class_map)

#             print("\n" + "*"*40)
#             print(f"📊 预测结果: 🟢 {pred_label}")
#             print(f"🎯 置信度: {conf_score:.4f} ({conf_score*100:.2f}%)")
            
#             # 显示详细概率
#             print("📈 概率分布:")
#             labels_list = list(class_map.values()) if class_map else [f"Class_{i}" for i in range(logits.shape[1])]
#             # 如果 class_map 是 {0: 'Name'} 这种格式，我们需要反转一下或者直接遍历
#             # 这里假设 class_map 是 {idx: name}
            
#             prob_array = probs.cpu().numpy()[0]
#             for i, p in enumerate(prob_array):
#                 name = get_class_name(i, class_map)
#                 bar = "█" * int(p * 20)
#                 print(f"   {name:10s}: {p:.4f} {bar}")
            
#             print("*"*40)

#         except Exception as e:
#             print(f"\n❌ 发生错误: {e}")
#             import traceback
#             traceback.print_exc()

# if __name__ == '__main__':
#     main()
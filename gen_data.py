#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import hdf5storage

# === 配置 ===
input_dir = "/home/zmd/RC-Flow/data"   # 包含 .mat 文件的文件夹
output_dir = "./bin"                   # 输出 .npy 文件的文件夹

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历 input_dir 中所有 .mat 文件
for filename in os.listdir(input_dir):
    if not filename.endswith(".mat"):
        continue

    mat_path = os.path.join(input_dir, filename)
    name_no_ext = filename[:-4]  # 移除 ".mat"

    # 尝试用下划线分割解析文件名
    parts = name_no_ext.split('_')
    if len(parts) < 5:
        print(f"Skipping (unexpected format): {filename}")
        continue

    # 期望格式: CDL-X_NtXX_NrXX_ULAXX_seedXXXX
    try:
        profile = parts[0]      # e.g., "CDL-A"
        nt_str  = parts[1]      # e.g., "Nt64"
        nr_str  = parts[2]      # e.g., "Nr16"
        ulastr  = parts[3]      # e.g., "ULA0.50"
        seed    = parts[4]      # e.g., "seed1234"

        # 验证前缀是否符合预期（可选）
        if not (profile.startswith("CDL-") and nt_str.startswith("Nt") and nr_str.startswith("Nr") and ulastr.startswith("ULA")):
            raise ValueError("Prefix mismatch")

        # 构造新的文件名：ULA → UPA，去掉 seed 部分
        spacing_val = ulastr[3:]  # "0.50"
        if seed == "seed1234":
            npy_name = f"{profile}_{nt_str}_{nr_str}_UPA{spacing_val}_train.npy"
        elif seed == "seed4321":
            npy_name = f"{profile}_{nt_str}_{nr_str}_UPA{spacing_val}_test.npy"
        npy_path = os.path.join(output_dir, npy_name)

        # 加载 .mat
        print(f"Processing: {filename}")
        data = hdf5storage.loadmat(mat_path)
        output_h = data['output_h']  # shape: (N, num_sc, Nr, Nt) or similar

        # 提取第一个子载波
        H_cplx = output_h[:, 0, :, :]  # (N, Nr, Nt)
        H_cplx = H_cplx.astype(np.complex64)

        

        # 保存
        np.save(npy_path, H_cplx)
        print(f"  → Saved {H_cplx.shape} to {npy_name}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue


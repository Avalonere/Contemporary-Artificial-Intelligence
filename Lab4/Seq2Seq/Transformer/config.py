from dataclasses import dataclass

import torch


@dataclass
class Config:
    # 模型参数
    model_name_map = {
        "bart": "facebook/bart-base",  # 约140M参数
        "t5": "google-t5/t5-small",  # 约60M参数
    }
    max_source_length = 148  # 根据统计结果设置
    max_target_length = 79

    # 训练参数
    model = "bart"
    batch_size = 8
    learning_rate = 5e-5
    num_epochs = 10
    warmup_ratio = 0.1
    num_beams = 6
    no_repeat_ngram_size = 3
    num_beam_groups = 2
    diversity_penalty = 1.0

    # 其他
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 4001
    patience = 3

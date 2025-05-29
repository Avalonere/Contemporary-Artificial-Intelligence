from abc import ABC, abstractmethod

import torch.nn as nn


class BaseSeq2Seq(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def generate(self, src, encoder_outputs=None, hidden=None, max_len=50):
        """统一生成函数接口
        Args:
            src: 输入序列
            encoder_outputs: 编码器输出 (可选)
            hidden: 隐藏状态 (可选)
            max_len: 最大生成长度
        """
        pass

    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

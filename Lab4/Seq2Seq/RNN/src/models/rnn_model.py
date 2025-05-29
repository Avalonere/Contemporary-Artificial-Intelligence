import torch
import torch.nn as nn

from .base_model import BaseSeq2Seq


class RNNSeq2Seq(BaseSeq2Seq):
    def __init__(self, config):
        super().__init__(config)

        # 编码器
        self.encoder_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.encoder_rnn = nn.RNN(config.embedding_dim,
                                  config.hidden_size,
                                  num_layers=config.num_layers,
                                  dropout=config.dropout if config.num_layers > 1 else 0,
                                  batch_first=True)

        # 解码器
        self.decoder_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.decoder_rnn = nn.RNN(config.embedding_dim,
                                  config.hidden_size,
                                  num_layers=config.num_layers,
                                  dropout=config.dropout if config.num_layers > 1 else 0,
                                  batch_first=True)
        self.fc_out = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, src, tgt=None):
        # 编码
        embedded = self.encoder_embedding(src)
        encoder_outputs, hidden = self.encoder_rnn(embedded)

        if tgt is None:  # 推理模式
            return self.generate(src)

        # 训练模式
        decoder_input = tgt[:, :-1]  # 去除最后一个token
        embedded = self.decoder_embedding(decoder_input)
        decoder_outputs, _ = self.decoder_rnn(embedded, hidden)
        logits = self.fc_out(decoder_outputs)

        return logits

    def generate(self, src, encoder_outputs=None, hidden=None, max_len=50):
        batch_size = src.size(0)
        device = src.device

        # 如果没有提供encoder输出和隐藏状态,重新计算
        if encoder_outputs is None or hidden is None:
            embedded = self.encoder_embedding(src)
            encoder_outputs, hidden = self.encoder_rnn(embedded)

        decoder_input = torch.ones(batch_size, 1).long().to(device)
        outputs = []

        for _ in range(max_len):
            embedded = self.decoder_embedding(decoder_input)
            decoder_output, hidden = self.decoder_rnn(embedded, hidden)
            logits = self.fc_out(decoder_output)
            next_token = logits.argmax(dim=-1)
            outputs.append(next_token)
            decoder_input = next_token

        return torch.cat(outputs, dim=1)

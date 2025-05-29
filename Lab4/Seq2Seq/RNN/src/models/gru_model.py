import torch
import torch.nn as nn

from .base_model import BaseSeq2Seq


class GRUSeq2Seq(BaseSeq2Seq):
    def __init__(self, config):
        super().__init__(config)

        # 编码器
        self.encoder_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.encoder_gru = nn.GRU(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )

        # 解码器
        decoder_hidden = config.hidden_size * (2 if config.bidirectional else 1)
        self.decoder_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.decoder_gru = nn.GRU(
            config.embedding_dim,
            decoder_hidden,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )

        # 注意力层
        self.attention = nn.Linear(decoder_hidden + config.hidden_size * 2, 1)

        # 输出层
        self.fc_out = nn.Linear(decoder_hidden + config.hidden_size * 2, config.vocab_size)

    def forward(self, src, tgt=None):
        batch_size = src.size(0)

        # 编码
        embedded = self.encoder_embedding(src)
        encoder_outputs, hidden = self.encoder_gru(embedded)

        # 处理双向GRU的隐藏状态
        if self.config.bidirectional:
            hidden = hidden.view(self.config.num_layers, 2, batch_size, -1)
            hidden = hidden.transpose(1, 2).contiguous()
            hidden = hidden.reshape(self.config.num_layers, batch_size, -1)

        if tgt is None:
            return self.generate(src, encoder_outputs, hidden)

        # 训练模式
        decoder_input = tgt[:, :-1]
        decoder_embedded = self.decoder_embedding(decoder_input)
        decoder_outputs, _ = self.decoder_gru(decoder_embedded, hidden)

        # 注意力计算
        seq_len = decoder_outputs.size(1)
        enc_len = encoder_outputs.size(1)

        # 扩展维度以便计算注意力分数
        decoder_outputs_expanded = decoder_outputs.unsqueeze(2).expand(
            batch_size, seq_len, enc_len, -1)
        encoder_outputs_expanded = encoder_outputs.unsqueeze(1).expand(
            batch_size, seq_len, enc_len, -1)

        # 计算注意力分数
        attn_input = torch.cat((decoder_outputs_expanded, encoder_outputs_expanded), dim=3)
        attn_scores = self.attention(attn_input).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=2)

        # 计算上下文向量
        context = torch.bmm(attn_weights, encoder_outputs)

        # 拼接并输出
        output = torch.cat((decoder_outputs, context), dim=2)
        logits = self.fc_out(output)

        return logits

    def generate(self, src, encoder_outputs=None, hidden=None, max_len=50):
        batch_size = src.size(0)
        device = src.device

        # 如果没有提供encoder输出和隐藏状态,重新计算
        if encoder_outputs is None or hidden is None:
            embedded = self.encoder_embedding(src)
            encoder_outputs, hidden = self.encoder_gru(embedded)
            if self.config.bidirectional:
                hidden = hidden.view(self.config.num_layers, 2, batch_size, -1)
                hidden = hidden.transpose(1, 2).contiguous()
                hidden = hidden.reshape(self.config.num_layers, batch_size, -1)

        decoder_input = torch.ones(batch_size, 1).long().to(device)
        outputs = []

        for _ in range(max_len):
            decoder_embedded = self.decoder_embedding(decoder_input)
            decoder_output, hidden = self.decoder_gru(decoder_embedded, hidden)

            # 计算注意力
            attn_input = torch.cat([
                decoder_output.expand(-1, encoder_outputs.size(1), -1),
                encoder_outputs
            ], dim=2)

            # 确保维度正确性 [batch_size, seq_len, hidden_dim]
            scores = self.attention(attn_input)
            attention_weights = torch.softmax(scores, dim=1).transpose(1, 2)  # [B, 1, seq_len]

            # bmm: [B, 1, seq_len] x [B, seq_len, hidden]
            context = torch.bmm(attention_weights, encoder_outputs)

            # 拼接解码器输出和上下文向量
            output = torch.cat([decoder_output, context], dim=2)
            logits = self.fc_out(output)
            next_token = logits.argmax(dim=-1)
            outputs.append(next_token)
            decoder_input = next_token

        return torch.cat(outputs, dim=1)

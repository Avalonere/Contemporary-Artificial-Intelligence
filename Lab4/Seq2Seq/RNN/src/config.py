import torch


class Config:
    # 数据相关
    train_file = '../data/train.csv'
    test_file = '../data/test.csv'
    max_src_len = 148
    max_tgt_len = 79

    # 模型相关
    model_type = 'gru'
    vocab_size = 1300
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.1

    # 训练相关
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    learning_rate = 0.001
    validation_split = 0.1
    random_seed = 42

    # 优化器配置
    weight_decay = 1e-2
    max_grad_norm = 1.0

    # 学习率调度
    warmup_ratio = 0.1
    min_lr = 1e-6

    # 训练控制
    max_epochs = 20
    patience = 3

    # 路径
    model_dir = './outputs/models'
    log_dir = './outputs/logs'
    output_dir = "./outputs/predictions"

    # LSTM/GRU特定参数
    bidirectional = True
    n_directions = 2 if bidirectional else 1

    # 特殊标记
    pad_token_id = 0
    sos_token_id = 1
    eos_token_id = 2

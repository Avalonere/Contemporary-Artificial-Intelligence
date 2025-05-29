from .gru_model import GRUSeq2Seq
from .lstm_model import LSTMSeq2Seq
from .rnn_model import RNNSeq2Seq


class ModelFactory:
    @staticmethod
    def create_model(model_name, config):
        if model_name == 'rnn':
            return RNNSeq2Seq(config)
        elif model_name == 'lstm':
            return LSTMSeq2Seq(config)
        elif model_name == 'gru':
            return GRUSeq2Seq(config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

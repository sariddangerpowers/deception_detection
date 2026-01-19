from .sequence_branch import SequenceBranch

class TextBranch(SequenceBranch):
    """
    Text branch processing 300D GloVe embeddings using Stacked Bi-LSTM.
    """
    def __init__(self, input_dim=None, hidden_dim=128, num_layers=2, output_dim=128, dropout=0.3, config=None):
        if config is None:
            from src.config import ModelConfig
            config = ModelConfig()
            
        if input_dim is None:
            input_dim = config.text_input_dim
            
        super(TextBranch, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            config=config # Pass through
        )

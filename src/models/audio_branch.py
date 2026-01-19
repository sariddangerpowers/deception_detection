from .sequence_branch import SequenceBranch

class AudioBranch(SequenceBranch):
    """
    Audio branch processing 39 MFCC features using Stacked Bi-LSTM.
    """
    def __init__(self, input_dim=39, hidden_dim=128, num_layers=2, output_dim=128, dropout=0.3, config=None):
        super(AudioBranch, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            config=config
        )

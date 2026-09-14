import torch

from nonergodic_memory.models.sequence import GRUPredictor, TransformerPredictor


def test_models_return_logits_and_hidden_states() -> None:
    tokens = torch.tensor([[0, 1, 2], [2, 1, 0]])
    for model in (
        GRUPredictor(vocab_size=4, width=8, layers=1),
        TransformerPredictor(vocab_size=4, width=8, layers=1, heads=2, max_length=8),
    ):
        logits, hidden = model(tokens)
        assert logits.shape == (2, 3, 4)
        assert hidden.shape == (2, 3, 8)


def test_transformer_is_causal() -> None:
    torch.manual_seed(2)
    model = TransformerPredictor(4, width=8, layers=1, heads=2, max_length=8)
    model.eval()
    left, _ = model(torch.tensor([[0, 1, 2, 3]]))
    right, _ = model(torch.tensor([[0, 1, 0, 0]]))
    torch.testing.assert_close(left[:, :2], right[:, :2])


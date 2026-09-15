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


def test_transformer_layerwise_continuations_reproduce_logits() -> None:
    torch.manual_seed(5)
    model = TransformerPredictor(4, width=8, layers=2, heads=2, max_length=8)
    model.eval()
    tokens = torch.tensor([[0, 1, 2, 3]])
    expected, layer_activations = model.forward_with_layers(tokens)
    assert len(layer_activations) == 3  # two block outputs plus final normalization
    for depth, hidden in enumerate(layer_activations):
        actual, _ = model.logits_from_depth(hidden, depth)
        torch.testing.assert_close(actual, expected)


def test_transformer_accepts_per_example_absolute_position_offsets() -> None:
    torch.manual_seed(7)
    model = TransformerPredictor(4, width=8, heads=2, max_length=16).eval()
    tokens = torch.tensor([[0, 1, 2], [0, 1, 2]])
    reset, _ = model(tokens)
    absolute, _ = model(tokens, position_offsets=torch.tensor([0, 5]))
    torch.testing.assert_close(absolute[0], reset[0])
    assert not torch.allclose(absolute[1], reset[1])
    with torch.no_grad():
        try:
            model(tokens, position_offsets=torch.tensor([0, 14]))
        except ValueError as exc:
            assert "position" in str(exc)
        else:
            raise AssertionError("out-of-range offset was accepted")

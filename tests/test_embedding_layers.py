import math

import pytest
import torch

from experiments.common.embedding_layers import (
    ConcatEmbedding,
    PeriodicEmbedding,
    TrainableFourierEmbedding,
)


def test_concat_embedding_returns_coordinates_unchanged():
    t = torch.tensor([[0.0], [0.5]])
    x = torch.tensor([[-1.0], [1.0]])
    embedding = ConcatEmbedding(2)

    result = embedding(t, x)

    assert embedding.out_dim == 2
    assert torch.equal(result, torch.cat([t, x], dim=1))


def test_periodic_embedding_default_order_and_values():
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    x = torch.tensor([[0.0], [0.5]], dtype=torch.float64)
    embedding = PeriodicEmbedding(2, n_harmonics=3, wavenumber=math.pi)

    result = embedding(t, x)
    expected = torch.cat(
        [
            t,
            torch.cos(math.pi * x),
            torch.sin(math.pi * x),
            torch.cos(2.0 * math.pi * x),
            torch.sin(2.0 * math.pi * x),
            torch.cos(3.0 * math.pi * x),
            torch.sin(3.0 * math.pi * x),
        ],
        dim=1,
    )

    assert embedding.out_dim == 7
    assert torch.allclose(result, expected)
    assert torch.allclose(embedding(t, x + 2.0), result, atol=1e-14)


def test_periodic_embedding_can_embed_every_coordinate():
    x = torch.tensor([[0.1], [0.2]])
    y = torch.tensor([[0.3], [0.4]])
    embedding = PeriodicEmbedding(
        2,
        n_harmonics=2,
        wavenumber=math.pi,
        periodic_dims=(0, 1),
    )

    result = embedding(x, y)

    assert embedding.out_dim == 8
    assert result.shape == (2, 8)
    x_only = PeriodicEmbedding(
        1, n_harmonics=2, wavenumber=math.pi, periodic_dims=(0,)
    )(x)
    assert torch.allclose(result[:, :4], x_only)


@pytest.mark.parametrize("include_input, expected_dim", [(False, 8), (True, 10)])
def test_trainable_fourier_embedding_shape_and_parameters(
    include_input, expected_dim
):
    torch.manual_seed(0)
    embedding = TrainableFourierEmbedding(
        2, embed_dim=8, scale=2.0, include_input=include_input
    )
    t = torch.rand(5, 1)
    x = torch.rand(5, 1)

    result = embedding(t, x)
    result[:, -embedding.embed_dim:][:, :4].sum().backward()

    assert embedding.out_dim == expected_dim
    assert result.shape == (5, expected_dim)
    assert isinstance(embedding.B, torch.nn.Parameter)
    assert embedding.B.grad is not None
    assert torch.isfinite(embedding.B.grad).all()
    assert torch.count_nonzero(embedding.B.grad)
    if include_input:
        assert torch.equal(result[:, :2], torch.cat([t, x], dim=1))


@pytest.mark.parametrize(
    "embedding",
    [
        ConcatEmbedding(2),
        PeriodicEmbedding(2),
        TrainableFourierEmbedding(2, embed_dim=4),
    ],
)
def test_embeddings_validate_coordinate_count(embedding):
    with pytest.raises(ValueError, match="expected 2 coordinate tensors"):
        embedding(torch.zeros(2, 1))

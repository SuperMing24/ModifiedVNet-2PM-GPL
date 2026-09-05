# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import pytest
import torch

from modified_vnet_2pm.data import (
    TianDamsehGridPatchDataset3D,
    _official_grid_starts,
    normalize_volume,
)
from modified_vnet_2pm.loss import TianDamsehTVBCELoss
from modified_vnet_2pm.model import TianDamsehVNet


def test_model_forward_backward_and_shape_guard():
    model = TianDamsehVNet()
    data = torch.randn(1, 1, 16, 16, 16)
    target = torch.rand_like(data)
    logits = model(data)
    loss = TianDamsehTVBCELoss()(logits, target)
    loss.backward()
    assert logits.shape == data.shape
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="divisible by 8"):
        model(torch.randn(1, 1, 15, 16, 16))


def test_official_grid_and_dataset_are_deterministic():
    assert _official_grid_starts(512, 128, 64) == (
        0,
        64,
        128,
        192,
        256,
        320,
        384,
    )
    image = np.arange(128 * 128 * 32, dtype=np.float32).reshape(128, 128, 32)
    mask = (image % 7 == 0).astype(np.float32)
    dataset = TianDamsehGridPatchDataset3D(
        [{"image": image, "mask": mask, "name": "sample"}]
    )
    first = dataset[0]
    second = dataset[0]
    assert len(dataset) == 1
    assert first[0].shape == (1, 128, 128, 128)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_normalize_constant_volume_is_finite():
    normalized = normalize_volume(np.ones((2, 2, 2), dtype=np.float32))
    assert np.array_equal(normalized, np.zeros_like(normalized))

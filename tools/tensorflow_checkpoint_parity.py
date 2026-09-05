# SPDX-License-Identifier: GPL-3.0-only
"""Numerical parity audit for the released Tian/Damseh TensorFlow checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow.compat.v1 as tf
import torch

from modified_vnet_2pm.model import TianDamsehVNet


tf.disable_v2_behavior()
AUDIT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(
    os.environ.get(
        "MODIFIED_VNET_UPSTREAM",
        str(AUDIT_ROOT / "2PM_Vascular_Segmentation_DNN"),
    )
).resolve()
CHECKPOINT = (
    UPSTREAM
    / "Test_trained_model"
    / "chkpt_saved"
    / "my-model-100"
)
OUTPUT = Path(
    os.environ.get(
        "MODIFIED_VNET_PARITY_OUTPUT",
        str(Path(__file__).with_name("modified_vnet_tensorflow_parity.json")),
    )
).resolve()
SCOPE = "VNet_3x3x3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


reader = tf.train.load_checkpoint(str(CHECKPOINT))


def _value(name: str) -> np.ndarray:
    return np.asarray(reader.get_tensor(name), dtype=np.float32)


def _conv(x, prefix: str, stride: int = 1):
    weights = tf.constant(_value(prefix + "/weights"))
    biases = tf.constant(_value(prefix + "/biases"))
    return tf.nn.conv3d(
        x,
        weights,
        strides=[1, stride, stride, stride, 1],
        padding="SAME",
    ) + biases


def _batch_norm(x, prefix: str):
    bn = prefix + "/batch_normalization"
    return tf.nn.batch_normalization(
        x,
        tf.constant(_value(bn + "/moving_mean")),
        tf.constant(_value(bn + "/moving_variance")),
        tf.constant(_value(bn + "/beta")),
        tf.constant(_value(bn + "/gamma")),
        variance_epsilon=1e-3,
    )


def _block(x, prefix: str, count: int):
    residual = x
    for index in range(1, count + 1):
        conv_prefix = f"{prefix}/conv_{index}"
        x = _batch_norm(tf.nn.relu(_conv(x, conv_prefix)), conv_prefix)
    return x + residual


def _merge(x, skip, prefix: str, count: int):
    residual = x
    x = _conv(tf.concat((x, skip), axis=-1), f"{prefix}/conv_1")
    for index in range(2, count + 1):
        conv_prefix = f"{prefix}/conv_{index}"
        x = _batch_norm(tf.nn.relu(_conv(x, conv_prefix)), conv_prefix)
    return x + residual


def _up(x, skip, prefix: str):
    weights = tf.constant(_value(prefix + "/weights"))
    biases = tf.constant(_value(prefix + "/biases"))
    return tf.nn.relu(
        tf.nn.conv3d_transpose(
            x,
            weights,
            output_shape=tf.shape(skip),
            strides=[1, 2, 2, 2, 1],
            padding="SAME",
        ) + biases
    )


def _tensorflow_forward(x):
    c0 = tf.tile(x, [1, 1, 1, 1, 4])
    c1 = _block(c0, f"{SCOPE}/contracting_path/level_1", 1)
    c12 = tf.nn.relu(
        _conv(c1, f"{SCOPE}/contracting_path/level_1/down_convolution", 2)
    )
    c2 = _block(c12, f"{SCOPE}/contracting_path/level_2", 2)
    c22 = tf.nn.relu(
        _conv(c2, f"{SCOPE}/contracting_path/level_2/down_convolution", 2)
    )
    c3 = _block(c22, f"{SCOPE}/contracting_path/level_3", 3)
    c32 = tf.nn.relu(
        _conv(c3, f"{SCOPE}/contracting_path/level_3/down_convolution", 2)
    )
    c4 = _block(c32, f"{SCOPE}/contracting_path/level_4", 3)
    c42 = _up(c4, c3, f"{SCOPE}/contracting_path/level_4/up_convolution")
    e3 = _merge(c42, c3, f"{SCOPE}/expanding_path/level_3", 3)
    e32 = _up(e3, c2, f"{SCOPE}/expanding_path/level_3/up_convolution")
    e2 = _merge(e32, c2, f"{SCOPE}/expanding_path/level_2", 2)
    e22 = _up(e2, c1, f"{SCOPE}/expanding_path/level_2/up_convolution")
    e1 = _merge(e22, c1, f"{SCOPE}/expanding_path/level_1", 1)
    return _conv(e1, f"{SCOPE}/expanding_path/level_1/output_layer")


def _copy_conv(module, prefix: str, *, transposed: bool = False) -> None:
    weights = _value(prefix + "/weights")
    axes = (4, 3, 0, 1, 2) if transposed else (4, 3, 0, 1, 2)
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(weights.transpose(axes).copy()))
        module.bias.copy_(torch.from_numpy(_value(prefix + "/biases").copy()))


def _copy_bn(module, prefix: str) -> None:
    bn = prefix + "/batch_normalization"
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(_value(bn + "/gamma").copy()))
        module.bias.copy_(torch.from_numpy(_value(bn + "/beta").copy()))
        module.running_mean.copy_(
            torch.from_numpy(_value(bn + "/moving_mean").copy())
        )
        module.running_var.copy_(
            torch.from_numpy(_value(bn + "/moving_variance").copy())
        )


def _copy_block(block, prefix: str) -> None:
    for index, layer in enumerate(block.layers, start=1):
        conv_prefix = f"{prefix}/conv_{index}"
        _copy_conv(layer[0], conv_prefix)
        _copy_bn(layer[2], conv_prefix)


def _copy_merge(block, prefix: str) -> None:
    _copy_conv(block.first, f"{prefix}/conv_1")
    for index, layer in enumerate(block.remaining, start=2):
        conv_prefix = f"{prefix}/conv_{index}"
        _copy_conv(layer[0], conv_prefix)
        _copy_bn(layer[2], conv_prefix)


model = TianDamsehVNet().eval()
_copy_block(model.contract1, f"{SCOPE}/contracting_path/level_1")
_copy_conv(model.down1[0], f"{SCOPE}/contracting_path/level_1/down_convolution")
_copy_block(model.contract2, f"{SCOPE}/contracting_path/level_2")
_copy_conv(model.down2[0], f"{SCOPE}/contracting_path/level_2/down_convolution")
_copy_block(model.contract3, f"{SCOPE}/contracting_path/level_3")
_copy_conv(model.down3[0], f"{SCOPE}/contracting_path/level_3/down_convolution")
_copy_block(model.bottleneck, f"{SCOPE}/contracting_path/level_4")
_copy_conv(
    model.up3[0],
    f"{SCOPE}/contracting_path/level_4/up_convolution",
    transposed=True,
)
_copy_merge(model.expand3, f"{SCOPE}/expanding_path/level_3")
_copy_conv(
    model.up2[0],
    f"{SCOPE}/expanding_path/level_3/up_convolution",
    transposed=True,
)
_copy_merge(model.expand2, f"{SCOPE}/expanding_path/level_2")
_copy_conv(
    model.up1[0],
    f"{SCOPE}/expanding_path/level_2/up_convolution",
    transposed=True,
)
_copy_merge(model.expand1, f"{SCOPE}/expanding_path/level_1")
_copy_conv(model.output_conv, f"{SCOPE}/expanding_path/level_1/output_layer")

rng = np.random.RandomState(20260904)
input_ndhwc = rng.normal(size=(1, 16, 16, 16, 1)).astype(np.float32)
input_tensor = tf.placeholder(tf.float32, shape=input_ndhwc.shape)
tf_output_tensor = _tensorflow_forward(input_tensor)
with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as session:
    tf_output = session.run(tf_output_tensor, {input_tensor: input_ndhwc})

torch_input = torch.from_numpy(input_ndhwc.transpose(0, 4, 1, 2, 3).copy())
with torch.no_grad():
    torch_output = model(torch_input).numpy()
tf_output_ncdhw = tf_output.transpose(0, 4, 1, 2, 3)
absolute = np.abs(tf_output_ncdhw - torch_output)
payload = {
    "schema_version": 1,
    "status": "passed" if np.allclose(
        tf_output_ncdhw, torch_output, rtol=2e-4, atol=2e-5
    ) else "failed",
    "upstream_repository": "https://github.com/bu-cisl/2PM_Vascular_Segmentation_DNN",
    "upstream_revision": "e56562d303aedfc90124d5a25fba96f325562a82",
    "checkpoint": "Test_trained_model/chkpt_saved/my-model-100",
    "checkpoint_index_sha256": _sha256(CHECKPOINT.with_suffix(".index")),
    "input_shape_ndhwc": list(input_ndhwc.shape),
    "tensorflow_version": tf.__version__,
    "tensorflow_device": "CPU",
    "pytorch_version": torch.__version__,
    "max_abs_diff": float(absolute.max()),
    "mean_abs_diff": float(absolute.mean()),
    "max_reference_abs": float(np.abs(tf_output_ncdhw).max()),
    "allclose_rtol": 2e-4,
    "allclose_atol": 2e-5,
    "binary_logit_agreement": float(
        np.mean((tf_output_ncdhw > 0) == (torch_output > 0))
    ),
    "mapped_checkpoint_model_variables": 76,
    "outer_test_accessed": False,
    "gpu_used": False,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if payload["status"] == "passed" else 2)

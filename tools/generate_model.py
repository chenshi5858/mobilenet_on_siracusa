#!/usr/bin/env python3
"""Build a tiny MobileNet-like integer model and emit plain C tensors.

This script is intentionally independent from Deeploy, ONNX and PyTorch. It
defines the model, evaluates a CPU reference, packs N-EUREKA weights and emits
the data consumed by the manually written C application.
"""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "model_data.c"

INPUT_H = 10
INPUT_W = 10
CLASS_COUNT = 3
WEIGHT_BITS = 8


def conv_valid(
    input_hwc: np.ndarray,
    weights_oihw: np.ndarray,
    shift: int,
    depthwise: bool = False,
) -> np.ndarray:
    """Integer convolution matching N-EUREKA norm/requant semantics."""
    # Esta función implementa una convolución entera de referencia. Recibe un tensor de entrada, los pesos de una capa
    # convolucional y ejecuta manualmente la convolución, seguida de ReLU + shift/recuantización + saturación a uint8.

    # La idea es que el resultado calculado por esta función sea el que después N-EUREKA debería reproducir.
    # weights_oihw contiene los pesos en formato O (canales de salida) x I (canales de entrada) x H (alto kernel) x W (ancho kernel)
    # input_hwc contiene la entrada en formato H (alto) x W (ancho) x C (canales)
    kernel_h, kernel_w = weights_oihw.shape[2:] # kernel_h y kernel_w son las dimensiones del kernel de convolución
    output_h = input_hwc.shape[0] - kernel_h + 1 # output_h es la altura de la salida después de la convolución válida
    output_w = input_hwc.shape[1] - kernel_w + 1 # output_w es la anchura de la salida después de la convolución válida
    output_c = weights_oihw.shape[0] # output_c es el número de canales de salida, que corresponde al número de filtros en la capa convolucional
    output = np.zeros((output_h, output_w, output_c), dtype=np.int64) # output es el tensor de salida inicializado a cero, con dimensiones (altura de salida, anchura de salida, canales de salida)

    for h_out in range(output_h):
        for w_out in range(output_w):
            # recorre cada posición de la salida y calcula la convolución correspondiente
            window = input_hwc[
                h_out : h_out + kernel_h, w_out : w_out + kernel_w, :
            ] # window es la ventana de entrada que se va a convolucionar, seleccionada de la entrada original. Es decir,
            # extrae la ventana. ejemplo: si kernel_h=3 y kernel_w=3, entonces window será un bloque de 3x3 de la entrada en la posición (h_out, w_out)
            # windows es un arreglo de dimensiones (kernel_h, kernel_w, input_c).
            if depthwise:
                for channel in range(output_c):
                    output[h_out, w_out, channel] = np.sum(
                        window[:, :, channel] * weights_oihw[channel, 0]
                    )
            else:
                for channel in range(output_c):
                    kernel_hwc = weights_oihw[channel].transpose(1, 2, 0)
                    output[h_out, w_out, channel] = np.sum(window * kernel_hwc)

    output = np.maximum(output, 0)
    output = output >> shift
    return np.clip(output, 0, 255).astype(np.uint8)


def pack_neureka_weights(
    signed_weights: np.ndarray, depthwise: bool = False
) -> np.ndarray:
    """Pack OIHW int8 weights into N-EUREKA's 256-bit bit-plane layout."""
    if signed_weights.dtype != np.int8:
        raise TypeError("weights must use int8 storage")

    # N-EUREKA applies -128 as a layer-wise offset during execution.
    weights = (signed_weights.astype(np.int16) + 128).astype(np.uint8)
    if depthwise:
        weights = weights.transpose(1, 0, 2, 3)

    output_c, input_c, kernel_h, kernel_w = weights.shape
    input_subtile = 28 if kernel_h == 3 else 32
    input_major = (input_c + input_subtile - 1) // input_subtile
    padded_input_c = input_major * input_subtile

    if padded_input_c != input_c:
        weights = np.pad(
            weights,
            ((0, 0), (0, padded_input_c - input_c), (0, 0), (0, 0)),
        )

    weights = weights.reshape(
        output_c, input_major, input_subtile, kernel_h * kernel_w, 1
    )
    weights = np.unpackbits(
        weights, axis=-1, count=WEIGHT_BITS, bitorder="little"
    )
    weights = weights.transpose(0, 1, 4, 3, 2)

    if (kernel_h, kernel_w) == (3, 3):
        weights = weights.reshape(-1, kernel_h * kernel_w * input_subtile)
        weights = np.pad(weights, ((0, 0), (0, 256 - weights.shape[-1])))
    elif (kernel_h, kernel_w) == (1, 1):
        weights = weights.reshape(
            output_c, input_major, WEIGHT_BITS, 1, input_subtile // 4, 4
        )
        weights = weights.transpose(0, 1, 3, 4, 2, 5)
        weights = weights.reshape(output_c * input_major, 256)
    else:
        raise ValueError("N-EUREKA only supports 1x1 and 3x3 kernels")

    weights = weights.reshape(-1, 8)
    return np.packbits(weights, axis=-1, bitorder="little").flatten()


def make_model():
    # Three correlation filters: vertical, horizontal and main diagonal.
    stem = np.array(
        [
            [[[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]],
            [[[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]],
            [[[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]],
        ],
        dtype=np.int8,
    )

    # Depthwise spatial aggregation, one independent kernel per channel.
    depthwise = np.ones((3, 1, 3, 3), dtype=np.int8)

    # MobileNet-style pointwise expansion from 3 to 6 channels.
    pointwise = np.zeros((6, 3, 1, 1), dtype=np.int8)
    for channel in range(3):
        pointwise[2 * channel, channel, 0, 0] = 1
        pointwise[2 * channel + 1, channel, 0, 0] = 1

    # Pointwise projection from 6 channels to the 3 class maps.
    head = np.zeros((3, 6, 1, 1), dtype=np.int8)
    for channel in range(3):
        head[channel, 2 * channel : 2 * channel + 2, 0, 0] = 1

    return stem, depthwise, pointwise, head


def make_inputs():
    inputs = []

    vertical = np.zeros((INPUT_H, INPUT_W, 1), dtype=np.uint8)
    vertical[1:9, 4, 0] = 8
    inputs.append(vertical)

    horizontal = np.zeros((INPUT_H, INPUT_W, 1), dtype=np.uint8)
    horizontal[4, 1:9, 0] = 8
    inputs.append(horizontal)

    diagonal = np.zeros((INPUT_H, INPUT_W, 1), dtype=np.uint8)
    for index in range(1, 9):
        diagonal[index, index, 0] = 8
    inputs.append(diagonal)

    return inputs


def c_array(name: str, values: np.ndarray, dimensions: str = "") -> str:
    flat = values.astype(np.uint8).flatten()
    rows = []
    for index in range(0, flat.size, 16):
        rows.append("    " + ", ".join(str(int(v)) for v in flat[index : index + 16]))
    body = ",\n".join(rows)
    return f"const uint8_t {name}{dimensions} = {{\n{body}\n}};\n"


def c_array_2d(name: str, values: np.ndarray, row_size: str) -> str:
    matrix = values.astype(np.uint8).reshape(values.shape[0], -1)
    samples = []
    for sample in matrix:
        rows = []
        for index in range(0, sample.size, 16):
            rows.append(
                "        "
                + ", ".join(str(int(v)) for v in sample[index : index + 16])
            )
        samples.append("    {\n" + ",\n".join(rows) + "\n    }")
    body = ",\n".join(samples)
    return (
        f"const uint8_t {name}[MODEL_SAMPLE_COUNT][{row_size}] = {{\n"
        f"{body}\n}};\n"
    )


def main() -> None:
    stem_weights, dw_weights, pw_weights, head_weights = make_model()
    inputs = make_inputs()

    stem_outputs = []
    dw_outputs = []
    pw_outputs = []
    head_outputs = []
    predictions = []

    for input_tensor in inputs:
        stem = conv_valid(input_tensor, stem_weights, shift=0)
        depthwise = conv_valid(stem, dw_weights, shift=2, depthwise=True)
        pointwise = conv_valid(depthwise, pw_weights, shift=0)
        head = conv_valid(pointwise, head_weights, shift=1)

        stem_outputs.append(stem)
        dw_outputs.append(depthwise)
        pw_outputs.append(pointwise)
        head_outputs.append(head)
        predictions.append(int(np.argmax(head.sum(axis=(0, 1)))))

    packed_stem = pack_neureka_weights(stem_weights)
    packed_dw = pack_neureka_weights(dw_weights, depthwise=True)
    packed_pw = pack_neureka_weights(pw_weights)
    packed_head = pack_neureka_weights(head_weights)

    expected_sizes = {
        "stem": 768,
        "dw": 256,
        "pw": 192,
        "head": 96,
    }
    actual_sizes = {
        "stem": packed_stem.size,
        "dw": packed_dw.size,
        "pw": packed_pw.size,
        "head": packed_head.size,
    }
    if actual_sizes != expected_sizes:
        raise RuntimeError(f"unexpected packed sizes: {actual_sizes}")
    if predictions != [0, 1, 2]:
        raise RuntimeError(f"CPU model predictions are incorrect: {predictions}")

    source = [
        '#include "model_data.h"\n',
        'const char *const model_labels[MODEL_CLASS_COUNT] = {\n'
        '    "vertical", "horizontal", "diagonal"\n};\n',
        c_array("model_expected_class", np.array(predictions, dtype=np.uint8),
                "[MODEL_SAMPLE_COUNT]"),
        c_array_2d("model_inputs", np.stack(inputs), "MODEL_INPUT_SIZE"),
        c_array_2d(
            "model_expected_stem", np.stack(stem_outputs), "MODEL_STEM_SIZE"
        ),
        c_array_2d("model_expected_dw", np.stack(dw_outputs), "MODEL_DW_SIZE"),
        c_array_2d("model_expected_pw", np.stack(pw_outputs), "MODEL_PW_SIZE"),
        c_array_2d(
            "model_expected_head", np.stack(head_outputs), "MODEL_HEAD_SIZE"
        ),
        c_array("model_stem_weights", packed_stem,
                "[MODEL_STEM_WEIGHTS_SIZE]"),
        c_array("model_dw_weights", packed_dw,
                "[MODEL_DW_WEIGHTS_SIZE]"),
        c_array("model_pw_weights", packed_pw,
                "[MODEL_PW_WEIGHTS_SIZE]"),
        c_array("model_head_weights", packed_head,
                "[MODEL_HEAD_WEIGHTS_SIZE]"),
    ]

    OUTPUT.write_text("\n".join(source), encoding="ascii")
    print(f"Generated {OUTPUT}")
    for index, prediction in enumerate(predictions):
        scores = np.asarray(head_outputs[index]).sum(axis=(0, 1))
        print(f"sample {index}: scores={scores.tolist()} class={prediction}")


if __name__ == "__main__":
    main()

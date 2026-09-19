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
    # Por cada O es un kernel, por cada I es un canal del kernel de tamaño HxW. Es un arreglo NumPy de 4 dimensiones (OxIxHxW)
    # input_hwc contiene la entrada en formato H (alto) x W (ancho) x C (canales). Es un arreglo
    # NumPy de 3 dimensiones (HxWxC)
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
            # extrae la ventana. ejemplo: si kernel_h=3 y kernel_w=3, entonces window será un bloque de 3x3 de la entrada 
            # en la posición (h_out, w_out).
            # windows es un arreglo de dimensiones (kernel_h, kernel_w, input_c).
            if depthwise: # Para convolución depthwise, cada canal de salida se calcula usando solo su correspondiente canal de entrada y su kernel.
                for channel in range(output_c):
                    output[h_out, w_out, channel] = np.sum(
                        window[:, :, channel] * weights_oihw[channel, 0]
                    )
            else: # Para convolución estándar, cada canal de salida se calcula usando todos los canales de entrada y sus correspondientes kernels.
                for channel in range(output_c):
                    kernel_hwc = weights_oihw[channel].transpose(1, 2, 0)
                    output[h_out, w_out, channel] = np.sum(window * kernel_hwc)

    output = np.maximum(output, 0) # Aplica ReLU, estableciendo todos los valores negativos a cero.
    output = output >> shift # Aplica el shift a la derecha para la requantización. Esto es equivalente a dividir por 2^shift.
    return np.clip(output, 0, 255).astype(np.uint8)


def pack_neureka_weights(
    signed_weights: np.ndarray, depthwise: bool = False
) -> np.ndarray:
    # Esta función toma pesos int8 normales de una convolución y los convierte al formato que espera N-EUREKA, que es un formato de 256 bits por plano de bits. Esto implica reorganizar los pesos y empaquetarlos en bits.
    """Pack OIHW int8 weights into N-EUREKA's 256-bit bit-plane layout."""
    if signed_weights.dtype != np.int8:
        # valida que los pesos sean int8 (entre -128 y 127)
        raise TypeError("weights must use int8 storage")

    # N-EUREKA applies -128 as a layer-wise offset during execution.
    weights = (signed_weights.astype(np.int16) + 128).astype(np.uint8) # Convierte los pesos de int8 a uint8 sumando 128, para que el rango pase de [-128, 127] a [0, 255].
    if depthwise:
        # Para convoluciones depthwise, los pesos se reorganizan para que cada canal de salida tenga su propio kernel independiente. Esto significa que la forma de los pesos cambia de (output_c, input_c, kernel_h, kernel_w) a (input_c, output_c, kernel_h, kernel_w), donde cada canal de salida tiene su propio conjunto de pesos.
        # Esto calza mejor con una depthwise, porque una depthwise no mezcla canales. Cada canale tiene su propio kernel.
        # Por ejemplo:
        # weights.shape == (1, 3, 3, 3) en el formato OIKW)
        # Significa:
            # 1 canal de salida
            # 3 canales de entrada
            # kernel 3x3

                    # weights[0, 0, :, :]  pesos para input channel 0
                    # weights[0, 1, :, :]  pesos para input channel 1
                    # weights[0, 2, :, :]  pesos para input channel 2

        # Después de weights.transpose(1, 0, 2, 3) queda:
        # weights.shape == (3, 1, 3, 3)
            # weights[0, 0, :, :]  kernel del canal 0
            # weights[1, 0, :, :]  kernel del canal 1
            # weights[2, 0, :, :]  kernel del canal 2

        # Para depthwise, cada canal tiene su propio filtro. Esta línea reordena los ejes para acomodar los pesos al formato que espera N-EUREKA en modo depthwise.

        weights = weights.transpose(1, 0, 2, 3)

    output_c, input_c, kernel_h, kernel_w = weights.shape # Extrae las dimensiones de los pesos
    # input subtile es el número de canales de entrada que caben en un bloque de 256 bits, es decir, es el bloque de canales usado para empaquetar pesos.
    # 256 porque los streamers internos soportan 256 bits de ancho. Cada canal de entrada necesita kernel_h * kernel_w pesos, y cada peso es 8 bits. Entonces, el número máximo de canales de entrada que caben en un bloque de 256 bits depende del tamaño del kernel.
    input_subtile = 28 if kernel_h == 3 else 32 # Para una convolución 3x3, por cada canal de entrada necesitas 3x3=9 pesos, si quisieras meter 32 canales de entrada en un solo bloque: 9 pesos x 32 canales = 288 bits, que no cabe en 256 bits. Por eso se usa 28 canales de entrada (9 pesos x 28 canales = 252 bits, lo cual sí cabe en una palabra de 256 bits y sobran 4 bits para padding). Para una convolución 1x1, cada canal de entrada solo necesita 1 peso, entonces puedes meter hasta 32 canales de entrada en un bloque de 256 bits.
    # Por lo tanto input subtile es el número máximo de canales de entrada que se pueden empaquetar en un bloque de 256 bits.
    input_major = (input_c + input_subtile - 1) // input_subtile # Input major es el número de bloques completos de canales de entrada que se necesitan para cubrir todos los canales de entrada. Se calcula dividiendo el número total de canales de entrada entre el tamaño del subtile, redondeando hacia arriba. Esto asegura que incluso si no hay un número exacto de subtile, se reserve espacio suficiente para todos los canales de entrada.
    padded_input_c = input_major * input_subtile # Padded input c es el número total de canales de entrada después de rellenar con padding para que sea un múltiplo de input_subtile. Esto es necesario porque N-EUREKA espera que los pesos estén organizados en bloques completos de 256 bits, y si el número de canales de entrada no es un múltiplo de input_subtile, se agregan canales ficticios (relleno) para completar el bloque.
    # Estas dos líneas calculan cuántos bloques completos de canales necesita N-EUREKA y cuántos canales habrá después de rellenar con padding.
    # Por ejemplo, si input_c = 29:
        
        # input_subtile = 28 (para kernel 3x3)
        # input_major = 2
        
        # padded_input_c = 2 * 28
        # padded_input_c = 56

        # Entonces:
            # canales reales: 29
            # canales después de padding: 56
            # canales falsos agregados: 27
    
    # Esto se hace porque el hardware espera bloques completos de 256 bits.


    if padded_input_c != input_c:
        # Agrega padding (ceros) a los pesos para que el número de canales de entrada sea un múltiplo de input_subtile.
        weights = np.pad(
            weights,
            ((0, 0), (0, padded_input_c - input_c), (0, 0), (0, 0)),
        ) # Esos canales de entrada extra no representan datos reales; sólo están para cumplir el layout del hardware.

    """Partimos con pesos en la forma normal:
    weights.shape == (output_c, input_c, kernel_h, kernel_w)
    Después del padding, input_c ya es múltiplo de input_subtile, y podemos reorganizar los pesos en bloques de 256 bits.
    La línea:
        weights = weights.reshape(output_c, input_major, input_subtile, kernel_h * kernel_w, 1)
    los reorganiza como:
        output_c
        input_major
        input_subtile
        kernel_positions (kernel_h * kernel_w)
        1 (placeholder para facilitar la transposición posterior)
        
    Se separa input_c en input_major e input_subtile porque N-EUREKA procesa canales en bloques.
    Ejemplo:
        input_c real = 33
        kernel = 3x3
        input_subtile = 28

        Después del padding, padded_input_c = 56
        Eso puede verse como 2 bloques de 28 canales de entrada (input_major = 2, input_subtile = 28)
        Por lo que en vez de tener (output_c, 56, 3, 3), tenemos (output_c, 2, 28, 9, 1)

        Así queda explícito:
            para cada canal de salida
            para cada bloque de canales de entrada
                toma 28 canales
                toma las 9 posiciones del kernel (o 1 posición si es 1x1)
                cada peso ocupa 1 byte antes de separar bits
        
        Por qué kernel_h * kernel_w? La función ya no necesita distinguir fila/columna del kernel, sólo necesita empaquetar todos los pesos de un canal de entrada en un bloque de bits. Por eso se aplanan las dimensiones del kernel. Sólo necesita una lista de posiciones:
            k0, k1, k2, ..., k8
            por eso aplana:
                (kernel_h, kernel_w)
            a: 
                (kernel_h * kernel_w)
        
        Por qué el 1 final? Ese 1 representa que cada peso todavía está guardado como 1 byte uint8, Justo después se ahce:
            np.unpackbits(weights, axis=-1, count=WEIGHT_BITS, bitorder="little")
        Eso convierte ese último eje: 
            1 byte
        En:
            8 bits
        
        Entonces:
            antes de unpackbits:
            (output_c, input_major, input_subtile, kernel_positions, 1)

            después de unpackbits:
            (output_c, input_major, input_subtile, kernel_positions, 8)


            
        En resumen, esa forma se elige porque calza con el empaquetado que viene después:
            separar canales en bloques
            aplanar posiciones del kernel
            convertir cada peso-byte en 8 bits
            reordenar esos bits en bloques de 256 bits para N-EUREKA
    """
    weights = weights.reshape(
        output_c, input_major, input_subtile, kernel_h * kernel_w, 1
    ) # Reorganiza los pesos en una forma que facilita el empaquetado en bits. La nueva forma es:
    # (output_c, input_major, input_subtile, kernel_h * kernel_w, 1)
    # Esto significa que para cada canal de salida, tenemos un número de bloques de entrada (input_major), cada uno con un número de canales de entrada (input_subtile), y cada canal de entrada tiene un kernel de tamaño kernel_h * kernel_w. El último 1 es un placeholder para facilitar la transposición posterior.
    weights = np.unpackbits(
        weights, axis=-1, count=WEIGHT_BITS, bitorder="little"
    ) # Ahora, weights tiene la forma (output_c, input_major, input_subtile, kernel_positions, bits u 8)
    weights = weights.transpose(0, 1, 4, 3, 2) # Transpone los ejes quedando así: (output_c, input_major, bits, kernel_positions, input_subtile)

    """
    Esto se hace porque N-EUREKA espera los pesos agrupados por bit-plane. Antes del transpose, los datos están organizados así:

        eje 0 = output_c
        eje 1 = input_major
        eje 2 = input_subtile
        eje 3 = kernel_positions
        eje 4 = bits

            para cada canal de entrada
                para cada posición del kernel
                    tengo sus 8 bits

    Después del transpose, quedan así:

        eje 0 = output_c
        eje 1 = input_major
        eje 4 = bits
        eje 3 = kernel_positions
        eje 2 = input_subtile

            para cada bit
                para cada posición del kernel
                    para cada canal de entrada
                        tengo ese bit del peso

    Esto permite construir bloques como:
        bit 0 de todos los pesos del bloque
        bit 1 de todos los pesos del bloque
        ...
        bit 7 de todos los pesos del bloque                  

    lo cual se conoce como bit-plane layout, que es lo que N-EUREKA espera.     
    """

    """
    Entonces, los pesos ya quedaron con forma conceptual:
        (output_c, input_major, bit, kernel_position, input_subtile)
    es decir:
        para cada canal de salida
            para cada bloque de canales de entrada
                para cada bit del peso
                    para cada posición del kernel
                        para cada canal de entrada en el bloque
                            tengo ese bit del peso
    
    Ahora falta convertir eso al formato final que N-EUREKA espera, que es un arreglo de 256 bits por bloque de canales de entrada. Esto se hace aplanando los ejes y empaquetando los bits en bytes.

    Para kernel 3x3:
        kernel_h * kernel_w = 9
    y para 3x3 usamos:
        input_subtile = 28
    Entonces:
        9 posiciones del kernel * 28 canales de entrada = 252 bits
    La línea:
        weights = weights.reshape(-1, kernel_h * kernel_w * input_subtile)
    agrupa cada plano en filas de 252 bits. Cada fila representa un bloque de pesos para:
        1 output channel
        1 input_major
        1 bit-plane
    Pero N-EUREKA espera 256 bits, así que se hace padding con ceros para completar a 256 bits:
        weights = np.pad(weights, ((0, 0), (0, 256 - weights.shape[-1])))


    Para kernel 1x1, el kernel tiene una sola posición:
        kernel_h * kernel_w = 1
    y se usa:
        input_subtile = 32
    Después del transpose anterior tienes algo ocmo:
        (output_c, input_major, 8, 1, 32)
    La línea:
        weights = weights.reshape(output_c, input_major, WEIGHT_BITS, 1, input_subtile // 4, 4)
    convierte los 32 canales en:
        input_subtile // 4 = 8
        4 canales por grupo
    Es decir, 8 grupos de 4 canales. Entonces queda:
        (output_c, input_major, 8 bits, 1 kernel_pos, 8 grupos, 4 canales)
    Luego:
        weights = weights.transpose(0, 1, 3, 4, 2, 5)
    lo reordena como:
        (output_c, input_major, 1 kernel_pos, 8 grupos_de_4_canales, 8 bits, 4_canales)
    La idea es acomodar los bits en el orden específico que N-EUREKA espera para 1x1. El 1x1 tiene un layout distinto al 3x3 porque 
    no tiene 9 posiciones espaciales; usa una organización más compacta de canales/bits. Finalmente:
        weights = weights.reshape(output_c * input_major, 256)
    deja cada bloque de 256 bits listo para N-EUREKA.
    Por lo que en este caso también cada fila final tiene 256 bits.
    """
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
    """
    Esta es la conversión final de bits a bytes. Hasta aquí, weights es una matriz de bits (0s y 1s) con forma (num_planes, 256). Cada fila representa un plano de bits de 256 bits.
    Esta línea agrupa los bits de a 8:
        [bit0 bit1 bit2 bit3 bit4 bit5 bit6 bit7]
    Cada grupo de 8 bits será un byte.
    Luego:
        return np.packbits(weights, axis=-1, bitorder="little").flatten()
    convierte cada grupo de 8 bits en un uint8 y flatten deja todo como un arreglo lineal:
    [byte0, byte1, byte2, byte3, ...]
    Ese arreglo final es lo que se guarda en model_data.c
    """
    """OJO:
        1x1 empaqueta todos los bits juntos:
        porque 32 pesos completos de 8 bits llenan 256 bits justo.

        3x3 empaqueta un bit-plane por bloque:
        porque hay 252 pesos; si metiera pesos completos necesitaría 2016 bits,
        así que cada bloque de 256 lleva solo un bit de cada peso.
    """
    return np.packbits(weights, axis=-1, bitorder="little").flatten()


def make_model():
    # Three correlation filters: vertical, horizontal and main diagonal.
    # Esta función define el "modelo neuronal" completo, pero con pesos inventados a mano, no se entrena nada.
    stem = np.array(
        [
            [[[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]],
            [[[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]],
            [[[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]],
        ],
        dtype=np.int8,
    ) # Stem es una convolucion 3x3 normal. Tiene la forma (output_c, input_c, kernel_h, kernel_w) = (3, 1, 3, 3). Cada canal de salida es un filtro que detecta una orientación específica: vertical, horizontal y diagonal. Los valores de los pesos son enteros pequeños para simplificar la demostración.

    # Depthwise spatial aggregation, one independent kernel per channel.
    depthwise = np.ones((3, 1, 3, 3), dtype=np.int8) # Tiene forma (output_c, input_c, kernel_h, kernel_w) = (3, 1, 3, 3). Cada canal de salida tiene su propio kernel de 3x3 que es todo unos. Esto significa que cada canal de salida simplemente suma los valores en una ventana de 3x3 de su correspondiente canal de entrada. Es una operación de agregación espacial simple.
    # Cada kernerl mira un solo canal de entrada, por eso input_c=1. Cada canal de salida tiene su propio kernel independiente, por eso output_c=3.
    # canal 0 de entrada -> kernel depthwise 0 -> canal 0 de salida
    # canal 1 de entrada -> kernel depthwise 1 -> canal 1 de salida
    # canal 2 de entrada -> kernel depthwise 2 -> canal 2 de salida

    # MobileNet-style pointwise expansion from 3 to 6 channels.
    pointwise = np.zeros((6, 3, 1, 1), dtype=np.int8) # llena de ceros un tensor de pesos para una convolución 1x1 que expande de 3 canales a 6 canales. Tiene forma (output_c, input_c, kernel_h, kernel_w) = (6, 3, 1, 1). Inicialmente todos los pesos son cero.
    for channel in range(3):
        pointwise[2 * channel, channel, 0, 0] = 1
        pointwise[2 * channel + 1, channel, 0, 0] = 1
    # Cada canal de entrada se expande a dos canales de salida.
    # entrada:  [in0, in1, in2]
    # salida:   [in0, in0, in1, in1, in2, in2]

    # Pointwise projection from 6 channels to the 3 class maps.
    head = np.zeros((3, 6, 1, 1), dtype=np.int8) # Conv 1x1 de salida/clases. Tiene forma (output_c, input_c, kernel_h, kernel_w) = (3, 6, 1, 1). Inicialmente todos los pesos son cero. Cada canal de salida corresponde a una clase: vertical, horizontal o diagonal.
    # Esta capa toma los 6 canales de entrada y los combina en 3 canales de salida, uno por clase.
    for channel in range(3):
        head[channel, 2 * channel : 2 * channel + 2, 0, 0] = 1
        # El loop hace:

            # clase 0 = canal 0 + canal 1
            # clase 1 = canal 2 + canal 3
            # clase 2 = canal 4 + canal 5
    """
        Entonces el modelo completo hace esto:
            imagen 10x10x1
            ↓ stem 3x3
            mapas vertical/horizontal/diagonal
            ↓ depthwise 3x3
            agrega evidencia espacial por canal
            ↓ pointwise 1x1
            expande canales 3 -> 6
            ↓ head 1x1
            vuelve a 3 mapas de clase
            ↓ suma global
            scores vertical/horizontal/diagonal

    """
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

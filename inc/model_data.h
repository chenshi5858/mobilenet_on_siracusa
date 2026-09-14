#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

#define MODEL_SAMPLE_COUNT 3
#define MODEL_CLASS_COUNT 3

// Model input corresponde a una imagen de 10x10 en escala de grises (1 canal)
#define MODEL_INPUT_H 10
#define MODEL_INPUT_W 10
#define MODEL_INPUT_C 1
#define MODEL_INPUT_SIZE (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)

// Model stem corresponde a la primera capa convolucional de 8x8 con 3 canales
#define MODEL_STEM_H 8
#define MODEL_STEM_W 8
#define MODEL_STEM_C 3
#define MODEL_STEM_SIZE (MODEL_STEM_H * MODEL_STEM_W * MODEL_STEM_C)

// Convolución depthwise de 6x6 con 3 canales (hace una convolución espacial independientemente sobre cada canal, a 
// diferencia de una convolución normal, no mezclamos canales)
#define MODEL_DW_H 6
#define MODEL_DW_W 6
#define MODEL_DW_C 3
#define MODEL_DW_SIZE (MODEL_DW_H * MODEL_DW_W * MODEL_DW_C)

// Convolución pointwise de 6x6 con 6 canales (mezcla los canales de la convolución depthwise). Su propósito
// principal es mezclar información entre canales, y es una operación de convolución 1x1}.
#define MODEL_PW_H 6
#define MODEL_PW_W 6
#define MODEL_PW_C 6
#define MODEL_PW_SIZE (MODEL_PW_H * MODEL_PW_W * MODEL_PW_C)

// Capa final de la red, que reduce la salida a 3 clases (3 canales)
#define MODEL_HEAD_H 6
#define MODEL_HEAD_W 6
#define MODEL_HEAD_C 3
#define MODEL_HEAD_SIZE (MODEL_HEAD_H * MODEL_HEAD_W * MODEL_HEAD_C)

#define MODEL_STEM_WEIGHTS_SIZE 768
#define MODEL_DW_WEIGHTS_SIZE 256
#define MODEL_PW_WEIGHTS_SIZE 192
#define MODEL_HEAD_WEIGHTS_SIZE 96

// Lo que hace "extern" es declarar que estas variables están definidas en otro archivo, y que se pueden usar en este archivo.
extern const char *const model_labels[MODEL_CLASS_COUNT];
extern const uint8_t model_expected_class[MODEL_SAMPLE_COUNT];
extern const uint8_t model_inputs[MODEL_SAMPLE_COUNT][MODEL_INPUT_SIZE];

extern const uint8_t
    model_expected_stem[MODEL_SAMPLE_COUNT][MODEL_STEM_SIZE];
extern const uint8_t model_expected_dw[MODEL_SAMPLE_COUNT][MODEL_DW_SIZE];
extern const uint8_t model_expected_pw[MODEL_SAMPLE_COUNT][MODEL_PW_SIZE];
extern const uint8_t
    model_expected_head[MODEL_SAMPLE_COUNT][MODEL_HEAD_SIZE];

extern const uint8_t model_stem_weights[MODEL_STEM_WEIGHTS_SIZE];
extern const uint8_t model_dw_weights[MODEL_DW_WEIGHTS_SIZE];
extern const uint8_t model_pw_weights[MODEL_PW_WEIGHTS_SIZE];
extern const uint8_t model_head_weights[MODEL_HEAD_WEIGHTS_SIZE];

#endif


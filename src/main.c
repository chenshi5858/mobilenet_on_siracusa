#include "model_data.h" // model_data.h es el archivo de cabecera que contiene las definiciones de los datos del modelo, como los pesos y las entradas esperadas para cada capa de la red neuronal.
#include "pmsis.h" // pmpsis es la librería de PULP que nos permite interactuar con el cluster y la memoria L1/L2
#include "pulp_nnx_hal.h" // pulp_nnx_hal es la librería de PULP que nos permite interactuar con el acelerador N-EUREKA

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef enum {
  LAYER_CONV_3X3,
  LAYER_DEPTHWISE_3X3,
  LAYER_CONV_1X1,
} layer_operation_t; // Enumeración para representar los tipos de operaciones de capa que se pueden realizar en la red neuronal. Cada valor de la enumeración corresponde a un tipo específico de operación de convolución que se puede aplicar a los datos de entrada.

typedef struct {
  const char *name;
  layer_operation_t operation;
  uint16_t input_h;
  uint16_t input_w;
  uint16_t input_c;
  uint16_t output_h;
  uint16_t output_w;
  uint16_t output_c;
  uint8_t shift; // shift es un valor que se utiliza para ajustar los valores de salida después de la operación de convolución. Se utiliza para escalar los valores de salida a un rango adecuado para la siguiente capa de la red neuronal. El valor de shift se aplica a los valores de salida después de la operación de convolución, y puede ser positivo o negativo dependiendo del rango deseado.
} layer_config_t; // Estructura que representa la configuración de una capa de la red neuronal. Contiene información sobre el nombre de la capa, el tipo de operación que realiza, las dimensiones de entrada y salida, y un valor de desplazamiento (shift) que se utiliza para ajustar los valores de salida después de la operación de convolución.

static const layer_config_t layers[] = {
    {.name = "stem 3x3",
     .operation = LAYER_CONV_3X3,
     .input_h = MODEL_INPUT_H,
     .input_w = MODEL_INPUT_W,
     .input_c = MODEL_INPUT_C,
     .output_h = MODEL_STEM_H,
     .output_w = MODEL_STEM_W,
     .output_c = MODEL_STEM_C,
     .shift = 0},
    {.name = "depthwise 3x3",
     .operation = LAYER_DEPTHWISE_3X3,
     .input_h = MODEL_STEM_H,
     .input_w = MODEL_STEM_W,
     .input_c = MODEL_STEM_C,
     .output_h = MODEL_DW_H,
     .output_w = MODEL_DW_W,
     .output_c = MODEL_DW_C,
     .shift = 2},
    {.name = "pointwise expand 1x1",
     .operation = LAYER_CONV_1X1,
     .input_h = MODEL_DW_H,
     .input_w = MODEL_DW_W,
     .input_c = MODEL_DW_C,
     .output_h = MODEL_PW_H,
     .output_w = MODEL_PW_W,
     .output_c = MODEL_PW_C,
     .shift = 0},
    {.name = "pointwise head 1x1",
     .operation = LAYER_CONV_1X1,
     .input_h = MODEL_PW_H,
     .input_w = MODEL_PW_W,
     .input_c = MODEL_PW_C,
     .output_h = MODEL_HEAD_H,
     .output_w = MODEL_HEAD_W,
     .output_c = MODEL_HEAD_C,
     .shift = 1},
};


/* Estas líneas reservan buffers en memoria L1 para que el programa y N-EUREKA trabajen con datos rápidos
 y cercanos al clúster. PI_L1 lee dice al SDK/Linker que estos buffers deben ser asignados en la memoria L1. 
 static significa que estas variables son visibles solo dentro de este archivo. 
 uint8_t es un tipo de dato que representa un número entero sin signo de 8 bits. (0 al 255)*/
PI_L1 static uint8_t input_l1[MODEL_INPUT_SIZE]; // buffer para la imagen de entrada actual
PI_L1 static uint8_t stem_l1[MODEL_STEM_SIZE]; // buffer para la salida de la capa stem
PI_L1 static uint8_t dw_l1[MODEL_DW_SIZE]; // buffer para la salida de la capa depthwise
PI_L1 static uint8_t pw_l1[MODEL_PW_SIZE]; // buffer para la salida de la capa pointwise
PI_L1 static uint8_t head_l1[MODEL_HEAD_SIZE]; // buffer para la salida de la capa head

PI_L1 static uint8_t stem_weights_l1[MODEL_STEM_WEIGHTS_SIZE]; // buffer para los pesos de la capa stem
PI_L1 static uint8_t dw_weights_l1[MODEL_DW_WEIGHTS_SIZE]; // buffer para los pesos de la capa depthwise
PI_L1 static uint8_t pw_weights_l1[MODEL_PW_WEIGHTS_SIZE]; // buffer para los pesos de la capa pointwise
PI_L1 static uint8_t head_weights_l1[MODEL_HEAD_WEIGHTS_SIZE]; // buffer para los pesos de la capa head

PI_L1 static uint32_t scale_l1[MODEL_PW_C]; // buffer para los factores de escala de la capa pointwise
PI_L1 static int32_t bias_l1[MODEL_PW_C]; // buffer para los sesgos de la capa pointwise

static int errors;

static int compare_tensor(const char *name, const uint8_t *actual,
                          const uint8_t *expected, uint32_t size) {
  /*
  Esta función compara dos tensores (arrays de datos) y cuenta cuántos elementos no coinciden.
  */
  int mismatches = 0;
  for (uint32_t index = 0; index < size; ++index) {
    if (actual[index] != expected[index]) {
      if (mismatches < 5) {
        printf("  %s mismatch[%lu]: got=%u expected=%u\n", name,
               (unsigned long)index,
               actual[index], expected[index]);
      }
      ++mismatches;
    }
  }
  return mismatches;
}

static int configure_task(nnx_task_t *task, const layer_config_t *layer,
                          uint8_t *input, uint8_t *output,
                          uint8_t *packed_weights) {
  /*
  Esta función configura una tarea de N-EUREKA para ejecutar una capa de la red neuronal.
  Toma como parámetros un puntero a la tarea, un puntero a la configuración de la capa, un puntero a los datos 
  de entrada, un puntero a los datos de salida y un puntero a los pesos empaquetados. 
  Inicializa la tarea, configura el padding de la entrada, define las descripciones de las 
  características de entrada y salida, define la descripción de los pesos, y luego llama a la función de 
  convolución correspondiente según el tipo de operación de la capa. Finalmente, configura la normalización 
  y cuantización de la salida, y establece los punteros a los datos de entrada, salida y pesos en la tarea.
  */
  nnx_task_init(task); // Inicializa la tarea de N-EUREKA, preparando la estructura de datos para almacenar la configuración de la tarea y asegurando que todos los campos estén en un estado conocido antes de configurarla.
  nnx_pad_input(&task->cfg, 0, 0, 0, 0, 0); // Configura el padding de la entrada de la tarea de N-EUREKA. El padding es un proceso que agrega píxeles adicionales alrededor de los bordes de la imagen de entrada para evitar que la información se pierda durante la operación de convolución. En este caso, se está configurando un padding de 0 píxeles en todos los lados (arriba, abajo, izquierda, derecha) y un padding de 0 píxeles en el canal de profundidad. Esto significa que no se agregará ningún padding a la imagen de entrada.

  nnx_feature_t input_desc = {.data = input, // input es un puntero a los datos de entrada de la capa, que se pasan a N-EUREKA para que pueda procesarlos durante la operación de convolución.
                              .height = layer->input_h,
                              .width = layer->input_w,
                              .depth = layer->input_c,
                              .bitwidth = featureBitwidth8Bit}; // featureBitwidth8Bit es un valor que indica que los datos de entrada son de 8 bits de ancho. Esto significa que cada elemento de los datos de entrada se representa con un número entero sin signo de 8 bits, lo que permite representar valores en el rango de 0 a 255. Esta información es importante para N-EUREKA, ya que le permite interpretar correctamente los datos de entrada durante la operación de convolución.
/* Define la descripción de las características de entrada para la tarea de N-EUREKA. Esta estructura contiene información sobre los datos de entrada, 
incluyendo un puntero a los datos reales, las dimensiones de la entrada (altura, ancho y profundidad) y el ancho de bits de los datos 
(en este caso, 8 bits). Esta descripción se utiliza para informar a N-EUREKA sobre cómo interpretar los datos de entrada durante la operación de 
convolución*/
  nnx_feature_t output_desc = {.data = output,
                               .height = layer->output_h,
                               .width = layer->output_w,
                               .depth = layer->output_c,
                               .bitwidth = featureBitwidth8Bit};

  const uint16_t kernel_size = layer->operation == LAYER_CONV_1X1 ? 1 : 3;
  nnx_weights_t weights_desc = {
      .data = packed_weights, // packed_weights es un puntero a los pesos de la capa.
      .height = kernel_size,
      .width = kernel_size,
      .depth = layer->input_c,
      .n_weights = layer->output_c, // n_weights es el número de filtros (o kernels) que se aplicarán a la entrada durante la operación de convolución. Cada filtro produce un mapa de características (feature map) en la salida, y el número total de filtros determina cuántos mapas de características se generarán. En este caso, n_weights se establece en layer->output_c, que es el número de canales de salida de la capa. Esto significa que habrá un filtro para cada canal de salida, y cada filtro se aplicará a todos los canales de entrada para generar los mapas de características correspondientes.
      .bitwidth = 8,
      .offset_factor = -128, // offset_factor es un valor que se utiliza para ajustar los valores de los pesos durante la operación de convolución. En este caso, se establece en -128, lo que significa que se restará 128 a cada valor de peso antes de realizar la operación de convolución. Esto es útil cuando los pesos se almacenan en un formato de 8 bits sin signo (0 a 255), pero se desea que tengan un rango centrado alrededor de cero (-128 a 127). Al restar 128, los valores de peso se ajustan para que puedan representar tanto valores positivos como negativos durante la operación de convolución.
      .offset_mode = weightOffsetModeLayerWise, // offset_mode es un valor que indica cómo se aplicará el offset_factor a los pesos durante la operación de convolución. En este caso, se establece en weightOffsetModeLayerWise, lo que significa que el offset_factor se aplicará de manera uniforme a todos los pesos de la capa. Esto asegura que todos los pesos se ajusten de la misma manera, manteniendo la coherencia en la operación de convolución y evitando sesgos no deseados en los resultados.
  };

  nnx_error_code status;
  if (layer->operation == LAYER_CONV_1X1) {
    status = nnx_conv_1x1(&task->cfg, weights_desc, input_desc, output_desc); // Esta función está definida en pulp_nnx_hal.c y configura la tarea de N-EUREKA para realizar una operación de convolución 1x1. Toma como parámetros un puntero a la configuración de la tarea, una descripción de los pesos, una descripción de las características de entrada y una descripción de las características de salida. La función devuelve un código de error que indica si la configuración fue exitosa o si hubo algún problema durante el proceso.
  } else if (layer->operation == LAYER_DEPTHWISE_3X3) {
    status =
        nnx_conv_3x3_dw(&task->cfg, weights_desc, input_desc, output_desc);
  } else {
    status = nnx_conv_3x3(&task->cfg, weights_desc, input_desc, output_desc); // es la convolución 3x3 normal.
  }
  if (status != 0) {
    return status;
  }

  nnx_norm_t norm = {.mode = normMode32Bit, .flag_bias = 0, .flag_shift = 0}; // norm es una estructura que define cómo se normalizarán los valores de salida después de la operación de convolución. En este caso, se establece en normMode32Bit, lo que significa que los valores de salida se normalizarán a un rango de 32 bits. Los campos flag_bias y flag_shift se establecen en 0, lo que indica que no se aplicará ningún sesgo ni desplazamiento adicional durante la normalización. Esta configuración asegura que los valores de salida estén en un rango adecuado para su posterior procesamiento en la red neuronal.
  nnx_quant_t quant = {.shift_amount = layer->shift,
                       .mode = quantMode8Bit,
                       .function = quantFunctionRelu,
                       .flag_rounding = 0}; // quant es una estructura que define cómo se cuantizarán los valores de salida después de la normalización. En este caso, se establece en quantMode8Bit, lo que significa que los valores de salida se cuantizarán a un rango de 8 bits. El campo shift_amount se establece en layer->shift, lo que permite ajustar los valores de salida según el valor de desplazamiento especificado en la configuración de la capa. El campo function se establece en quantFunctionRelu, lo que indica que se aplicará la función de activación ReLU (Rectified Linear Unit) a los valores de salida durante la cuantización. El campo flag_rounding se establece en 0, lo que significa que no se aplicará ningún redondeo adicional durante la cuantización. Esta configuración asegura que los valores de salida estén en un rango adecuado para su posterior procesamiento en la red neuronal y que se aplique la función de activación ReLU para introducir no linealidad en el modelo.
  if (nnx_norm_quant(&task->cfg, norm, quant) != 0) {
    return -1;
  }

  BIT_SET(task->cfg.conf0, NEUREKA_FLAG_USE_TCDM);
  task->infeat_ptr = (uint32_t)input;
  task->outfeat_ptr = (uint32_t)output;
  task->weights_ptr = (uint32_t)packed_weights;
  task->scale_ptr = (uint32_t)scale_l1;
  task->scale_shift_ptr = 0;
  task->scale_bias_ptr = (uint32_t)bias_l1;
  return 0;
}

static uint32_t execute_layer(const layer_config_t *layer, uint8_t *input,
                              uint8_t *output, uint8_t *packed_weights) {
  nnx_task_t task;
  if (configure_task(&task, layer, input, output, packed_weights) != 0) {
    printf("Could not configure %s\n", layer->name);
    ++errors;
    return 0;
  }

  nnx_soft_clear();
  nnx_acquire();
  pi_perf_reset();
  pi_perf_start();
  uint32_t start = pi_perf_read(PI_PERF_CYCLES);
  nnx_offload(&task);
  nnx_run_blocking();
  uint32_t elapsed = pi_perf_read(PI_PERF_CYCLES) - start;
  pi_perf_stop();
  return elapsed;
}

static void load_weights(void) {
  memcpy(stem_weights_l1, model_stem_weights, MODEL_STEM_WEIGHTS_SIZE);
  memcpy(dw_weights_l1, model_dw_weights, MODEL_DW_WEIGHTS_SIZE);
  memcpy(pw_weights_l1, model_pw_weights, MODEL_PW_WEIGHTS_SIZE);
  memcpy(head_weights_l1, model_head_weights, MODEL_HEAD_WEIGHTS_SIZE);
  for (int channel = 0; channel < MODEL_PW_C; ++channel) {
    scale_l1[channel] = 1;
    bias_l1[channel] = 0;
  }
}

static void classify(uint32_t sample) {
  uint32_t cycles[4];
  uint32_t scores[MODEL_CLASS_COUNT] = {0};

  memcpy(input_l1, model_inputs[sample], MODEL_INPUT_SIZE);
  memset(stem_l1, 0, MODEL_STEM_SIZE);
  memset(dw_l1, 0, MODEL_DW_SIZE);
  memset(pw_l1, 0, MODEL_PW_SIZE);
  memset(head_l1, 0, MODEL_HEAD_SIZE);

  cycles[0] =
      execute_layer(&layers[0], input_l1, stem_l1, stem_weights_l1);
  cycles[1] = execute_layer(&layers[1], stem_l1, dw_l1, dw_weights_l1);
  cycles[2] = execute_layer(&layers[2], dw_l1, pw_l1, pw_weights_l1);
  cycles[3] = execute_layer(&layers[3], pw_l1, head_l1, head_weights_l1);

  int sample_errors = 0;
  sample_errors += compare_tensor("stem", stem_l1,
                                  model_expected_stem[sample], MODEL_STEM_SIZE);
  sample_errors += compare_tensor("depthwise", dw_l1,
                                  model_expected_dw[sample], MODEL_DW_SIZE);
  sample_errors += compare_tensor("pointwise", pw_l1,
                                  model_expected_pw[sample], MODEL_PW_SIZE);
  sample_errors += compare_tensor("head", head_l1,
                                  model_expected_head[sample], MODEL_HEAD_SIZE);

  for (uint32_t pixel = 0; pixel < MODEL_HEAD_H * MODEL_HEAD_W; ++pixel) {
    for (uint32_t channel = 0; channel < MODEL_CLASS_COUNT; ++channel) {
      scores[channel] += head_l1[pixel * MODEL_CLASS_COUNT + channel];
    }
  }

  uint32_t prediction = 0;
  for (uint32_t channel = 1; channel < MODEL_CLASS_COUNT; ++channel) {
    if (scores[channel] > scores[prediction]) {
      prediction = channel;
    }
  }

  printf("\nSample %lu (%s)\n", (unsigned long)sample,
         model_labels[model_expected_class[sample]]);
  printf("  scores: vertical=%lu horizontal=%lu diagonal=%lu\n",
         (unsigned long)scores[0], (unsigned long)scores[1],
         (unsigned long)scores[2]);
  printf("  predicted: %s\n", model_labels[prediction]);
  for (uint32_t index = 0; index < 4; ++index) {
    printf("  %-22s %lu cycles\n", layers[index].name,
           (unsigned long)cycles[index]);
  }

  if (prediction != model_expected_class[sample]) {
    printf("  ERROR: wrong class\n");
    ++sample_errors;
  }
  if (sample_errors == 0) {
    printf("  layer-by-layer check: PASS\n");
  } else {
    printf("  layer-by-layer check: FAIL (%d mismatches)\n", sample_errors);
  }
  errors += sample_errors;
}

static void cluster_entry(void *arg) {
  (void)arg;
  NEUREKA_CG_ENABLE();
  NEUREKA_SETPRIORITY_NEUREKA();
  NEUREKA_RESET_MAXSTALL();
  NEUREKA_SET_MAXSTALL(8);
  pi_perf_conf(1 << PI_PERF_CYCLES);

  load_weights();
  for (uint32_t sample = 0; sample < MODEL_SAMPLE_COUNT; ++sample) {
    classify(sample);
  }

  NEUREKA_CG_DISABLE();
}

int main(void) {
  printf("Manual MiniMobileNet on Siracusa/N-EUREKA\n");
  printf("No Deeploy, no ONNX runtime, four manually scheduled layers.\n");

  struct pi_device cluster;
  struct pi_cluster_conf conf;
  struct pi_cluster_task task = {0};

  pi_cluster_conf_init(&conf);
  conf.id = 0;
  pi_open_from_conf(&cluster, &conf);
  if (pi_cluster_open(&cluster)) {
    printf("ERROR: could not open cluster\n");
    return 1;
  }

  pi_cluster_task(&task, cluster_entry, NULL);
  pi_cluster_send_task_to_cl(&cluster, &task);
  pi_cluster_close(&cluster);

  if (errors == 0) {
    printf("\nRESULT: PASS - all samples and intermediate tensors match.\n");
    return 0;
  }

  printf("\nRESULT: FAIL - %d errors.\n", errors);
  return 1;
}

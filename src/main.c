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
  nnx_feature_t output_desc = {.data = output, // output es un puntero a los datos de salida de la capa, representa dónde quedará guardadda la salida después de la convolución.
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
  }; // weights_desc es una estructura que define la descripción de los pesos para la tarea de N-EUREKA. Contiene información sobre los datos de los pesos, incluyendo un puntero a los datos reales, las dimensiones del kernel (altura y ancho), la profundidad de los pesos (que corresponde al número de canales de entrada), el número de filtros (n_weights) que se aplicarán a la entrada, el ancho de bits de los pesos (bitwidth), un factor de compensación (offset_factor) para ajustar los valores de los pesos y un modo de compensación (offset_mode) que indica cómo se aplicará el factor de compensación a los pesos durante la operación de convolución.

  nnx_error_code status;
  /* Se configura task->cfg según el tipo de convolución, todavía no ejecutan la convolución*/
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

  BIT_SET(task->cfg.conf0, NEUREKA_FLAG_USE_TCDM); // BIT_SET es una macro que se utiliza para establecer un bit específico en un registro de configuración. En este caso, se está estableciendo el bit NEUREKA_FLAG_USE_TCDM en el registro conf0 de la configuración de la tarea. Esto indica a N-EUREKA que utilice la memoria TCDM (Tightly Coupled Data Memory) para almacenar los datos de entrada, salida y pesos durante la operación de convolución. La TCDM es una memoria rápida y cercana al clúster, lo que permite un acceso más eficiente a los datos y mejora el rendimiento de la operación de convolución.
  task->infeat_ptr = (uint32_t)input; // input es la dirección donde N-EUREKA leerá los datos de entrada para la operación de convolución. Se está asignando esta dirección al campo infeat_ptr de la tarea, que indica a N-EUREKA dónde encontrar los datos de entrada en la memoria.
  task->outfeat_ptr = (uint32_t)output; // output es la dirección donde N-EUREKA almacenará los resultados de la operación de convolución.
  task->weights_ptr = (uint32_t)packed_weights; // Guarda la dirección de los pesos empaquetados, estos pesos ya vienen en el formato especial de N-EUREKA, generado por generate_model.py.
  task->scale_ptr = (uint32_t)scale_l1; // Dirección del arreglo de escalas.
  task->scale_shift_ptr = 0; // Dirección del arreglo de shifts, no se usa en este ejemplo.
  task->scale_bias_ptr = (uint32_t)bias_l1; // Dirección del arreglo de bias.
  return 0;
}

/*
La tarea completa tiene la estructura:

typedef struct {
    uint32_t weights_ptr;
    uint32_t infeat_ptr;
    uint32_t outfeat_ptr;
    uint32_t scale_ptr;
    uint32_t scale_shift_ptr;
    uint32_t scale_bias_ptr;
    nnx_cfg_t cfg;
} nnx_task_t;

y dentro de cfg hay:

typedef struct {
    nnx_stride_t input_stride;
    nnx_stride_t output_stride;
    nnx_stride_t weights_stride;
    nnx_subtile_t subtile;
    uint32_t padding;
    uint32_t weight_offset_factor;
    uint32_t filter_mask;
    uint32_t conf0;
} nnx_cfg_t;

donde conf0 es un registro o palabra de largo 32 que contiene varios flags de configuración.

Antes, cuando hicimos:
  nnx_feature_t input_desc
  nnx_feature_t output_desc
  nnx_weights_t weights_desc

y luego:
  nnx_conv_1x1(&task->cfg, weights_desc, input_desc, output_desc); 

Lo que hicimos fue llenar task->cfg con la información de los strides, subtile, padding, weight_offset_factor, filter_mask y conf0.

Esto no los campos del task:

  task->infeat_ptr
  task->outfeat_ptr
  task->weights_ptr
  task->scale_ptr
  task->scale_shift_ptr
  task->scale_bias_ptr

Por eso después se asignan explícitamente con:
  task->infeat_ptr = (uint32_t)input;
  task->outfeat_ptr = (uint32_t)output;
  task->weights_ptr = (uint32_t)packed_weights;
  task->scale_ptr = (uint32_t)scale_l1;
  task->scale_shift_ptr = 0;
  task->scale_bias_ptr = (uint32_t)bias_l1;


las funciones nnx_conv_1x1, nnx_conv_3x3 y nnx_conv_3x3_dw solo llenan task->cfg, no los punteros a los datos. Usan:
  nnx_feature_t input_desc
  nnx_feature_t output_desc
  nnx_weights_t weights_desc

Con eso llenan task->cfg: 
  task->cfg.input_stride
  task->cfg.output_stride
  task->cfg.weights_stride
  task->cfg.subtile
  task->cfg.padding
  task->cfg.weight_offset_factor
  task->cfg.filter_mask
  task->cfg.conf0

Por ejemplo:

output.height/output.width
  sirven para calcular cuántos tiles espaciales necesita N-EUREKA

input.depth
  sirve para calcular cuántos bloques de canales de entrada hay

output.depth
  sirve para calcular cuántos canales/filtros de salida hay

weights.bitwidth
  sirve para configurar cuántos bits tienen los pesos

weights.offset_factor
  sirve para configurar el offset -128


Nota: nnx_stride_t tiene la forma:
typedef struct {
    uint16_t stride_h;
    uint16_t stride_w;
    uint16_t stride_c;
} nnx_stride_t;

En donde cada stride es la cantidad de bytes que hay que avanzar en memoria para pasar al siguiente elemento en esa dimensión. Por ejemplo, si la entrada es de 32x32x3 y cada elemento es de 1 byte, entonces:
  stride_h = 32*3 = 96
  stride_w = 3
  stride_c = 1

La fórmula para calcular el stride es:
  stride_h = width * depth * (bitwidth/8)
  stride_w = depth * (bitwidth/8)
  stride_c = (bitwidth/8)

  donde depth es el número de canales, width es el ancho de la imagen y bitwidth es el ancho en bits de cada elemento o celda.
  Esta forma de calcular el stride viene de la forma en que N-EUREKA organiza los datos en memoria, que es en el orden HWC (Height, Width, Channel).
  Por ejemplo, los datos están en la siguiente estructura en memoria:
  [H0W0C0, H0W0C1, H0W0C2, 
  H0W1C0, H0W1C1, H0W1C2, 
  H0W2C0, H0W2C1, H0W2C2, 
  H1W0C0, H1W0C1, H1W0C2, 
  H1W1C0, H1W1C1, H1W1C2,
  H1W2C0, H1W2C1, H1W2C2,
  H2W0C0, H2W0C1, H2W0C2...]
  
  Con C=3, cada pixel tiene 3 valores:
  pixel (0,0): c0 c1 c2
  pixel (0,1): c0 c1 c2
  pixel (0,2): c0 c1 c2
  ...
*/


/*

Orden general:
  1. nnx_task_init(...)
    limpia/inicializa la tarea

  2. crear descriptors:
    input_desc
    output_desc
    weights_desc

  3. nnx_conv_1x1 / nnx_conv_3x3 / nnx_conv_3x3_dw
    configura la tarea según el tipo de convolución

  4. nnx_norm_quant(...)
    configura ReLU, shift, cuantización

  5. llenar punteros:
    input, output, weights, scale, bias

  6. nnx_offload(...)
    escribe la tarea/configuración en registros de N-EUREKA

  7. nnx_run_blocking(...)
    lanza N-EUREKA y espera a que termine

Entonces, hay dos tipos de información:
  1. Informaicón para calcular cómo se ejecuta la capa:

    input_stride
    output_stride
    weights_stride
    subtile
    padding
    weight_offset_factor
    filter_mask
    conf0
  
    Esto responde a preguntas como:
      ¿Qué tipo de convolución es?
      ¿Cuántos tiles hay?
      ¿Cuántos canales?
      ¿Cómo avanzo en memoria?
      ¿Qué shift/ReLU/modo uso?
  
  2. Direcciones reales de memoria donde están los datos:
  
    task->infeat_ptr
    task->outfeat_ptr
    task->weights_ptr
    task->scale_ptr
    task->scale_bias_ptr
  
    Esto responde a preguntas como:
      ¿Dónde está la entrada?
      ¿Dónde está la salida?
      ¿Dónde están los pesos?
      ¿Dónde están los factores de escala y bias?
  
*/

static uint32_t execute_layer(const layer_config_t *layer, uint8_t *input,
                              uint8_t *output, uint8_t *packed_weights) {
  /*
  Esta función ejecuta una capa del modelo en N-EUREKA y devuelve cuántos ciclos de reloj demoró.
  Recibe:

  - layer: un puntero a la configuración de la capa que se va a ejecutar.
  - input: un puntero a los datos de entrada de la capa.
  - output: un puntero a los datos de salida de la capa.
  - packed_weights: un puntero a los pesos empaquetados de la capa.
  */
  nnx_task_t task; // Crea una tarea N-EUREKA local en la pila para almacenar la configuración de la capa que se va a ejecutar.
  if (configure_task(&task, layer, input, output, packed_weights) != 0) { // Llama a la función configure_task (definida anteriormente) para llenar el struct task.
    printf("Could not configure %s\n", layer->name);
    ++errors;
    return 0;
  } // Si la función devuelve un valor distinto a 0, hubo un error.

  nnx_soft_clear(); // Limpia los registros de N-EUREKA, sirve para dejar el acelerador en un estado conocido antes de lanzar una tarea.
  nnx_acquire(); // Intenta adquirir/reservar N-EUREKA para usarlo. Si N-EUREKA ya está ocupado, esta función esperará hasta que esté disponible.
  pi_perf_reset(); // Reinicia los contadores de rendimiento, para medir el tiempo de ejecución de la tarea. En este caso, se va a medir el tiempo en ciclos de reloj.
  pi_perf_start(); // Activa los contadores de rendimiento, para empezar a medir el tiempo de ejecución de la tarea.
  uint32_t start = pi_perf_read(PI_PERF_CYCLES); // Lee el valor actual del contador de ciclos de reloj, para tener un punto de referencia antes de ejecutar la tarea. Lo guarda como punto inicial.
  nnx_offload(&task); // Envía la tarea configurada a N-EUREKA, escribiendo la configuración en los registros del acelerador. Esto prepara a N-EUREKA para ejecutar la tarea.
  nnx_run_blocking(); // Lanza la ejecución de la tarea en N-EUREKA y espera a que termine. Esta función bloquea el programa hasta que N-EUREKA haya completado la tarea.
  uint32_t elapsed = pi_perf_read(PI_PERF_CYCLES) - start; // Lee de nuevo el contador de ciclos y reta el valor inicial, esto da cuántos ciclos de reloj denmoró ejecutando la tarea.
  pi_perf_stop(); // Detiene los contadores de rendimiento, ya no se necesita medir el tiempo..
  return elapsed; // Devuelve el número de ciclos de reloj que tomó ejecutar la capa en N-EUREKA.
}

static void load_weights(void) {
  /*
  Esta función carga los pesos del modelo desde la memoria L2 a la memoria L1/TCDM, para que N-EUREKA pueda acceder a ellos rápidamente durante la ejecución de las capas.
  */
  memcpy(stem_weights_l1, model_stem_weights, MODEL_STEM_WEIGHTS_SIZE); // Copia los pesos de la capa stem desde model_data.c (que están en memoria L2) hacia el buffer L1.
  /*
    origen:  model_stem_weights
    destino: stem_weights_l1
    tamaño:  MODEL_STEM_WEIGHTS_SIZE bytes
  */
  memcpy(dw_weights_l1, model_dw_weights, MODEL_DW_WEIGHTS_SIZE); // Copia los pesos de la capa depthwise a la memoria L1.
  memcpy(pw_weights_l1, model_pw_weights, MODEL_PW_WEIGHTS_SIZE); // Copia los pesos de la capa pointwise a la memoria L1.
  memcpy(head_weights_l1, model_head_weights, MODEL_HEAD_WEIGHTS_SIZE); // Copia los pesos de la capa head a la memoria L1.
  for (int channel = 0; channel < MODEL_PW_C; ++channel) { // Inicializa los factores de escala y sesgo para cada canal de salida de la capa pointwise. En este ejemplo, se establece un factor de escala de 1 y un sesgo de 0 para todos los canales, lo que significa que no se aplicará ningún ajuste a los valores de salida durante la normalización y cuantización.
    scale_l1[channel] = 1;
    bias_l1[channel] = 0;
  }
}

static void classify(uint32_t sample) {
  /*
  Esta función toma un índice de muestra (sample) y ejecuta las 4 capas del modelo en N-EUREKA,
  calcula la clase preducha y verifica que todo coincida con la referencia.
  */
  uint32_t cycles[4]; // Arreglo para almacenar los ciclos de reloj que tomó ejecutar cada capa del modelo.
  uint32_t scores[MODEL_CLASS_COUNT] = {0}; // Arreglo para almacenar las puntuaciones acumuladas para cada clase. Se inicializa en 0 para todas las clases antes de procesar la muestra.
  /*
  scores[0] -> vertical
  scores[1] -> horizontal
  scores[2] -> diagonal
  */
  memcpy(input_l1, model_inputs[sample], MODEL_INPUT_SIZE); // Copia la imagen de entrada desde model_inputs[sample] (viene de model_data.c) hacia input_l1.
  memset(stem_l1, 0, MODEL_STEM_SIZE); // Limpia el buffer de salida de la capa stem, llenándolo con ceros. Esto asegura que no haya datos residuales de ejecuciones anteriores antes de procesar la nueva muestra.
  memset(dw_l1, 0, MODEL_DW_SIZE);
  memset(pw_l1, 0, MODEL_PW_SIZE);
  memset(head_l1, 0, MODEL_HEAD_SIZE);

  cycles[0] =
      execute_layer(&layers[0], input_l1, stem_l1, stem_weights_l1); // Ejecuta la primera capa (stem 3x3) del modelo en N-EUREKA, pasando los datos de entrada, el buffer de salida y los pesos correspondientes. Almacena el número de ciclos de reloj que tomó ejecutar esta capa en cycles[0].
  cycles[1] = execute_layer(&layers[1], stem_l1, dw_l1, dw_weights_l1);
  cycles[2] = execute_layer(&layers[2], dw_l1, pw_l1, pw_weights_l1);
  cycles[3] = execute_layer(&layers[3], pw_l1, head_l1, head_weights_l1);

  int sample_errors = 0; // Variable para contar el número de errores encontrados al comparar los resultados de las capas con los valores esperados.
  sample_errors += compare_tensor("stem", stem_l1,
                                  model_expected_stem[sample], MODEL_STEM_SIZE);
  sample_errors += compare_tensor("depthwise", dw_l1,
                                  model_expected_dw[sample], MODEL_DW_SIZE);
  sample_errors += compare_tensor("pointwise", pw_l1,
                                  model_expected_pw[sample], MODEL_PW_SIZE);
  sample_errors += compare_tensor("head", head_l1,
                                  model_expected_head[sample], MODEL_HEAD_SIZE);

  for (uint32_t pixel = 0; pixel < MODEL_HEAD_H * MODEL_HEAD_W; ++pixel) { 
    /* 
    Hace una suma global por clase sobre el feature map final. head_l1 está en formato HWC/interleaved. Como hay MODEL_CLASS_COUNT = 3, cada pixel tiene:
      canal 0 vertical
      canal 1 horizontal
      canal 2 diagonal
    Entonces:
      head_l1[pixel * MODEL_CLASS_COUNT + channel] accede al valor de una clase en un pixel específico.

    La forma de calcular la predicción es sumar todos los valores de cada clase en el feature map final y luego elegir la clase con la puntuación más alta como la predicción final.
    Al principio, en el feature map final, cada posición espacial (píxel) tiene 3 valores:
      [score_vertical_local, score_horizontal_local, score_diagonal_local]
    La idea es sumar todos los scores por canal para obtener un score global por clase:
      score_vertical_global = sum(score_vertical_local)
      score_horizontal_global = sum(score_horizontal_local)
      score_diagonal_global = sum(score_diagonal_local)
    Luego, la clase predicha será la que tenga el score global más alto.
    Entonces al final, después de la suma por canal, tenemos un vector de 3 elementos:
      scores[0] = score_vertical_global
      scores[1] = score_horizontal_global
      scores[2] = score_diagonal_global
    */
    for (uint32_t channel = 0; channel < MODEL_CLASS_COUNT; ++channel) {
      scores[channel] += head_l1[pixel * MODEL_CLASS_COUNT + channel];
    }
  }

  uint32_t prediction = 0; // Inicializa la predicción con la primera clase (vertical). Luego, se comparan las puntuaciones de las otras clases para determinar cuál tiene la puntuación más alta y se actualiza la predicción en consecuencia.
  for (uint32_t channel = 1; channel < MODEL_CLASS_COUNT; ++channel) {
    if (scores[channel] > scores[prediction]) {
      prediction = channel;
    }
  } // básicamente, busca el índice del canal con la puntuación más alta en el vector scores y lo asigna a prediction.

  printf("\nSample %lu (%s)\n", (unsigned long)sample,
         model_labels[model_expected_class[sample]]); // IMprime el número de sample y su clase esperada
  printf("  scores: vertical=%lu horizontal=%lu diagonal=%lu\n",
         (unsigned long)scores[0], (unsigned long)scores[1],
         (unsigned long)scores[2]); // Imprime los scores globales por clase.
  printf("  predicted: %s\n", model_labels[prediction]); // Imprime la clase predicha por el modelo.
  for (uint32_t index = 0; index < 4; ++index) { // Imprime el tiempo de ejecución en ciclos de reloj para cada capa del modelo.
    printf("  %-22s %lu cycles\n", layers[index].name,
           (unsigned long)cycles[index]);
  }

  if (prediction != model_expected_class[sample]) { // Compara la clase predicha con la clase esperada para la muestra actual. Si no coinciden, se considera un error de predicción.
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

  /*
   Esta función es la entrada del clúster de procesamiento. Se ejecuta en el contexto del clúster y se encarga de habilitar N-EUREKA, cargar los pesos del modelo,
   ejecutar la clasificación de todas las muestras del conjunto de datos. Al final, deshabilita N-EUREKA y termina la ejecución del clúster. Esta función
   recibe un puntero genérico arg que no se utiliza en este caso, por lo que se hace un cast a void para evitar advertencias del compilador. Este formato es
   típico cuando se lanza una tarea al cluster, ya que la función de entrada debe tener un prototipo específico para ser compatible con la API del clúster.
  */
  (void)arg; // Se hace un cast a void para evitar advertencias del compilador sobre el argumento no utilizado. Básicamente le dice al compilador: "Sí, sé que no estoy usando este argumento, y está bien".
  NEUREKA_CG_ENABLE(); // Habilita el reloj (clock/gating) de N-EUREKA, permitiendo que el acelerador funcione. Esto es necesario antes de enviar cualquier tarea a N-EUREKA, ya que el acelerador necesita estar activo para procesar las operaciones de convolución. Básicamente enciende/habilita N-EUREKA para poder usarlo.
  NEUREKA_SETPRIORITY_NEUREKA(); // Da prioridad a N-EUREKA en el acceso al interconnect/memoria. Como N-EUREKA necesita leer inputs, pesos y escribir outputs, esto ayuda a que tenga prioridad frente a los cores cuando compite por memoria-
  
  /*
  Es como un switch binario:
    #define NEUREKA_SETPRIORITY_CORE() \
      *(volatile int*) (...) &= ~CLUS_CTRL_HWPE_HCI_PRIO_MASK

    #define NEUREKA_SETPRIORITY_NEUREKA() \
      *(volatile int*) (...) |= CLUS_CTRL_HWPE_HCI_PRIO_MASK
  
    Es decir:
      NEUREKA_SETPRIORITY_CORE()  -> prioridad a los cores
      NEUREKA_SETPRIORITY_NEUREKA() -> prioridad a N-EUREKA

  */
  
  NEUREKA_RESET_MAXSTALL(); // Resetea el contador de ciclos de espera (stall) de N-EUREKA. Esto es útil para medir el rendimiento y detectar si N-EUREKA está experimentando retrasos significativos debido a la contención de recursos o problemas de memoria. Al resetear este contador, se puede obtener una medición más precisa del tiempo que N-EUREKA pasa esperando por recursos antes de ejecutar las tareas.
  NEUREKA_SET_MAXSTALL(8); // Configura el umbral de ciclos de espera (stall) de N-EUREKA a 8. Esto significa que si N-EUREKA experimenta más de 8 ciclos de espera consecutivos, se puede considerar que hay un problema de rendimiento o contención de recursos. Este valor puede ser ajustado según las necesidades del sistema y el comportamiento esperado del acelerador. Básicamente ajusta cuánto puede esperar o cómo se regula su acceso al bus/memoria. Es una configuración de bajo nivel del HWPE/interconnect.
  
    
  /*
    1. N-EUREKA pide acceso a L1/TCDM.
    2. Hay contención con cores u otro acceso.
    3. N-EUREKA queda esperando: stall.
    4. Si la espera llega al umbral configurado, por ejemplo 8 ciclos,
      la interfaz puede ceder, reintentar, cambiar prioridad efectiva,
      o permitir que otros masters avancen.
    5. Después N-EUREKA continúa cuando obtiene acceso.
  */
  
  /*
  NEUREKA_SETPRIORITY_NEUREKA()
  pone a N-EUREKA como master de mayor prioridad

  NEUREKA_SET_MAXSTALL(8)
    permite que el bus stallee al master de menor prioridad (en este caso el master de menor prioridad pasa a ser los cores) por hasta 8 ciclos

  Entonces NEUREKA_SET_MAXSTALL(8) significa conceptualmente:
    con N-EUREKA priorizada, el bus puede hacer esperar al core hasta 8 ciclos antes de aplicar su política de arbitraje


  En resumen:

      priority bit:
        elige quién tiene prioridad: core o N-EUREKA

      maxstall:
        define cuántos ciclos el HCI puede stalleear al master de menor prioridad

      con NEUREKA_SETPRIORITY_NEUREKA + MAXSTALL(8):
        N-EUREKA tiene prioridad
        el core puede ser postergado/stalleado hasta 8 ciclos según la política HCI
  */
  pi_perf_conf(1 << PI_PERF_CYCLES); // Configura el contador de rendimiento para medir sólo los ciclos de reloj.

  load_weights(); // Copia los pesos empaquetados desde model_data.c/L2 hacia buffers en L1, e inicializa scale_l1 y bias_l1.
  for (uint32_t sample = 0; sample < MODEL_SAMPLE_COUNT; ++sample) { // Recorre todos los samples de prueba y ejecuta la clasificación de cada uno, verificando los resultados y acumulando errores.
    classify(sample);
  }

  NEUREKA_CG_DISABLE(); // Deshabilita el reloj (clock/gating) de N-EUREKA, apagando el acelerador.
}

int main(void) {
  /*
  main() corre en el Fabric Controller (FC) y lanza el trabajo pesado al clúster, donde está N-EUREKA. El FC se encarga
  de inicializar el clúster, abrirlo, enviar la tarea de clasificación y luego cerrarlo. Al final, imprime el resultado
  global de la ejecución, indicando si todas las muestras pasaron la verificación o si hubo errores.
  */
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

/*

NOTAS:
Actualmente, el código está usando sólo L1/TCDM para almacenar los datos de entrada, salida y pesos para ser usados por N-EUREKA. Estas líneas:

  PI_L1 static uint8_t input_l1[...];
  PI_L1 static uint8_t stem_l1[...];
  PI_L1 static uint8_t stem_weights_l1[...];

  ponen datos en L1/TCDM. Y esta línea:

    BIT_SET(task->cfg.conf0, NEUREKA_FLAG_USE_TCDM);
  
  le dice a N-EUREKA:
    lee los pesos desde TCDM/L1

  Mapa mental:

    L1 / TCDM:
      memoria compartida entre cores RISC-V y N-EUREKA
      aquí están ahora inputs, outputs, feature maps y pesos

    WEIGHTMEM_MRAM:
      MRAM del subsistema de pesos de N-EUREKA
      dirección base aprox: 0x10400020

    WEIGHTMEM_SRAM:
      SRAM del subsistema de pesos de N-EUREKA
      dirección base aprox: 0x10800020

  Para mover pesos a MRAM o SRAM de N-EUREKA tendríamos que:
    1. No copiarlos a stem_weights_l1, dw_weights_l1, etc.
    2. Copiarlos/escribirlos en WEIGHT_MEM_BASE + MRAM_OFFSET o WEIGHT_MEM_BASE + SRAM_OFFSET.
    3. Cambiar el flag:
      BIT_SET(task->cfg.conf0, NEUREKA_FLAG_USE_WMEM);
    4. Pasar ese puntero como task->weights_ptr.
  
  Ejemplo:

    #define WEIGHT_MEM_BASE 0x10400020
    #define MRAM_OFFSET 0x00000000
    #define SRAM_OFFSET 0x00400000 // definidos en pulp_nnx_hal.h

    uint8_t *weights_wmem = (uint8_t *)(WEIGHT_MEM_BASE + MRAM_OFFSET);
    memcpy(weights_wmem, model_stem_weights, MODEL_STEM_WEIGHTS_SIZE);

    BIT_SET(task->cfg.conf0, NEUREKA_FLAG_USE_WMEM);
    task->weights_ptr = (uint32_t)weights_mram;


  Si queremos definir varios pesos (distintas capas) en MRAM/SRAM, en lugar de usar direcciones absolutas
  (para evitar sobreescribir pesos entre capas, solapamiento) podemos hacer:
    __attribute__((section(".weightmem_mram"), aligned(32))) // Pone el arreglo en la sección llamada .weightmem_mram. Y en el linker script de Siracusam esa sección está asociada a la región de memoria WEIGHTMEM_MRAM. Entonces esta variable queda ubicada en la MRAM del subsistema de pesos N-EUREKA. aligned(32) asegura que la dirección de inicio del arreglo esté alineada a 32 bytes, lo cual es un requisito para N-EUREKA, porque N-EUREKA accede a los pesos en bloques de 32 bytes. Esto evita problemas de alineación y garantiza un acceso eficiente a la memoria.
    static uint8_t stem_weights_mram[MODEL_STEM_WEIGHTS_SIZE];

    __attribute__((section(".weightmem_mram"), aligned(32)))
    static uint8_t dw_weights_mram[MODEL_DW_WEIGHTS_SIZE];

    __attribute__((section(".weightmem_mram"), aligned(32)))
    static uint8_t pw_weights_mram[MODEL_PW_WEIGHTS_SIZE];

    __attribute__((section(".weightmem_mram"), aligned(32)))
    static uint8_t head_weights_mram[MODEL_HEAD_WEIGHTS_SIZE];

  Esto pondría cada arreglo en la sección de MRAM, y el linker colocaría cada uno en una dirección distinta, evitando solapamientos. Luego, en load_weights() haríamos:

    memcpy(stem_weights_mram, model_stem_weights, MODEL_STEM_WEIGHTS_SIZE);
    memcpy(dw_weights_mram, model_dw_weights, MODEL_DW_WEIGHTS_SIZE);
    memcpy(pw_weights_mram, model_pw_weights, MODEL_PW_WEIGHTS_SIZE);
    memcpy(head_weights_mram, model_head_weights, MODEL_HEAD_WEIGHTS_SIZE);
*/
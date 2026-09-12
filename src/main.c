#include "model_data.h"
#include "pmsis.h"
#include "pulp_nnx_hal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef enum {
  LAYER_CONV_3X3,
  LAYER_DEPTHWISE_3X3,
  LAYER_CONV_1X1,
} layer_operation_t;

typedef struct {
  const char *name;
  layer_operation_t operation;
  uint16_t input_h;
  uint16_t input_w;
  uint16_t input_c;
  uint16_t output_h;
  uint16_t output_w;
  uint16_t output_c;
  uint8_t shift;
} layer_config_t;

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

PI_L1 static uint8_t input_l1[MODEL_INPUT_SIZE];
PI_L1 static uint8_t stem_l1[MODEL_STEM_SIZE];
PI_L1 static uint8_t dw_l1[MODEL_DW_SIZE];
PI_L1 static uint8_t pw_l1[MODEL_PW_SIZE];
PI_L1 static uint8_t head_l1[MODEL_HEAD_SIZE];

PI_L1 static uint8_t stem_weights_l1[MODEL_STEM_WEIGHTS_SIZE];
PI_L1 static uint8_t dw_weights_l1[MODEL_DW_WEIGHTS_SIZE];
PI_L1 static uint8_t pw_weights_l1[MODEL_PW_WEIGHTS_SIZE];
PI_L1 static uint8_t head_weights_l1[MODEL_HEAD_WEIGHTS_SIZE];

PI_L1 static uint32_t scale_l1[MODEL_PW_C];
PI_L1 static int32_t bias_l1[MODEL_PW_C];

static int errors;

static int compare_tensor(const char *name, const uint8_t *actual,
                          const uint8_t *expected, uint32_t size) {
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
  nnx_task_init(task);
  nnx_pad_input(&task->cfg, 0, 0, 0, 0, 0);

  nnx_feature_t input_desc = {.data = input,
                              .height = layer->input_h,
                              .width = layer->input_w,
                              .depth = layer->input_c,
                              .bitwidth = featureBitwidth8Bit};
  nnx_feature_t output_desc = {.data = output,
                               .height = layer->output_h,
                               .width = layer->output_w,
                               .depth = layer->output_c,
                               .bitwidth = featureBitwidth8Bit};

  const uint16_t kernel_size =
      layer->operation == LAYER_CONV_1X1 ? 1 : 3;
  nnx_weights_t weights_desc = {
      .data = packed_weights,
      .height = kernel_size,
      .width = kernel_size,
      .depth = layer->input_c,
      .n_weights = layer->output_c,
      .bitwidth = 8,
      .offset_factor = -128,
      .offset_mode = weightOffsetModeLayerWise,
  };

  nnx_error_code status;
  if (layer->operation == LAYER_CONV_1X1) {
    status = nnx_conv_1x1(&task->cfg, weights_desc, input_desc, output_desc);
  } else if (layer->operation == LAYER_DEPTHWISE_3X3) {
    status =
        nnx_conv_3x3_dw(&task->cfg, weights_desc, input_desc, output_desc);
  } else {
    status = nnx_conv_3x3(&task->cfg, weights_desc, input_desc, output_desc);
  }
  if (status != 0) {
    return status;
  }

  nnx_norm_t norm = {
      .mode = normMode32Bit, .flag_bias = 0, .flag_shift = 0};
  nnx_quant_t quant = {.shift_amount = layer->shift,
                       .mode = quantMode8Bit,
                       .function = quantFunctionRelu,
                       .flag_rounding = 0};
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

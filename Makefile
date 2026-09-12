APP = manual_mobilenet

NNX_ROOT := vendor/pulp-nnx

APP_SRCS += src/main.c
APP_SRCS += src/model_data.c
APP_SRCS += $(NNX_ROOT)/src/pulp_nnx_hal.c

APP_CFLAGS += -Iinc -I$(NNX_ROOT)/inc -O3 -Wall
APP_CFLAGS += -Wno-unused-function
APP_LDFLAGS += -Wl,--print-memory-usage

PMSIS_OS = pulpos

include $(RULES_DIR)/pmsis_rules.mk

.PHONY: model
model:
	python3 tools/generate_model.py

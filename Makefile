APP = manual_mobilenet # nombre de la aplicación/ejecutable que se va a construir

NNX_ROOT := vendor/pulp-nnx # directorio raíz de la librería pulp-nnx, ":=" es un operador de asignación recursiva, lo que significa que el valor de NNX_ROOT se evaluará cada vez que se use.

# APP_SRCS es una variable que contiene la lista de archivos fuente que se van a compilar para construir la aplicación. En este caso, se incluyen tres archivos fuente: src/main.c, src/model_data.c y $(NNX_ROOT)/src/pulp_nnx_hal.c. El operador "+=" se utiliza para agregar estos archivos a la variable APP_SRCS.
APP_SRCS += src/main.c
APP_SRCS += src/model_data.c
APP_SRCS += $(NNX_ROOT)/src/pulp_nnx_hal.c

# APP_CFLAGS es una variable que contiene las opciones de compilación para el compilador. En este caso, se agregan las siguientes opciones:

APP_CFLAGS += -Iinc -I$(NNX_ROOT)/inc -O3 -Wall # -Iinc: agrega el directorio "inc" a la lista de directorios donde el compilador buscará archivos de encabezado.
APP_CFLAGS += -Wno-unused-function # Desactiva las advertencias sobre funciones no utilizadas.
APP_LDFLAGS += -Wl,--print-memory-usage # Imprime el uso de memoria durante el enlazado.
# APP_LDFLAGS es  una variable que contiene las opciones de enlace para el enlazador. En este caso, se agrega la opción -Wl,--print-memory-usage, que indica al enlazador que imprima información sobre el uso de memoria durante el proceso de enlace.

PMSIS_OS = pulpos # Define qué sistema operativo usar. PULPOS es el sistema operativo de PULP (basado en un scheduler ligero). Otras opciones son  freertos, baremetal, etc. Esta variable la usa el sistema de build de PMSIS para enlazar el sistema operativo correcto.

include $(RULES_DIR)/pmsis_rules.mk # Incluye un makefile con reglas predefinidas de PMSIS. Esto permite que el makefile herede reglas y configuraciones específicas de PMSIS, facilitando la construcción de aplicaciones para la plataforma PULP.
# pmsis_rules.mk contiene reglas y configuraciones específicas para compilar aplicaciones que utilizan el sistema operativo PULPOS. Al incluir este archivo, el makefile hereda estas reglas y configuraciones, ahora sabe cómo compilar APP_SCRS.
# RULES_DIR se define cuando hacemos source /home/chen/master/pulp-sdk/configs/siracusa.sh

.PHONY: model # Indica a Make que "model" es un objetivo que no corresponde a un archivo real. Esto significa que cada vez que se invoque "make model", se ejecutará la receta asociada, independientemente de si existe un archivo llamado "model" o no.
model:
	python3 tools/generate_model.py # Invoca un script de Python llamado generate_model.py ubicado en el directorio tools. Este script se encarga de generar o preparar el modelo de red neuronal que se utilizará en la aplicación.

# Comparativa y análisis de KSWIN modificado para la detección de concept drift

Este repositorio contiene el desarrollo y evaluación de un detector de cambios conceptuales (concept drift) en flujos de datos, basado en una modificación del algoritmo **KSWIN** de la librería `river`. El trabajo incluye tanto la implementación del detector `KSWIN_modificado` como una batería de experimentos comparativos frente a otras variantes de `KSWIN`, en escenarios con diferentes tipos de drift (abrupto, gradual, incremental y recurrente).

## Contenido del repositorio

- `KSWIN_modificado.py`: implementación extendida de `KSWIN`, con detección confirmada, identificación del tipo de drift y pruebas estadísticas múltiples corregidas.
- `Experimentos/`: scripts utilizados para generar flujos de datos sintéticos y reales con diferentes tipos de concept drift, así como los experimentos de evaluación y visualización de resultados.
- `EntornoWindows.yml` y `EntornoUbuntu.yml`: archivos para crear el entorno Conda necesario para ejecutar los experimentos, con las dependencias correspondientes.
- `Resultados/`: directorios y scripts para almacenar y analizar resultados como métricas de rendimiento, precisión en la detección y falsas alarmas.
- `Documentación_TFG/`: incluye documentación y figuras del trabajo de fin de grado asociado.

## Instalación y ejecución

### 1. Instalación de Miniconda

Para ejecutar correctamente los experimentos es necesario contar con **Miniconda** instalado. Puede descargarse desde el siguiente enlace: https://docs.conda.io/en/latest/miniconda.html

### 2. Creación del entorno Conda

Una vez instalado Miniconda, se debe crear un entorno virtual utilizando el archivo `.yml` correspondiente según el sistema operativo:

- **En Windows** (ejecutar en la terminal de Anaconda Prompt):

```bash
conda env create -f EntornoWindows.yml


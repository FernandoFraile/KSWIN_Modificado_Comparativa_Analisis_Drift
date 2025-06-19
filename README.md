# Comparativa y análisis de KSWIN modificado para la detección de concept drift

Este repositorio contiene el desarrollo y evaluación de un detector de Concept Drift en flujos de datos, basado en una modificación del algoritmo **KSWIN** de la librería `RiverML` (https://riverml.xyz/0.21.2/). El trabajo incluye tanto la implementación del detector `KSWIN_modificado` como una batería de experimentos comparativos frente a otras variantes de `KSWIN`, en escenarios con diferentes tipos de drift (abrupto, gradual, incremental y recurrente).

## 📁 Contenido del repositorio

- `KSWIN_modificado.py`: implementación extendida del detector KSWIN para detección de concept drift, con tres configuraciones mejoradas:
  - **Configuración 1:** compara ventanas aleatorias mediante KS-test con corrección FDR. Alta sensibilidad.
  - **Configuración 2:** reemplaza el muestreo por divisiones secuenciales para mejorar estabilidad y robustez.
  - **Configuración 3:** añade una ventana de confirmación tras la detección y permite identificar el tipo de drift (abrupto, gradual o incremental) usando regresión kernel.
- `AgrawalIncDrift.py`: modificación del generador de datos Agrawal para simular drifts incrementales. Permite controlar el ancho de transición, interpolación entre funciones y reversibilidad del cambio.
- `EntornoWindows.yml` y `EntornoUbuntu.yml`: archivos para crear el entorno Conda necesario para ejecutar los experimentos, con las dependencias correspondientes.
- **Notebooks de experimentación**:
  - `Experimentacion_KSWINModConfig1y2(DistribucionesContinuasSimuladas).ipynb`: pruebas sobre datos univariantes simulados con distribuciones estadísticas (Normal, Gamma, Exponencial), usando las configuraciones 1 y 2 del detector KSWIN modificado.
  - `Experimentacion_KSWINModConfig1y2(ModeloPrediccion).ipynb`: evaluación del detector con configuraciones 1 y 2 en el contexto de aprendizaje supervisado, sobre un modelo de predicción, monitorizando el error cometido. 
  - `Experimentacion_KSWINModConfig3_Reentreno10k.ipynb`: pruebas completas con la configuración 3, incorporando ventana de confirmación, identificación del tipo de drift y mecanismo de reentrenamiento del modelo cada 10.000 instancias tras una detección.


## 🛠 Instalación y ejecución

### 1. Instalación de Miniconda

Para ejecutar correctamente los experimentos es necesario contar con **Miniconda** instalado. Puede descargarse desde el siguiente enlace oficial:

👉 https://docs.conda.io/en/latest/miniconda.html

### 2. Creación del entorno de ejecución

Una vez instalado Miniconda, se debe crear un entorno virtual específico para el proyecto. Dependiendo del sistema operativo, ejecutar uno de los siguientes comandos:

- **En Windows** (en la terminal de Anaconda Prompt):

```bash
conda env create -f EntornoWindows.yml
```

- **En Ubuntu/Linux** (en la terminal del sistema):

```bash
conda env create -f EntornoUbuntu.yml
```

Esto instalará todas las dependencias necesarias (paquetes de Python, bibliotecas científicas, entorno de visualización, etc.).

### 3. Activación del entorno

Una vez creado el entorno, activarlo con:

```bash
conda activate EntornoTFG
```

> El nombre del entorno se define dentro del archivo `.yml`. Asegúrese de usar el que corresponda (por defecto, `EntornoTFG` o `EntornoWindows`, según el caso).

### 4. Instalación de Jupyter (si no está incluido)

Si por alguna razón Jupyter Notebook no se instaló automáticamente al crear el entorno, puede instalarse manualmente con:

```bash
conda install jupyter
```

### 5. Ejecución de notebooks

Una vez activado el entorno, navegar a la carpeta del repositorio y ejecutar:

```bash
jupyter notebook
```

Esto abrirá una ventana del navegador donde se podrán explorar y ejecutar los notebooks interactivos (`.ipynb`). Cada celda puede ejecutarse con **Shift + Enter**. Se recomienda seguir el orden lógico de los notebooks para reproducir los experimentos de forma completa.

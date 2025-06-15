from __future__ import annotations


from river.drift import KSWIN
from river import metrics
import collections
import random
import itertools
import warnings
from scipy import stats
from statsmodels.stats.multitest import multipletests
import math
import pandas as pd  # Importamos pandas para usar Series y EWMA
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg
import seaborn as sns
import typing 
import copy



class KSWIN_modificado(KSWIN):

    """
    Detector de concept drift basado en el test de Kolmogorov-Smirnov (KS), con extensiones para:
    - Múltiples configuraciones de comparación estadística.
    - Confirmación opcional del drift (útil para evitar falsos positivos).
    - Identificación automática del tipo de drift: 'abrupt', 'gradual' o 'incremental'.
    - Integración con métricas de rendimiento mediante `river.metrics`.
    - Suavizado de métricas mediante media móvil exponencial (EWMA).

    Esta clase hereda del detector `KSWIN` de la librería `river`, extendiéndolo con lógica adicional
    para adaptarlo a entornos donde se requiere análisis más detallado del comportamiento post-drift.

    ----------------------
    Parámetros del init:
    ----------------------
    - alpha (float): nivel de significancia del test KS. Cuanto menor, mayor sensibilidad.
    - window_size (int): tamaño de la ventana deslizante para detección.
    - stat_size (int): tamaño del subconjunto usado en las comparaciones estadísticas.
    - seed (int | None): semilla para reproducibilidad en procesos aleatorios.
    - window (Iterable | None): ventana inicial de datos. Si None, comienza vacía.
    - window_start (int): número de observaciones necesarias antes de comenzar a evaluar drift.
    - alternative (str): hipótesis alternativa del test KS ('greater', 'less' o 'two-sided').
    - configuracion (int): estrategia de comparación (1, 2 o 3) usada en el método `update`:
        * 1 = comparación con muestreo aleatorio,
        * 2 = comparación con bloques consecutivos,
        * 3 = comparación con múltiples desplazamientos + confirmación.
    - es_continua (bool): indica si los datos son valores continuos o métricas (suavizadas).
    - metric (Metric | None): métrica de rendimiento de `river.metrics` para analizar degradación.

    ----------------------
    Atributos internos:
    ----------------------
    - self._rng: generador aleatorio controlado por `seed`.
    - self.metric_aux_window: almacén de métricas para EWMA.
    - self.confirm_window: ventana para confirmar y analizar tipo de drift.
    - self._drift_detected: indica si se ha detectado un drift.
    - self.drift_confirmed: indica si la detección fue verificada como real.
    - self._tipo_drift: tipo de drift identificado ('abrupt', 'gradual', 'incremental').
    - self.identificado_tipo: evita recalcular el tipo si ya fue determinado.

    ----------------------
    Métodos públicos:
    ----------------------

    - update(x):
        Método principal de actualización. Recibe un nuevo lote de datos `x` y decide si hay drift.
        En función de la configuración elegida, compara ventanas usando múltiples tests KS corregidos
        por FDR (Benjamini-Hochberg). Si se detecta drift y `configuracion == 3`, se entra en una fase
        de confirmación para reducir falsos positivos y determinar el tipo de drift.

    - drift_detected (property):
        Getter/setter para exponer la detección externa de drift como booleano público.

    - tipo_drift (property):
        Devuelve el tipo de drift detectado si se ha identificado.

    - metric (property/setter):
        Permite asociar o actualizar una métrica de `river.metrics` a esta instancia.
    
    ----------------------
    Métodos privados:
    ----------------------

    - _identificar_tipo_drift(confirm_window):
        Analiza la forma de la serie tras el drift. Si hay tendencia creciente → 'gradual',
        si no, compara la ventana real con una curva de degradación simulada para clasificar
        como 'abrupt' o 'incremental'.

    - _suavizar_metrica(x, ventana_confirmacion=False):
        Aplica suavizado exponencial (EWMA) sobre las métricas de rendimiento, útil cuando
        `es_continua=False`. Si `ventana_confirmacion=True`, solo devuelve el suavizado
        sin almacenar.

    ----------------------
    Notas:
    ----------------------
    - El uso de múltiples tests KS y corrección FDR aumenta la robustez frente a falsos positivos.
    - El análisis posterior al drift permite enriquecer los resultados con tipología de cambio.
    - Este detector está diseñado para escenarios en los que se desea tanto detectar cambios
      como caracterizarlos y validarlos.
    """
    def __init__(
        self,
        alpha: float = 0.00001,
        window_size: int = 3000,
        stat_size: int = 300,
        seed: int | None = None,
        window: typing.Iterable | None = None,
        window_start: int = 1300,
        alternative: str = "greater", 
        configuracion: int = 1,
        es_continua: bool = False,
        metric : metrics.base.Metric | None = None
    ):
        # Llama al constructor de la clase base
        super().__init__(alpha, window_size, stat_size, seed, window)
        self._rng = random.Random(seed)
        self.window_start = window_start
        self.alternative = alternative
        self.drift_confirmed = False
        self._drift_detected = False
        self.analisisPrevio = True
        self.configuracion = configuracion 
        self.metric_aux_window = collections.deque(maxlen=window_size)
        self.es_continua = es_continua
        self._tipo_drift = None
        self.identificado_tipo = False
        self.valor_drift = []
        self._metric = metric
        self.confirm_window = []

    @property
    def drift_detected(self):
        """Getter para el atributo público `drift_detected`."""
        return self._drift_detected
    
    @drift_detected.setter
    def drift_detected(self, value):
        """Setter para el atributo público `drift_detected`."""
        if not isinstance(value, bool):
            raise ValueError("drift_detected debe ser un valor booleano.")
        self._drift_detected = value

    @property
    def tipo_drift(self):
        return self._tipo_drift



    
    @property 
    def metric(self):
        return self._metric
    
    @metric.setter
    def metric(self, metric):
        if metric is not None and not isinstance(metric, metrics.base.Metric):
            raise ValueError("El objeto metric debe ser una instancia de la clase base.Metric")
        self._metric = copy.deepcopy(metric)




    

    def _identificar_tipo_drift(self, confirm_window):
        """
        Método interno que intenta identificar automáticamente el tipo de 'concept drift' 
        ocurrido (gradual, abrupt o incremental) a partir de una ventana de confirmación 
        de datos tras una detección.

        Requisitos:
        - Es necesario que la instancia tenga definida una métrica compatible con 
          `river.metrics.base.Metric`, ya que se utiliza para simular patrones de rendimiento 
          bajo distintas hipótesis de drift.

        Parámetros:
        - confirm_window (list or array): conjunto de observaciones posterior al punto de detección, 
          utilizado para analizar el comportamiento del sistema tras el cambio.

        Procedimiento:
        1. Se aplica regresión no paramétrica (KernelReg) sobre la mitad inicial de la ventana de confirmación.
        2. Si la derivada estimada muestra tendencia creciente, se clasifica como drift "gradual".
        3. Si no hay crecimiento, se simula una caída abrupta en la métrica y se compara contra la ventana real:
           - Si el test de Kolmogorov-Smirnov encuentra diferencias → drift "abrupt".
           - En caso contrario, se asume un drift "incremental".
        """

        # Verificación del tipo de métrica proporcionada
        if not isinstance(self._metric, metrics.base.Metric):
            raise ValueError("Se necesita un objeto base.Metric para identificar el tipo de drift")

        # Establecer semilla para reproducibilidad
        np.random.seed(123)

        # Preparar datos para estimación no paramétrica
        x = np.linspace(0, len(confirm_window[0:self.window_size // 2]), len(confirm_window[0:self.window_size // 2]))
        y = np.array(confirm_window[0:self.window_size // 2])

        # Ajuste de regresión kernel: estimación de tendencia y derivada local
        kr = KernelReg(y, x, var_type='c')  # 'c' = variable continua
        y_pred, dy_dx = kr.fit(x)

        # Caso 1: pendiente creciente → drift gradual
        if any(dy_dx > 0):
            self._tipo_drift = "gradual"

        # Caso 2: sin evidencia de pendiente → evaluar entre abrupto o incremental
        else:
            self.valores_en_drift = []

            # Simulación de evolución métrica ante caída de rendimiento tras drift abrupto
            for i in range(1, self.stat_size):
                valor = np.random.choice([0, 1], p=[0.6, 0.4])  # 60% de acierto, 40% error
                if valor == 0:
                    self._metric.update(1, 1)  # correcto
                else:
                    self._metric.update(1, 0)  # error
                self.valores_en_drift.append(self._metric.get())

            # Suavizado exponencial de los valores simulados
            accuracies = pd.Series(self.valores_en_drift)
            smoothed_accuracies = accuracies.ewm(span=self.stat_size, adjust=False).mean()

            # Comparación estadística: KS test entre ventana real y simulada
            st, p_valor = stats.ks_2samp(
                confirm_window[self.stat_size:(self.stat_size * 2)],
                smoothed_accuracies.to_list()[-self.stat_size:],
                method="auto",
                alternative="greater"
            )

            # Si p-valor indica diferencia significativa → drift abrupto
            if p_valor < 0.05:
                self._tipo_drift = "abrupt"
            else:
                self._tipo_drift = "incremental"



    def _suavizar_metrica(self, x, ventana_confirmacion: bool = False):
        """
        Método interno para aplicar un suavizado exponencial (EWMA) sobre los valores de métrica acumulados.


        Parámetros:
        - x (iterable): nuevos valores de métrica que se desean suavizar.
        - ventana_confirmacion (bool): 
            - Si False (por defecto), los valores se acumulan en `self.metric_aux_window`.
            - Si True, se asume que los valores ya están completos y no se almacenan en la ventana.

        Retorna:
        - smoothed_accuracies (pd.Series): últimos `len(x)` valores suavizados, tras aplicar EWMA.
        """

        # Si no es una ventana de confirmación, acumular las métricas
        if not ventana_confirmacion:
            self.metric_aux_window.extend(x)

        # Crear Serie con todos los valores acumulados y aplicar suavizado exponencial
        accuracies = pd.Series(self.metric_aux_window)
        smoothed_accuracies = accuracies.ewm(span=self.stat_size, adjust=False).mean()

        # Devolver sólo los últimos valores suavizados, del mismo tamaño que la entrada x
        return smoothed_accuracies[-len(x):]



    def update(self, x):

        if len(x) > self.window_size:
            raise ValueError("El tamaño de la ventana debe ser mayor que el tamaño de la muestra")

        if self.drift_confirmed == False: 
            self._drift_detected = False
       

        if self.window_start <= 0:

            if self.es_continua:
                self.window.extend(x)
            else:  
                self.window.extend(self._suavizar_metrica(x))

            if len(self.window) >= self.window_size:


                if self.configuracion == 3 and self._drift_detected == False: 

                    #Modificación para hacer múltiples pruebas 

                    p_values = []
                        
                    for i in range(self.stat_size):


                        less_recent = list(
                            itertools.islice(self.window,0, self.window_size - self.stat_size)
                        )

                        most_recent = list(
                            itertools.islice(self.window, 0+i, self.window_size - self.stat_size + i)
                        )


                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            st, self.p_value = stats.ks_2samp(most_recent, less_recent, method="auto", alternative=self.alternative)

                        p_values.append(self.p_value)

                    #Corrección de BENJAMINI-HOCHBERG

                    
                    corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

                    # Modificación para que se detecte el cambio si alguna de las pruebas es significativa
                    if any([p <= self.alpha for p in corrected_p_values]):

                        self._drift_detected = True
                        self.drift_confirmed = True
                        self.analisisPrevio = True
                        if self.es_continua:
                            self.valores_en_drift = collections.deque(self.window, maxlen=self.window_size)
                        else: 
                            self.valores_en_drift = collections.deque(self.metric_aux_window, maxlen=self.window_size)

                elif self.configuracion == 1:

                    p_values = []

                    for i in range(self.stat_size):

                        less_recent = [
                            self.window[r]
                            for r in self._rng.sample(range(self.window_size - self.stat_size), self.stat_size)
                        ]
                        most_recent = list(
                            itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
                        )


                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            st, self.p_value = stats.ks_2samp(most_recent, less_recent, method="auto", alternative=self.alternative)

                        p_values.append(self.p_value)

                    #Corrección de BENJAMINI-HOCHBERG

                        
                    corrected_p_values = multipletests(p_values, method='fdr_bh')[1]



                    # Modificación para que se detecte el cambio si alguna de las pruebas es significativa
                    if any([p <= self.alpha for p in corrected_p_values]):
                        self._drift_detected = True

                elif self.configuracion == 2 :

                    p_values = []

                    for i in range((self.window_size//self.stat_size)-1):

                        less_recent = list(
                            itertools.islice(self.window,0+i*self.stat_size, i*self.stat_size + (i+1)*self.stat_size)
                        )

                        most_recent = list(
                            itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
                        )


                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            st, self.p_value = stats.ks_2samp(most_recent, less_recent, method="auto", alternative='greater')

                        p_values.append(self.p_value)

                    #Corrección de BENJAMINI-HOCHBERG

                    
                    corrected_p_values = multipletests(p_values, method='fdr_bh')[1]


                    # Modificación para que se detecte el cambio si alguna de las pruebas es significativa
                    if any([p <= self.alpha for p in corrected_p_values]):
                        self._drift_detected = True
               

                
        else:
            self.window_start -= len(x)

        if self._drift_detected:


            if self.configuracion == 3:
                
                # Comprobacion de si es un pico al alza en los últimos valores anteriores a la confirmacion de drift 

                if self.analisisPrevio:

                    less_recent = list(
                        itertools.islice(self.window,self.window_size- 2*self.stat_size, self.window_size- self.stat_size)
                    )

                    most_recent = list(
                        itertools.islice(self.window, self.window_size- self.stat_size, self.window_size)
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        st, self.p_value = stats.ks_2samp(most_recent, less_recent, method="auto", alternative='less')

                    
                    if self.p_value <= self.alpha:
                        self.drift_confirmed = False
                        self._drift_detected = False
                        self.confirm_window = []
                     
                    self.analisisPrevio = False

                if self.drift_confirmed == True: 

                    if len(self.confirm_window) < self.window_size:

                        self.confirm_window.extend(list(self.window)[-self.stat_size:])

                        #Comprobación para que no se pase del tamaño de la ventana. Si se pasa, eliminar la diferencia
                        if len(self.confirm_window) > self.window_size:
                            self.confirm_window = self.confirm_window[0:self.window_size]


                        if len(self.confirm_window) >= (2*self.stat_size): 

                            p_values = []
                            
                            for i in range((len(self.confirm_window)//self.stat_size)-1):

                                less_recent = self.confirm_window[i*self.stat_size : (i+1)*self.stat_size]

                                most_recent = self.confirm_window[-self.stat_size:]

                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=RuntimeWarning)
                                    st, self.p_value = stats.ks_2samp(most_recent, less_recent, method="auto", alternative='less')

                                p_values.append(self.p_value)

                            
                            corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

                            if any([p <= self.alpha for p in corrected_p_values]):
                                self.drift_confirmed = False
                                self._drift_detected = False
                                self.confirm_window = []
                                self.analisisPrevio = True
                                self.identificado_tipo = False

                    else: 
                        if self.identificado_tipo == False:
                            self._identificar_tipo_drift(self.confirm_window)
                            self.identificado_tipo = True
                            



    
       

    
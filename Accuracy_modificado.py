from collections import defaultdict, deque
import pandas as pd  # Para cálculo de suavizado EWMA
import numpy as np
from river.metrics import Accuracy

# ----------------------------------------------------------
# Clase ConfusionMatrix: almacena una matriz de confusión simple acumulada
# ----------------------------------------------------------
class ConfusionMatrix:
    """
    Implementación simple de una matriz de confusión acumulada,
    con contadores de verdaderos positivos y total de muestras evaluadas.
    """

    def __init__(self):
        self.matrix = defaultdict(lambda: defaultdict(int))
        self.total_true_positives = 0
        self.total_weight = 0

    def update(self, y_true, y_pred):
        """
        Registra una predicción comparándola con su valor real.

        Parámetros:
        - y_true: etiqueta verdadera.
        - y_pred: etiqueta predicha por el modelo.
        """
        self.matrix[y_true][y_pred] += 1
        if y_true == y_pred:
            self.total_true_positives += 1
        self.total_weight += 1

# ----------------------------------------------------------
# Clase WindowedConfusionMatrix: versión con ventana deslizante de tamaño fijo
# ----------------------------------------------------------
class WindowedConfusionMatrix:
    """
    Variante de matriz de confusión que mantiene una ventana móvil de observaciones,
    eliminando las más antiguas al alcanzar el límite definido.
    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.matrix = defaultdict(lambda: defaultdict(int))
        self.total_true_positives = 0
        self.total_weight = 0

    def update(self, y_true, y_pred):
        """
        Añade una nueva observación a la ventana y elimina la más antigua si es necesario.
        Actualiza contadores internos para el cálculo de Accuracy en ventana.

        Parámetros:
        - y_true: etiqueta verdadera.
        - y_pred: etiqueta predicha.
        """
        # Eliminar la observación más antigua si la ventana está llena
        if len(self.history) == self.window_size:
            old_y_true, old_y_pred = self.history.popleft()
            self.matrix[old_y_true][old_y_pred] -= 1
            if old_y_true == old_y_pred:
                self.total_true_positives -= 1
            self.total_weight -= 1

        # Añadir nueva observación
        self.history.append((y_true, y_pred))
        self.matrix[y_true][y_pred] += 1
        if y_true == y_pred:
            self.total_true_positives += 1
        self.total_weight += 1

# ----------------------------------------------------------
# Clase AccuracyModificado: calcula Accuracy con EWMA y/o ventana deslizante
# ----------------------------------------------------------
class AccuracyModificado(Accuracy):
    """
    Métrica de precisión (Accuracy) extendida con soporte para:
    - Ventana deslizante (para evaluar sólo las últimas N observaciones).
    - Suavizado exponencial (EWMA) para obtener una versión estable del rendimiento.

    Hereda de river.metrics.Accuracy.
    """

    def __init__(self, cm=None, window_size=None, span=300, adjust=False):
        """
        Constructor de la métrica.

        Parámetros:
        - cm: matriz de confusión personalizada (por defecto, acumulativa).
        - window_size (int): tamaño de la ventana si se desea usar una matriz deslizante.
        - span (int): parámetro del suavizado exponencial (más bajo = más sensible).
        - adjust (bool): si True, aplica corrección de sesgo en el suavizado EWMA.
        """
        super().__init__()

        if window_size is not None:
            self.cm = WindowedConfusionMatrix(window_size)
            self.accuracy_history = deque(maxlen=window_size)
        else:
            self.cm = cm if cm is not None else ConfusionMatrix()
            self.accuracy_history = []

        self.span = span
        self.adjust = adjust

    def update(self, y_true, y_pred):
        """
        Actualiza la métrica con una nueva predicción y su resultado.
        Calcula y guarda la precisión actual y su versión suavizada.

        Parámetros:
        - y_true: etiqueta verdadera.
        - y_pred: etiqueta predicha.
        """
        self.cm.update(y_true, y_pred)
        current_accuracy = self.get()
        self.accuracy_history.append(current_accuracy)
        self.smoothed_accuracy = self.get_smoothed_accuracy()
        return self

    def get(self):
        """
        Devuelve la precisión actual (proporción de aciertos).
        """
        try:
            return self.cm.total_true_positives / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0

    def get_smoothed_accuracy(self):
        """
        Devuelve la precisión suavizada usando EWMA sobre la serie histórica.
        """
        accuracies = pd.Series(self.accuracy_history)
        smoothed_accuracies = accuracies.ewm(span=self.span, adjust=self.adjust).mean()
        return smoothed_accuracies.iloc[-1]  # Último valor suavizado

    def __str__(self):
        """
        Representación en cadena de la precisión actual.
        """
        return f'Accuracy actual: {self.get() * 100:.2f}%'


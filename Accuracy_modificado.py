from collections import defaultdict, deque
import pandas as pd  # Importamos pandas para usar Series y EWMA
import numpy as np
from river.metrics import Accuracy

class WindowedConfusionMatrix:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.matrix = defaultdict(lambda: defaultdict(int))
        self.total_true_positives = 0
        self.total_weight = 0

    def update(self, y_true, y_pred):
        # Si la ventana está llena, eliminamos la observación más antigua
        if len(self.history) == self.window_size:
            old_y_true, old_y_pred = self.history.popleft()
            self.matrix[old_y_true][old_y_pred] -= 1
            
            if old_y_true == old_y_pred:
                self.total_true_positives -= 1
            self.total_weight -= 1

        # Agregamos la nueva observación
        self.history.append((y_true, y_pred))
        self.matrix[y_true][y_pred] += 1
        if y_true == y_pred:
            self.total_true_positives += 1
        self.total_weight += 1

class AccuracyModificado(Accuracy):
    def __init__(self, cm=None, window_size=None, span=300, adjust=False):

        super().__init__()

        if window_size is not None:
            self.cm = WindowedConfusionMatrix(window_size)
            self.accuracy_history = deque(maxlen=window_size)  # Pila para almacenar las precisiones en cada paso
        else:
            self.cm = cm if cm is not None else ConfusionMatrix()
            self.accuracy_history = []  # Lista para almacenar las precisiones en cada paso
        # self.accuracy_history = []  # Lista para almacenar las precisiones en cada paso
        self.span = span            # Parámetro para el suavizado EWMA
        self.adjust = adjust        # Parámetro adjust para EWMA



    def update(self, y_true, y_pred):
        self.cm.update(y_true, y_pred)
        current_accuracy = self.get()
        self.accuracy_history.append(current_accuracy)
        self.smoothed_accuracy = self.get_smoothed_accuracy()
        return self

    def get(self):
        try:
            return self.cm.total_true_positives / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0
        
    # def get_current(self):
    #     try:
    #         return self.cm.total_true_positives / self.cm.total_weight
    #     except ZeroDivisionError:
    #         return 0.0

    def get_smoothed_accuracy(self):
        accuracies = pd.Series(self.accuracy_history)
        smoothed_accuracies = accuracies.ewm(span=self.span, adjust=self.adjust).mean()
        return smoothed_accuracies.iloc[-1]  # Devolvemos el último valor suavizado

    # def get(self):
    #     accuracies = pd.Series(self.accuracy_history)
    #     smoothed_accuracies = accuracies.ewm(span=self.span, adjust=self.adjust).mean()
    #     return smoothed_accuracies.iloc[-1]  # Devolvemos el último valor suavizado

    # def __str__(self):
    #     return f'Accuracy actual: {self.get() * 100:.2f}%, Accuracy suavizado: {self.smoothed_accuracy * 100:.2f}%'

    def __str__(self):
        return f'Accuracy actual: {self.get() * 100:.2f}%'

class ConfusionMatrix:
    def __init__(self):
        self.matrix = defaultdict(lambda: defaultdict(int))
        self.total_true_positives = 0
        self.total_weight = 0

    def update(self, y_true, y_pred):
        self.matrix[y_true][y_pred] += 1
        if y_true == y_pred:
            self.total_true_positives += 1
        self.total_weight += 1

from __future__ import annotations

import random
from river.datasets.synth import Agrawal

class AgrawalIncDriftFunc(Agrawal): 

    """ Modificación del generador de datos de Agrawal para que pueda generar drift incremental. 
    Se parte de la funcion de clasificacion 6 de Agrawal, la cual es: 

        def _classification_function_6(
            salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
            disposable = 2 * (salary + commission) / 3 - loan / 5 - 20000
        return 0 if disposable > 1 else 1

    Por defecto, se genera el drift en la posicion 2000, con un ancho de 1000
    
    """

    def __init__(self, classification_function: int = 6, seed: int | None = None,
                 balance_classes: bool = False, perturbation: float = 0.0, 
                    position: int = 2000, width: int = 1000, revert_drift: bool = False, interpolacion86: bool = False, interpolacion87: bool = False):   
        if classification_function not in [6,7,8]:
            raise ValueError("classification_function debe ser 6,7, 8")
        super().__init__(classification_function, seed, balance_classes, perturbation)


        self.position = position
        self.width = width
        self.drift_rate = 1/width
        self.classification_function = classification_function
        self.indice_actual = 0  # Indice actual de la iteración
        self.rever_drift = revert_drift
        if revert_drift:
            self.drift_actual = 1.0
        else:
            self.drift_actual = 0.0
        self.interpolacion86 = interpolacion86
        self.interpolacion87 = interpolacion87

    def generar_drift(self):
        """
        Aplica un drift incremental a la función de clasificación.
        """
        if self.rever_drift:
            if self.drift_actual > 0.0:
                self.drift_actual -= self.drift_rate
            else:
                self.drift_actual = 0.0
        else:
            if self.drift_actual < 1.0:
                self.drift_actual += self.drift_rate
            else:
                self.drift_actual = 1.0  # Fija el valor máximo del drift a 1

    
    @staticmethod
    def _classification_function_6(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan, drift
    ):
        disposable = (2+drift*3) * (salary + commission)  / 3 - loan / 5 - 20000 

        # Aplica un drift en la 
        return 0 if disposable > 1 else 1
    

    
    @staticmethod
    def _func8_a_func6(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan, alpha
    ):
        """
        Función de interpolacion que cambia de forma incremental de la función 8 a la funcion 6, a traves de un parametro alpha.
        Se aplica una interpolación lineal a los valores de loan y elevel, ademas de la constante
        El parámetro alpha controla la transición:
        - alpha = 0 -> corresponde a _classification_function_8.
        - alpha = 1 -> corresponde a _classification_function_6.
        """


        elevel_alpha = (1 - alpha) * (5000 * elevel)  

        constante = (1-alpha) * 10000 + alpha * 20000

        # Calculamos el disposable interpolado.
        disposable = 2 * (salary + commission) / 3 - elevel_alpha - loan / 5 - constante



        # Clasificación final
        return 0 if disposable > 1 else 1
    
    @staticmethod
    def _func8_a_func7(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan, alpha
    ):
        """
        Función de interpolacion que cambia de forma incremental de la función 8 a la funcion 6, a traves de un parametro alpha.
        Se aplica una interpolación lineal a los valores de loan y elevel, ademas de la constante
        El parámetro alpha controla la transición:
        - alpha = 0 -> corresponde a _classification_function_8.
        - alpha = 1 -> corresponde a _classification_function_6.
        """


        loan_alpha = (1 - alpha) * (loan / 5)  

        constante = (1-alpha) * 10000 + alpha * 20000

        # Calculamos el disposable interpolado.
        disposable = 2 * (salary + commission) / 3 - elevel * 5000 - loan_alpha - constante



        # Clasificación final
        return 0 if disposable > 1 else 1



    @staticmethod
    def _classification_function_8(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan, drift
    ):
        disposable = 2 * (salary + commission) / 3 - 5000 * elevel * (1+drift) - loan / 5 - 10000
        return 0 if disposable > 1 else 1


    def __iter__(self):
        self._rng = random.Random(self.seed)
        self._next_class_should_be_zero = False

        while True:
            
            y = 0
            desired_class_found = False

            while not desired_class_found:
                salary = 20000 + 130000 * self._rng.random()
                commission = 0 if (salary >= 75000) else (10000 + 75000 * self._rng.random())
                age = self._rng.randint(20, 80)
                elevel = self._rng.randint(0, 4)
                car = self._rng.randint(1, 20)
                zipcode = self._rng.randint(0, 8)
                hvalue = (8 - zipcode) * 100000 * (0.5 + self._rng.random())
                hyears = self._rng.randint(1, 30)
                loan = self._rng.random() * 500000
                
                # Se pasa el drift actual a la función de clasificación
                if self.interpolacion86 == True:
                    y = self._func8_a_func6(
                        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan, self.drift_actual
                    )
                elif self.interpolacion87 == True:
                    y = self._func8_a_func7(
                        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan, self.drift_actual
                    )
                else:                         
                    y = self._classification_functions[self.classification_function](
                        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan, self.drift_actual
                    )
                
                if not self.balance_classes: 
                    desired_class_found = True
                else:
                    if (self._next_class_should_be_zero and (y == 0)) or (
                        (not self._next_class_should_be_zero) and (y == 1)
                    ):
                        desired_class_found = True
                        self._next_class_should_be_zero = not self._next_class_should_be_zero

            if self.perturbation > 0.0:
                salary = self._perturb_value(salary, 20000, 150000)
                if commission > 0:
                    commission = self._perturb_value(commission, 10000, 75000)
                age = round(self._perturb_value(age, 20, 80))
                hvalue = self._perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)
                hyears = round(self._perturb_value(hyears, 1, 30))
                loan = self._perturb_value(loan, 0, 500000)

            x = dict()
            for feature in self.feature_names:
                x[feature] = eval(feature)

            yield x, y
            
            self.indice_actual += 1

            if(self.indice_actual >= self.position and self.indice_actual <= (self.position + self.width)):
                self.generar_drift()



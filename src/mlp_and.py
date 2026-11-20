from sklearn.neural_network import MLPClassifier


def entrenar_mlp_and(max_iter: int = 2000, random_state: int = 42) -> MLPClassifier:
    """Entrena un Perceptrón Multicapa (MLP) para la compuerta lógica AND."""
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]

    mlp = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="logistic",
        max_iter=max_iter,
        random_state=random_state,
    )
    mlp.fit(X, y)
    return mlp


def predecir_and(modelo: MLPClassifier, entradas: list[list[int]]) -> list[int]:
    """Genera predicciones binarias a partir de un modelo MLP entrenado."""
    return [int(pred) for pred in modelo.predict(entradas)]

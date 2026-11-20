from src.mlp_and import entrenar_mlp_and, predecir_and


def test_mlp_and_converge_y_predice_tabla_verdad() -> None:
    entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
    esperado = [0, 0, 0, 1]

    modelo = entrenar_mlp_and()
    resultado = predecir_and(modelo, entradas)

    assert resultado == esperado

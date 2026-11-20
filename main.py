from src.mlp_and import entrenar_mlp_and, predecir_and


def main() -> None:
    entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
    modelo = entrenar_mlp_and()
    predicciones = predecir_and(modelo, entradas)

    print("=== Perceptrón Multicapa (MLP) - Compuerta AND ===")
    for entrada, prediccion in zip(entradas, predicciones):
        print(f"Entrada: {entrada} - Predicción: {prediccion}")


if __name__ == "__main__":
    main()

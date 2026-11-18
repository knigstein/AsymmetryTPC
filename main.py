import os
import numpy as np
from src.model import ThreePhaseCircuitModel
from src.visualize import plot_vector_diagrams


def format_phasor(z):
    return f"{abs(z):.3f} degree {np.degrees(np.angle(z)):.2f}°"


def main():
    os.makedirs("results", exist_ok=True)
    circuit = ThreePhaseCircuitModel(E_phase=220.0)

    # Симметричный режим
    res = circuit.solve(Z_A=1e6, Z_B=1e6, Z_C=1e6, Z_G=1e6)

    print("=== Результаты расчёта симметричного режима ===")
    for idx, phase in enumerate(["A", "B", "C"]):
        print(
            f"Фаза {phase}: U = {format_phasor(res['U_phase'][idx])} В | "
            f"I_дв = {format_phasor(res['I_motor'][idx])} А"
        )

    U1, U2, U0 = res["U_seq"]
    print(f"Симметричные составляющие U: |U1|={abs(U1):.3f} В, |U2|={abs(U2):.2e} В, |U0|={abs(U0):.2e} В")
    print(f"Суммарная активная мощность P = {res['P_total_W'] / 1000:.3f} кВт")

    output_img = "results/symmetric_vectors.png"
    plot_vector_diagrams(res, save_path=output_img)
    print(f"Векторная диаграмма сохранена в: {output_img}")


if __name__ == "__main__":
    main()

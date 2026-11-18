import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_vector_diagrams(result, save_path="results/symmetric_vectors.png"):
    """Построение векторных диаграмм фазных напряжений и токов двигателя"""
    U_phase = result["U_phase"]
    I_motor = result["I_motor"]

    labels = ["Фаза A", "Фаза B", "Фаза C"]
    colors = ["#E69F00", "#009E73", "#D55E00"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), subplot_kw={"projection": "polar"})
    fig.suptitle("Векторные диаграммы симметричного режима (АИР100L4)", fontsize=13, fontweight="bold")

    # 1) Напряжения на зажимах двигателя
    ax_u = axes[0]
    u_max = np.max(np.abs(U_phase)) * 1.18
    ax_u.set_ylim(0, u_max)
    for vec, label, color in zip(U_phase, labels, colors):
        angle = np.angle(vec)
        mag = np.abs(vec)
        ax_u.annotate(
            "", xy=(angle, mag), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.2)
        )
        ax_u.plot(angle, mag, "o", color=color, label=f"{label}: {mag:.2f} В ({np.degrees(angle):.1f}°)")
    ax_u.set_title("Фазные напряжения двигателя, В", pad=14)
    ax_u.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08), fontsize=8.5)

    # 2) Токи в обмотках двигателя
    ax_i = axes[1]
    i_max = np.max(np.abs(I_motor)) * 1.18
    ax_i.set_ylim(0, i_max)
    for vec, label, color in zip(I_motor, labels, colors):
        angle = np.angle(vec)
        mag = np.abs(vec)
        ax_i.annotate(
            "", xy=(angle, mag), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.2)
        )
        ax_i.plot(angle, mag, "o", color=color, label=f"{label}: {mag:.3f} А ({np.degrees(angle):.1f}°)")
    ax_i.set_title("Фазные токи двигателя, А", pad=14)
    ax_i.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08), fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

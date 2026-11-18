import numpy as np


class ThreePhaseCircuitModel:
    """Модель трёхфазной цепи с асинхронным двигателем"""

    def __init__(self, E_phase=220.0):
        self.E_A = complex(E_phase, 0.0)
        self.a = np.exp(1j * 2 * np.pi / 3)
        self.a2 = self.a ** 2

        # Сопротивления источника, Ом)
        self.Z_src1 = complex(0.05, 0.15)
        self.Z_src2 = complex(0.05, 0.15)
        self.Z_src0 = complex(0.05, 0.15)

        # Кабель АВВГ 4x16, 100 м.
        self.Z_line1 = complex(0.198, 0.00675)
        self.Z_line2 = self.Z_line1
        self.Z_line0 = 4 * self.Z_line1  # Z_ж + 3*Z_N

        # Двигатель АИР100L4
        self.Z_m1 = complex(21.000, 13.565)
        self.Z_m2 = complex(1.429, 3.273)

    def equivalent_parameters(self):
        """Расчёт параметров эквивалентного генератора на зажимах узла несимметрии"""
        z_left1 = self.Z_src1 + self.Z_line1
        z_e1 = (z_left1 * self.Z_m1) / (z_left1 + self.Z_m1)
        e_e1 = self.E_A * (self.Z_m1 / (z_left1 + self.Z_m1))

        z_left2 = self.Z_src2 + self.Z_line2
        z_e2 = (z_left2 * self.Z_m2) / (z_left2 + self.Z_m2)

        z_e0 = self.Z_src0 + self.Z_line0
        return e_e1, z_e1, z_e2, z_e0

    def solve(self, Z_A=1e6, Z_B=1e6, Z_C=1e6, Z_G=1e6):
        """Решение системы уравнений A * X = B для заданных сопротивлений узла"""
        e_e1, z_e1, z_e2, z_e0 = self.equivalent_parameters()
        a, a2 = self.a, self.a2

        A = np.array([
            [1,  1,  1, -Z_A,     -Z_A,    -(Z_A + 3 * Z_G)],
            [a2, a,  1, -a2 * Z_B, -a * Z_B, -(Z_B + 3 * Z_G)],
            [a,  a2, 1, -a * Z_C,  -a2 * Z_C, -(Z_C + 3 * Z_G)],
            [1,  0,  0,  z_e1,     0,       0],
            [0,  1,  0,  0,        z_e2,    0],
            [0,  0,  1,  0,        0,       z_e0],
        ], dtype=complex)

        B = np.array([0, 0, 0, e_e1, 0, 0], dtype=complex)
        U1, U2, U0, I1_n, I2_n, I0_n = np.linalg.solve(A, B)

        # Фазовые напряжения на двигателе
        U_phase = np.array([
            U1 + U2 + U0,
            a2 * U1 + a * U2 + U0,
            a * U1 + a2 * U2 + U0,
        ])

        # Токи в двигателе
        I_m1 = U1 / self.Z_m1
        I_m2 = U2 / self.Z_m2
        I_m0 = complex(0.0, 0.0)
        I_motor = np.array([
            I_m1 + I_m2 + I_m0,
            a2 * I_m1 + a * I_m2 + I_m0,
            a * I_m1 + a2 * I_m2 + I_m0,
        ])

        # Токи в несимметричном узле
        I_node = np.array([
            I1_n + I2_n + I0_n,
            a2 * I1_n + a * I2_n + I0_n,
            a * I1_n + a2 * I2_n + I0_n,
        ])

        # Активная мощность двигателя, Вт
        P_total = sum(
            (u * np.conj(i)).real for u, i in zip(U_phase, I_motor)
        )

        return {
            "U_seq": (U1, U2, U0),
            "I_node_seq": (I1_n, I2_n, I0_n),
            "I_motor_seq": (I_m1, I_m2, I_m0),
            "U_phase": U_phase,
            "I_motor": I_motor,
            "I_node": I_node,
            "P_total_W": P_total,
        }

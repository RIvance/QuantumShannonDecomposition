from __future__ import annotations

import math
from abc import ABC
from copy import deepcopy
from typing import *

import numpy as np
import scipy.linalg
from numpy.typing import *

Mat = NDArray[complex]
StateVec = NDArray[complex]
Tensor = NDArray[complex]
Qubit = int


class Gate(ABC):
    ident: str
    mat: Mat

    def __init__(self, mat: Mat, ident: str):
        assert mat.shape[0] == mat.shape[1]
        self.mat = mat
        self.ident = ident

    def __str__(self) -> str:
        return self.ident

    @property
    def dim(self) -> int:
        return self.mat.shape[0]

    @property
    def size(self) -> int:
        return int(math.log(self.dim, 2))

    @property
    def args(self) -> List[int | float]:
        return list()


class Operation:
    gate: Gate
    targets: List[Qubit]

    def __init__(self, gate: Gate, targets: List[Qubit]) -> None:
        assert gate.size == len(targets)
        self.gate = gate
        self.targets = targets

    def __str__(self) -> str:
        return f"{self.gate}{'' if len(self.gate.args) == 0 else self.gate.args}{self.targets}"


class Circuit:
    operations: List[Operation]

    def __init__(self, operations: List[Operation] | None = None):
        if operations is None:
            self.operations = list()
        else:
            self.operations = operations

    def __str__(self) -> str:
        return "\n".join(map(str, self.operations))

    def __add__(self, other: Circuit) -> Circuit:
        return Circuit(self.operations + other.operations)

    def map_qubits(self, qubits: List[Qubit]) -> Circuit:
        circuit = deepcopy(self)
        for operation in circuit.operations:
            for i in range(len(operation.targets)):
                operation.targets[i] = qubits[operation.targets[i]]
        return circuit

    def append(self, gate: Gate, targets: List[Qubit] | Qubit | None = None):
        if targets is None or type(targets) == int:
            assert gate.size == 1
            targets = [0]
        self.operations.append(Operation(gate, targets))

    def insert(self, gate: Gate, targets: List[Qubit] | None = None, index: int | None = None):
        if index is None:
            index = 0
        if targets == 0:
            assert gate.size == 1
            self.operations.insert(index, Operation(gate, [0]))
        else:
            self.operations.insert(index, Operation(gate, targets))


class Unitary(Gate):

    def __init__(self, mat: Mat, ident: str):
        super().__init__(mat, ident)
        self.mat = mat


class PauliX(Gate):

    def __init__(self):
        super().__init__(
            np.array(
                [
                    [0, 1],
                    [1, 0],
                ]
            ), "X"
        )

    pass


class PauliY(Gate):

    def __init__(self):
        super().__init__(
            np.array(
                [
                    [0, -1j],
                    [1j, 0],
                ]
            ), "Y"
        )

    pass


class PauliZ(Gate):

    def __init__(self):
        super().__init__(
            np.array(
                [
                    [1, 0],
                    [0, -1],
                ]
            ), "Z"
        )

    pass


class Rx(Gate):
    angle: float

    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(
            np.array(
                [
                    [np.cos(angle / 2), -1j * np.sin(angle / 2)],
                    [-1j * np.sin(angle / 2), np.cos(angle / 2)],
                ]
            ), "RX"
        )

    @property
    def args(self) -> List[int | float]:
        return [self.angle]

    pass


class Ry(Gate):
    angle: float

    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(
            np.array(
                [
                    [np.cos(angle / 2), -np.sin(angle / 2)],
                    [np.sin(angle / 2), np.cos(angle / 2)],
                ]
            ), "RY"
        )

    @property
    def args(self) -> List[int | float]:
        return [self.angle]

    pass


class Rz(Gate):
    angle: float

    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(
            np.array(
                [
                    [np.exp(-1j * angle / 2), 0],
                    [0, np.exp(+1j * angle / 2)],
                ]
            ), "RZ"
        )

    @property
    def args(self) -> List[int | float]:
        return [self.angle]

    pass


class Phase(Gate):

    def __init__(self, angle: float):
        super().__init__(
            np.array(
                [
                    [1, 0],
                    [0, np.exp(1j * angle)],
                ]
            ), "P"
        )

    pass


class CNot(Gate):

    def __init__(self):
        super().__init__(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ), "CX"
        )

    pass


def close_to_zero(value: float | complex):
    return np.isclose(value, 0.0)


def su(mat: Mat) -> Mat:
    rank: int = mat.shape[0]
    return mat / (np.linalg.det(mat) ** (1 / rank))


def is_pow_of_2(x: int) -> bool:
    return (x & (x - 1) == 0) and x != 0


def zyz_decompose(mat: Mat) -> Circuit:
    """
    Decompose a 1-qubit gate to sequence of Z, Y, and Z rotations, and a phase
    :param mat:
    :return:
    """

    U: Mat = su(mat)

    rank: int = mat.shape[0]
    alpha = np.linalg.det(mat) ** (1 / rank)

    # φ = {
    #  2arccos(|U_00|) if |U_00| >= |U_01|
    #  2arcsin(|U_01|) if |U_00| <  |U_01|
    # }
    phi: float
    phase: float

    if abs(U[0, 0]) > abs(U[1, 0]):
        phi = -2 * np.arccos(min(abs(U[0, 0]), 1))
    else:
        phi = -2 * np.arcsin(min(abs(U[1, 0]), 1))

    cos_half_phi = np.cos(phi / 2)

    theta_plus_lambda: float
    theta_minus_lambda: float

    if not close_to_zero(cos_half_phi):
        # if cos(φ / 2) != 0
        #   then θ + λ = 2arctan2(Im(U_11 / cos(φ/2), Re(U_11 / cos(φ/2))
        phase = U[1, 1] / cos_half_phi
        theta_plus_lambda = 2 * np.arctan2(np.imag(phase), np.real(phase))
    else:
        # if cos(φ / 2) == 0 then θ + λ = 0
        theta_plus_lambda = 0.0

    sin_half_phi: float = np.sin(phi / 2)

    if not close_to_zero(sin_half_phi):
        # if sin(φ*1/2) != 0
        #   then θ + λ = 2arctan2(Im(U_10 / sin(φ/2), Re(U_10 / sin(φ/2))
        phase = U[1, 0] / sin_half_phi
        theta_minus_lambda = 2 * np.arctan2(np.imag(phase), np.real(phase))
    else:
        # if sin(φ*1/2) == 0 then θ - λ = 0
        theta_minus_lambda = 0.0

    # θ = ((θ + λ) + (θ - λ)) / 2
    theta = (theta_plus_lambda + theta_minus_lambda) / 2
    # λ = ((θ + λ) - (θ - λ)) / 2
    lam = (theta_plus_lambda - theta_minus_lambda) / 2

    # α' = α - (θ + φ + λ) / π
    # We just ignore the global phase here
    alpha -= (theta + phi + lam) / np.pi

    if np.isclose(phi, 0.0):
        lam = theta + lam
        phi = 0.0
        theta = 0.0

    circuit = Circuit()
    circuit.append(Rz(lam))
    circuit.append(Ry(phi))
    circuit.append(Rz(theta))

    return circuit


# def double_tensor_product_decompose(mat: Mat) -> Circuit:
#     """
#     Decompose U = Ul⊗Ur where U in SU(4), and Ul, Ur in SU(2).
#     :param mat:
#     :return:
#     """
#     R: Mat = mat[:2, :2].copy()
#     detR: float = np.linalg.det(R)
#     if abs(R) < 0.1:
#         R = mat[2:, :2].copy()
#         detR = np.linalg.det(R)
#     assert abs(R) >= 0.1
#     R /= np.sqrt(detR)
#
#     # | R   |
#     # |   R |
#     tmp: Mat = np.kron(np.eye(2), R.T.conj())
#     tmp = mat.dot(tmp)
#     L: Mat = tmp[::2, ::2]
#     detL: float = np.linalg.det(L)
#     assert abs(detL) >= 0.9
#     L /= np.sqrt(detL)
#     phase: float = cmath.phase(detL) / 2
#
#     tmp = np.kron(L, R)
#     assert abs(abs(tmp.conj().T.dot(mat).trace()) - 4) > 1e-13
#
#     left_circuit = zyz_decompose(L)
#     right_circuit = zyz_decompose(R)
#
#     left_circuit.insert(Phase(phase))
#
#     return left_circuit + right_circuit.map_qubits([1])


def is_hermitian(mat: Mat) -> bool:
    return np.allclose(mat, mat.conj().T)


def quantum_shannon_decomposition(mat: Mat) -> Circuit:
    """
           ┌───┐               ┌───┐     ┌───┐     ┌───┐
          ─┤   ├─       ───────┤ Rz├─────┤ Ry├─────┤ Rz├─────
           │   │    ≃     ┌───┐└─┬─┘┌───┐└─┬─┘┌───┐└─┬─┘┌───┐
         /─┤   ├─       /─┤   ├──□──┤   ├──□──┤   ├──□──┤   ├
           └───┘          └───┘     └───┘     └───┘     └───┘
    :param mat:
    :return:
    """
    dim: int = mat.shape[0]
    size = int(np.log2(dim))

    circuit = Circuit()

    if size == 1:
        # circuit = zyz_decompose(mat)
        circuit.append(Unitary(mat, "Unitary"))
    else:
        """
        Use cos-sin decomposition first
              ┌───┐               ┌───┐
            ──┤   ├──      ────□──┤ Ry├──□───
              │ U │    =     ┌─┴─┐└─┬─┘┌─┴─┐
            /─┤   ├──      /─┤ U ├──□──┤ V ├─
              └───┘          └───┘     └───┘
        """
        (u1, u2), ry_angles, (v1h, v2h) = scipy.linalg.cossin(mat, separate=True, p=dim // 2, q=dim // 2)

        # left circuit
        circuit += demultiplex(v1h, v2h, 0, list(range(1, size)))

        # middle circuit: uniformly ctrl ry
        # TODO: optimization: see [Appendix A: Additional Circuit Optimizations]
        circuit += demultiplex_pauli_rotation('Y', 2 * ry_angles, 0, list(range(1, size)))

        # right circuit
        circuit += demultiplex(u1, u2, 0, list(range(1, size)))

    return circuit


def demultiplex(u1: Mat, u2: Mat, ctrl: Qubit, targets: List[Qubit]) -> Circuit:
    """
    Decompose a multiplexor defined by a pair of unitary matrices operating on the same subspace.

        That is, decompose

            ctrl     ────□────
                      ┌──┴──┐
            target  /─┤     ├─
                      └─────┘

        represented by the block diagonal matrix

                ┏         ┓
                ┃ U1      ┃
                ┃      U2 ┃
                ┗         ┛

        to
                          ┌───┐
           ctrl    ───────┤ Rz├──────
                     ┌───┐└─┬─┘┌───┐
           target  /─┤ W ├──□──┤ V ├─
                     └───┘     └───┘

        by means of simultaneous unitary diagonalization.

    :param u1: applied if the control qubit is |0>
    :param u2: applied if the control qubit is |1>
    :param ctrl:
    :param targets:
    :return: Circuit, composed of 1-qubit gates and CNOT gates.
    """
    assert u1.shape == u2.shape
    dim: int = u1.shape[0]
    size = int(np.log2(dim))
    assert size == len(targets)

    u1_u2h = u1 @ u2.T.conj()
    if is_hermitian(u1_u2h):
        eigen_vals, vmat = scipy.linalg.eigh(u1_u2h)
    else:
        eigen_vals, vmat = scipy.linalg.schur(u1_u2h, output="complex")
        eigen_vals = eigen_vals.diagonal()
    dvals = np.sqrt(eigen_vals.astype(complex))
    dmat = np.diag(dvals)
    wmat = dmat @ vmat.T.conjugate() @ u2
    angles: List[float] = (2 * np.angle(dvals.conj())).tolist()

    circuit = Circuit()
    circuit += quantum_shannon_decomposition(wmat).map_qubits(targets)
    circuit += demultiplex_pauli_rotation('Z', angles, ctrl, targets)
    circuit += quantum_shannon_decomposition(vmat).map_qubits(targets)

    return circuit


def demultiplex_pauli_rotation(axis: str, angles: List[float] | Mat, target: Qubit, ctrls: List[Qubit]) -> Circuit:
    """
    Decompose a Pauli-rotation (RY or RZ) multiplexor defined by 2^(n-1) rotation angles.

         ────□───        ─────────●─────────●────
             │                    │         │
         ─/──□───   ==   ─/──□────┼────□────┼──/─
             │               │    │    │    │
         ────R───        ────R────X────R────X────

    :param axis: axis of the pauli rotation, 'Y' or 'Z'
    :param angles:
    :param target: target qubit
    :param ctrls: ctrl qubits
    :return: Circuit, composed of 1-qubit Pauli-rotation gates and CNOT gates.
    """

    def PauliRot(angle: float) -> Gate:
        assert axis in ['X', 'Y', 'Z']
        if axis == 'X':
            return Rx(float(angle))
        elif axis == 'Y':
            return Ry(float(angle))
        elif axis == 'Z':
            return Rz(float(angle))

    assert len(angles) == 2 ** len(ctrls)
    circuit = Circuit()

    if len(ctrls) == 1:
        circuit.append(PauliRot((angles[0] + angles[1]) / 2), target)
        circuit.append(CNot(), [ctrls[0], target])
        circuit.append(PauliRot((angles[0] - angles[1]) / 2), target)
        circuit.append(CNot(), [ctrls[0], target])
    elif len(ctrls) == 2:
        (s0, s1), (t0, t1) = calc_pauli_demultiplex_angles(angles)
        circuit.append(PauliRot(s0), target)
        circuit.append(CNot(), [ctrls[1], target])
        circuit.append(PauliRot(s1), target)
        circuit.append(CNot(), [ctrls[0], target])
        circuit.append(PauliRot(t1), target)
        circuit.append(CNot(), [ctrls[1], target])
        circuit.append(PauliRot(t0), target)
        circuit.append(CNot(), [ctrls[0], target])
    else:
        (s0, s1), (t0, t1) = calc_pauli_demultiplex_angles(angles)
        circuit += demultiplex_pauli_rotation(axis, s0, target, ctrls[2:])
        circuit.append(CNot(), [ctrls[1], target])
        circuit += demultiplex_pauli_rotation(axis, s1, target, ctrls[2:])
        circuit.append(CNot(), [ctrls[0], target])
        circuit += demultiplex_pauli_rotation(axis, t1, target, ctrls[2:])
        circuit.append(CNot(), [ctrls[1], target])
        circuit += demultiplex_pauli_rotation(axis, t0, target, ctrls[2:])
        circuit.append(CNot(), [ctrls[0], target])

    return circuit


def calc_pauli_demultiplex_angles(angles: List[float]) -> Tuple[
    Tuple[float | Mat, float | Mat], Tuple[float | Mat, float | Mat]
]:
    """
    Calculation rotation angles for two-level decomposing of a Pauli-rotation multiplexor.

    Reshape `rads` into a blocked matrix in presentation of

        ┏                           ┓
        ┃ θ_{00}                    ┃
        ┃                           ┃
        ┃       θ_{01}              ┃
        ┃                           ┃
        ┃             θ_{10}        ┃
        ┃                           ┃
        ┃                   θ_{11}  ┃
        ┗                           ┛

    Then calculate `φ`

        ┏           ┓         ┏              ┓         ┏              ┓
        ┃ φ_0       ┃         ┃ θ_{00}       ┃         ┃ θ_{10}       ┃
        ┃           ┃ = 1/2 * ┃              ┃ + 1/2 * ┃              ┃
        ┃       φ_1 ┃         ┃       θ_{01} ┃         ┃       θ_{11} ┃
        ┗           ┛         ┗              ┛         ┗              ┛

    and `λ`

        ┏           ┓         ┏              ┓         ┏              ┓
        ┃ λ_0       ┃         ┃ θ_{00}       ┃         ┃ θ_{10}       ┃
        ┃           ┃ = 1/2 * ┃              ┃ - 1/2 * ┃              ┃
        ┃       λ_1 ┃         ┃       θ_{01} ┃         ┃       θ_{11} ┃
        ┗           ┛         ┗              ┛         ┗              ┛

    Finally, decompose multiplexors in presentation of `φ` and `λ`, respectively.

    :param angles:
    :return:
    """
    dim: int = len(angles)
    thetas: Tensor = np.reshape(np.array(angles), [2, 2, int(dim / 2 / 2)])
    p0 = (thetas[0, 0, :] + thetas[1, 0, :]) / 2
    p1 = (thetas[0, 1, :] + thetas[1, 1, :]) / 2
    l0 = (thetas[0, 0, :] - thetas[1, 0, :]) / 2
    l1 = (thetas[0, 1, :] - thetas[1, 1, :]) / 2
    return ((p0 + p1) / 2, (p0 - p1) / 2), ((l0 + l1) / 2, (l0 - l1) / 2)


def generate_unitary(psi):
    dim = len(psi)  # Dimension of the state |ψ⟩

    # Normalize the state vector
    psi_normalized = psi / np.linalg.norm(psi)

    # Create the unitary matrix
    U = np.zeros((dim, dim), dtype=complex)
    U[:, 0] = psi_normalized  # Set the first column of U as the normalized state vector

    # Fill the remaining columns with arbitrary orthonormal vectors
    for i in range(1, dim):
        v = np.abs(np.random.randn(dim)).astype(complex)  # Generate a random vector
        v -= np.dot(U[:, :i], np.conj(U[:, :i]).T.dot(v))  # Orthogonalize with respect to the previous columns
        v /= np.linalg.norm(v)  # Normalize the vector
        U[:, i] = v

    return U.astype(complex)


def is_unitary(mat: Mat) -> bool:
    return np.isclose(mat @ mat.T.conj(), np.identity(mat.shape[0])).all()


def neighbor_cnot(ctrl: int, target: int) -> Circuit:
    assert ctrl != target
    circuit = Circuit()
    size = abs(ctrl - target) + 1
    for _ in range(2):
        for i in range(size - 1):
            circuit.append(CNot(), [i, i + 1])
        for i in range(size - 3, 0, -1):
            circuit.append(CNot(), [i, i + 1])
    if ctrl < target:
        return circuit.map_qubits(list(range(ctrl, target + 1)))
    else:
        return circuit.map_qubits(list(range(ctrl, target - 1, -1)))


def replace_long_range_cnot(circuit: Circuit) -> Circuit:
    replaced = Circuit()
    for operation in circuit.operations:
        if type(operation.gate) == CNot:
            if abs(operation.targets[0] - operation.targets[1]) > 1:
                replaced += neighbor_cnot(operation.targets[0], operation.targets[1])
            else:
                replaced.operations.append(operation)
        else:
            replaced.operations.append(operation)
    return replaced

from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np
import rotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DTYPE = np.complex128

def num_qubits(N: int) -> int:
    """Number of qubits n such that N == 2^n (N is the unitary / two-level size)."""
    return int(np.log2(N))


# ---------------------------------------------------------------------------
# Gate representations
# ---------------------------------------------------------------------------


@dataclass
class TwoLevel:
    """A two-level unitary: acts as the 2x2 `unitary` on the two basis states
    `level0`, `level1` of a size-`size` register, and as identity everywhere else.
    """

    size: int
    level0: int
    level1: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        U = np.eye(self.size, dtype=DTYPE)
        i = self.level0
        j = self.level1
        u = self.unitary
        U[i, i] = u[0, 0]
        U[i, j] = u[0, 1]
        U[j, i] = u[1, 0]
        U[j, j] = u[1, 1]
        return U


@dataclass
class SingleQubitGate:
    """A single-qubit gate acting as the 2x2 `unitary` on `qubit` of an n-qubit
    register (N = 2^n), identity on the other qubits.
    """

    n: int
    qubit: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        N = 2 ** self.n
        U = np.eye(N, dtype=DTYPE)
        u = self.unitary

        target_bit = self.n - 1 - self.qubit
        for i in range(N):
            if ((i >> target_bit) & 1) == 0:
                j = i ^ (1 << target_bit)
                U[i, i] = u[0, 0]
                U[i, j] = u[0, 1]
                U[j, i] = u[1, 0]
                U[j, j] = u[1, 1]
        return U


@dataclass
class ControlledU:
    """A fully-controlled single-qubit gate C^k(U): apply the 2x2 `unitary` to
    `target` iff every other qubit is 1. Controls are always conditioned on 1, so
    their positions need not be stored.
    """

    n: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        N = 2 ** self.n
        U = np.eye(N, dtype=DTYPE)
        j = N - 1
        target_bit = self.n - 1 - self.target
        i = j ^ (1 << target_bit)
        u = self.unitary
        U[i, i] = u[0, 0]
        U[i, j] = u[0, 1]
        U[j, i] = u[1, 0]
        U[j, j] = u[1, 1]
        return U


@dataclass
class CU:
    """A singly-controlled single-qubit gate C(U): apply the 2x2 `unitary` to
    `target` iff `control` is 1.
    """

    n: int
    control: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:

        N=2**self.n

        U=np.eye(N,dtype=DTYPE)

        cb=1<<(self.n-1-self.control)
        tb=1<<(self.n-1-self.target)

        for i in range(N):
            if (i&cb) and not(i&tb):
                j=i|tb
                U[i,i]=self.unitary[0,0]
                U[i,j]=self.unitary[0,1]
                U[j,i]=self.unitary[1,0]
                U[j,j]=self.unitary[1,1]
        return U


@dataclass
class CNOT:
    """A controlled-NOT: flip `target` iff `control` is 1."""

    n: int
    control: int
    target: int
    def to_unitary(self) -> np.ndarray:
        N=2**self.n
        U=np.eye(N,dtype=DTYPE)
        cb=1<<(self.n-1-self.control)
        tb=1<<(self.n-1-self.target)
        for i in range(N):
            if (i&cb) and not(i&tb):
                j=i|tb
                U[i,i]=0
                U[j,j]=0
                U[i,j]=1
                U[j,i]=1
        return U


@dataclass
class Swap:
    """A multi-controlled NOT (generalized Toffoli): flip `target` iff every other
    qubit equals its entry in `control_vals`.
    """

    target: int
    control_vals: list[bool]


Gate = Union[TwoLevel, SingleQubitGate, ControlledU, CU, CNOT]
Circuit = list
TwoLevels = list


def circuit_to_unitary(circuit: Circuit) -> np.ndarray:
    N = circuit[0].to_unitary().shape[0]
    U = np.eye(N, dtype=DTYPE)
    for g in circuit:
        U = g.to_unitary() @ U
    return U


def to_circuit(two_levels: TwoLevels) -> Circuit:
    return list(two_levels)


def error_up_to_phase(a: np.ndarray, b: np.ndarray) -> float:
    overlap = np.sum(np.conjugate(b) * a)
    if np.abs(overlap) == 0:
        return np.linalg.norm(a - b)
    phase = overlap / np.abs(overlap)
    b_aligned = b * np.conjugate(phase)
    return np.linalg.norm(a - b_aligned)


# ---------------------------------------------------------------------------
# Stage 1: Unitary -> two-level unitaries
# ---------------------------------------------------------------------------


def align(x: complex, y: complex, norm: float) -> np.ndarray:
    if norm == 0:
        return np.eye(2, dtype=DTYPE)
    U = np.array([[np.conjugate(x), np.conjugate(y)], [-y, x]], dtype=DTYPE)
    return U / norm


def decompose_vector(vec: np.ndarray) -> TwoLevels:
    vec = vec.astype(DTYPE).copy()
    n = len(vec)
    two_levels = []

    for i in range(n - 1, 0, -1):
        x = vec[i - 1]
        y = vec[i]
        norm = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2)
        if norm == 0:
            continue

        U = align(x, y, norm)
        vec[i-1:i+1] = U @ vec[i-1:i+1]

        tl = TwoLevel(size=n, level0=i - 1, level1=i, unitary=U)
        two_levels.append(tl)

    return two_levels


def expand_twolevels(two_levels: TwoLevels, full_size: int) -> TwoLevels:
    if not two_levels:
        return []

    offset = full_size - two_levels[0].size

    out = []

    for tl in two_levels:
        out.append(
            TwoLevel(
                size=full_size,
                level0=tl.level0 + offset,
                level1=tl.level1 + offset,
                unitary=tl.unitary,
            )
        )

    return out


def two_levels_to_unitary(two_levels: TwoLevels) -> np.ndarray:
    N = two_levels[0].size
    U = np.eye(N, dtype=DTYPE)
    for tl in two_levels:
        U = tl.to_unitary() @ U
    return U


def adjoint_twolevel(tl: TwoLevel) -> TwoLevel:
    return TwoLevel(
        size=tl.size, level0=tl.level0, level1=tl.level1, unitary=tl.unitary.conj().T
    )


def adjoint_twolevels(two_levels: TwoLevels) -> TwoLevels:
    return [adjoint_twolevel(tl) for tl in reversed(two_levels)]


def decompose_unitary(u: np.ndarray) -> TwoLevels:
    if len(u) == 1:
        return []
    vec = u[:,0]
    N = u.shape[0]
    tls = decompose_vector(vec)
    P = two_levels_to_unitary(tls)
    u = P @ u
    sub = u[1:,1:]
    sub_tls = decompose_unitary(sub)
    if sub_tls:
        expanded = expand_twolevels(sub_tls, N)
        u = two_levels_to_unitary(expanded) @ u
    else:
        expanded = []
    result = []
    result.extend(tls)
    result.extend(expanded)
    phase = u[-1,-1]
    if not np.isclose(phase,1):
        block = np.eye(2, dtype = DTYPE)
        block[1,1] = np.conjugate(phase) / np.abs(phase)
        result.append(
            TwoLevel(
                size = N,
                level0=N-2,
                level1=N-1,
                unitary=block
            )
        )
    return result


def twolevel_decomposition(u: np.ndarray) -> TwoLevels:
    return adjoint_twolevels(decompose_unitary(u))


# ---------------------------------------------------------------------------
# ABC decomposition of a single-qubit gate
# ---------------------------------------------------------------------------
@dataclass
class ABC:
    """Nielsen & Chuang Corollary 4.2: every single-qubit U factors as
    U = e^{i alpha} A X B X C with A B C = I (X is Pauli-X). Building block for a
    single-controlled C(U).
    """

    alpha: float  # global phase
    A: np.ndarray  # (2, 2)
    B: np.ndarray  # (2, 2)
    C: np.ndarray  # (2, 2)

def abc_decompose(u: np.ndarray) -> ABC:
    alpha, beta, gamma, delta = rotation.euler_angles_zyz(u)
    A = rotation.Rz(beta) @ rotation.Ry(gamma / 2)
    B = rotation.Ry(-gamma / 2) @ rotation.Rz(-(delta + beta) / 2)
    C = rotation.Rz((delta - beta) / 2)
    return ABC(alpha=alpha, A=A, B=B, C=C)


def abc_reconstruct(d: ABC) -> np.ndarray:
    X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
    return np.exp(1j * d.alpha) * d.A @ X @ d.B @ X @ d.C


# ---------------------------------------------------------------------------
# Gray code and controlled circuits
# ---------------------------------------------------------------------------


def gray_code(tl):
    n = num_qubits(tl.size)

    current = tl.level0
    swaps = []

    while current != tl.level1:
        diff = current ^ tl.level1
        bit = (diff & -diff).bit_length()-1
        qubit = n-1-bit
        controls = [
            bool((current >> (n-1-q)) & 1) for q in range(n)
        ]
        controls[qubit] = False
        swaps.append(
            Swap(
                target=qubit,
                control_vals=controls
            )
        )
        current ^= (1<<bit)

    return swaps


def decompose_swap(swap: Swap) -> Circuit:
    n = len(swap.control_vals)
    X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
    return controlled_circuit(
        n=n, target=swap.target, control_vals=swap.control_vals, unitary=X
    )


def controlled_circuit(
    n: int, target: int, control_vals: list[bool], unitary: np.ndarray
) -> Circuit:
    circuit = []
    Xmatrix = np.array([[0, 1], [1, 0]], dtype=DTYPE)
    for q, val in enumerate(control_vals):
        if q == target:
            continue
        if not val:
            circuit.append(SingleQubitGate(n=n, qubit=q, unitary=Xmatrix))
    circuit.append(ControlledU(n=n, target=target, unitary=unitary))
    for q, val in enumerate(control_vals):
        if q == target:
            continue
        if not val:
            circuit.append(SingleQubitGate(n=n, qubit=q, unitary=Xmatrix))
    return circuit


# ---------------------------------------------------------------------------
# Stage 2-5: the decomposition pipeline
# ---------------------------------------------------------------------------


def decompose_twolevel(tl: TwoLevel) -> Circuit:
    """Decomposes a Two-Level unitary matrix into standard controlled configurations,

    ensuring structural control line bitmasks pass safely to the recursive unroller.
    """
    path = gray_code(tl)
    if len(path) == 0:
        return []
    circuit = []
    current = tl.level0
    n = num_qubits(tl.size)

    for sw in path[:-1]:
        circuit.extend(decompose_swap(sw))
        bit = n-1-sw.target
        current ^= (1<<bit)
    
    last = path[-1]
    controls = [
        bool((current>>(n-1-q)&1)) for q in range(n)
    ]
    controls[last.target] = False
    circuit.extend(
        controlled_circuit(
            n=n,
            target=last.target,
            control_vals=controls,
            unitary=tl.unitary
        )
    )
    for sw in reversed(path[:-1]):
        circuit.extend(decompose_swap(sw))
    return circuit


def decompose_controlled(
    n: int, controls: list[int], target: int, u: np.ndarray
) -> Circuit:
    if len(controls) == 0:
        return [SingleQubitGate(n=n, qubit=target, unitary=u)]
    if len(controls) == 1:
        if np.allclose(u, np.array([[0, 1], [1, 0]], dtype=DTYPE)):
            return [CNOT(n=n, control=controls[0], target=target)]
        return [CU(n=n, control=controls[0], target=target, unitary=u)]
        
    X = np.array([
        [0,1],[1,0]], dtype=DTYPE
    )
    V = rotation.unitary2_sqrt(u)
    pivot = controls[0]
    rest = controls[1:]

    # Left-multiplication engine execution order adjustment
    circuit = []
    circuit += decompose_controlled(n, rest, target, V)
    circuit += decompose_controlled(n, rest, pivot, X)
    circuit += decompose_controlled(n, controls, target, V.conj().T)
    circuit += decompose_controlled(n, rest, pivot, X)
    circuit += decompose_controlled(n, rest, target, V)

    return circuit


def decompose_controlledU(g: ControlledU) -> Circuit:
    controls = [i for i in range(g.n) if i != g.target]
    return decompose_controlled(n=g.n, controls=controls, target=g.target, u=g.unitary)


def decompose_cu(g: CU) -> Circuit:
    abc = abc_decompose(g.unitary)
    n = g.n
    c = g.control
    t = g.target
    X = CNOT(n=n, control=c, target=t) 
    return [
        SingleQubitGate(n, t, abc.C),
        X,
        SingleQubitGate(n, t, abc.B),
        X,
        SingleQubitGate(n, t, abc.A),
        SingleQubitGate(
            n=n,
            qubit=c,
            unitary=np.array([[1, 0], [0, np.exp(1j * abc.alpha)]], dtype=DTYPE),
        ),
    ]


def decompose_to_basis(u: np.ndarray) -> Circuit:
    N = u.shape[0]
    n = num_qubits(N)
    if n == 1:
        return [SingleQubitGate(n=1, qubit=0, unitary=u)]
    
    two_levels = twolevel_decomposition(u)
    circuit = to_circuit(two_levels)

    stage2_circuit = []
    for g in circuit:
        if isinstance(g, TwoLevel):
            stage2_circuit.extend(decompose_twolevel(g))
        else:
            stage2_circuit.append(g)
            
    stage3_circuit = []
    for g in stage2_circuit:
        if isinstance(g, ControlledU):
            stage3_circuit.extend(decompose_controlledU(g))
        else:
            stage3_circuit.append(g)
            
    final_circuit = []
    for g in stage3_circuit:
        if isinstance(g, CU):
            final_circuit.extend(decompose_cu(g))
        else:
            final_circuit.append(g)
    return final_circuit


def ht_gates(n: int, qubit: int, word: str) -> Circuit:
    circuit = []
    for gate in reversed(word):
        if gate == "H":
            circuit.append(SingleQubitGate(n, qubit, rotation.H))
        elif gate == "T":
            circuit.append(SingleQubitGate(n, qubit, rotation.T))
    return circuit


def decompose_to_ht(u: np.ndarray, error: float) -> Circuit:
    circuit = decompose_to_basis(u)
    final = []
    for g in circuit:
        if isinstance(g, SingleQubitGate):
            word = rotation.approximate_in_ht(g.unitary, error)
            final += ht_gates(g.n, g.qubit, word)
        else:
            final.append(g)
    return final

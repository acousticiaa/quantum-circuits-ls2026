import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=DTYPE)
# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    # raise NotImplementedError("to_bloch is not implemented yet")
    alpha = np.angle(np.linalg.det(g))/2
    u = g*np.exp(-1j*alpha)
    tr = np.real(np.trace(u))/2
    tr = np.clip(tr,-1.0,1.0)
    theta = 2*np.arccos(tr)
    if np.isclose(theta, 0):
        b = Bloch()
        b.alpha = alpha
        b.n = np.array([0.0, 0.0, 1.0])
        b.theta = 0.0
        return b
    s = np.sin(theta/2)
    nx = -np.imag(u[0, 1] + u[1, 0]) / (2 * s)
    ny = np.real(u[1, 0] - u[0, 1]) / (2 * s)
    nz = -np.imag(u[0, 0] - u[1, 1]) / (2 * s)
    n = np.array([nx, ny, nz], dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n/=norm
    b = Bloch()
    b.alpha = alpha
    b.n = n
    b.theta = theta
    return b


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
n1 = np.array([INV_SQRT2, 0.0, INV_SQRT2])
n2 = np.array([-INV_SQRT2, 0.0, INV_SQRT2])

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    x = np.dot(b.n, a1)
    y = np.dot(b.n, a2)
    z = np.dot(b.n, a3)

    theta = b.theta

    if np.isclose(theta, 0.0):
        return (0.0, 0.0, 0.0, b.alpha)

    phi = theta / 2.0
    s = np.sin(phi)
    c = np.cos(phi)

    beta = np.arctan2(
        np.sqrt((y * s) ** 2 + (z * s) ** 2),
        np.sqrt(c ** 2 + (x * s) ** 2),
    )

    sum_angle = np.arctan2(x * s, c)
    diff_angle = np.arctan2(z * s, y * s)

    alpha = (sum_angle - diff_angle) / 2.0
    gamma = (sum_angle + diff_angle) / 2.0

    alpha %= TWO_PI
    beta %= TWO_PI
    gamma %= TWO_PI

    return (alpha, beta, gamma, b.alpha)


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    target = angle % TWO_PI

    if min(target, TWO_PI - target) <= tolerance:
        return 0
    value = 1
    # Guard against infinite loops by capping search bounds
    for value in range(1, 100000):
        candidate = (value * LAMBDA_PI) % TWO_PI
        dist = abs(candidate - target)
        dist = min(dist, TWO_PI - dist)
        if dist <= tolerance:
            return value
    return 0


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """
    # # TODO(student): implement using the steps above.
    # raise NotImplementedError("decompose_2x2 is not implemented yet")
    newu = to_bloch(u)
    (alpha, beta, gamma, _globalphase) = n1n2n1_angles(newu)
    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)
    return (k,l,m)


# ---------------------------------------------------------------------------
# Single-qubit rotation helpers (see cpp/src/Unitary2_Bloch.h).
#
# These are the inverse/companion operations to to_bloch and are reused by the
# multi-qubit decomposition pipeline in decompose.py.
# ---------------------------------------------------------------------------


def from_axis_angle(b: Bloch) -> np.ndarray:
    nx, ny, nz = b.n
    sigma = np.array([[nz, nx - 1j * ny], [nx + 1j * ny, -nz]], dtype=DTYPE)
    I = np.eye(2, dtype=DTYPE)
    c = np.cos(b.theta / 2)
    s = np.sin(b.theta / 2)
    return np.exp(1j * b.alpha) * (c * I - 1j * s * sigma)


def Rz(theta: float) -> np.ndarray:
    """Rotation about the z axis (no global phase):

    Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}).
    """
    # TODO: implement (hint: from_axis_angle about axis [0, 0, 1]).
    # raise NotImplementedError("Rz is not implemented yet")
    b = Bloch()
    b.alpha = 0.0
    b.n = np.array([0.0, 0.0, 1.0])
    b.theta = theta
    return from_axis_angle(b)


def Ry(theta: float) -> np.ndarray:
    """Rotation about the y axis (no global phase):

    Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]].
    """
    # TODO: implement (hint: from_axis_angle about axis [0, 1, 0]).
    # raise NotImplementedError("Ry is not implemented yet")
    b = Bloch()
    b.alpha = 0.0
    b.n = np.array([0.0, 1.0, 0.0])
    b.theta = theta
    return from_axis_angle(b)


def euler_angles_zyz(u: np.ndarray) -> tuple[float, float, float, float]:
    """ZYZ Euler decomposition of a 2x2 unitary: angles (alpha, beta, gamma, delta)
    with

        u = e^{i alpha} Rz(beta) Ry(gamma) Rz(delta).

    alpha is the global phase (arg(det u)/2); the rest come from S = e^{-i alpha} u
    in SU(2), where s00 = cos(gamma/2) e^{-i(beta+delta)/2} and
    s10 = sin(gamma/2) e^{i(beta-delta)/2}. When gamma = 0 (s10 = 0), beta/delta are
    split arbitrarily (gimbal lock); the identity still holds.
    """
    # TODO: implement using the relations above.
    # raise NotImplementedError("euler_angles_zyz is not implemented yet")
    alpha = np.angle(np.linalg.det(u)) * 0.5
    S = np.exp(-1j * alpha) * u
    s00 = S[0,0]
    s10 = S[1,0]
    gamma = 2.0 * np.arctan2(abs(s10), abs(s00))
    if np.abs(np.sin(gamma/2.0)) < 1e-12:
        beta = -2 * np.angle(s00)
        delta = 0.0
    else:
        sum_angle = -2.0 * np.angle(s00)
        diff_angle = 2.0 * np.angle(s10)
        beta = (sum_angle + diff_angle) / 2.0
        delta = (sum_angle - diff_angle) / 2.0
    beta %= TWO_PI
    gamma %= TWO_PI
    delta %= TWO_PI
    return (alpha, beta, gamma, delta)


def unitary2_sqrt(u: np.ndarray) -> np.ndarray:
    """Principal square root: a 2x2 unitary V with V @ V == u, phase included.
    Take the Bloch form of u and halve both alpha and theta (same axis); squaring
    back doubles them, reproducing u exactly.
    """
    u2 = to_bloch(u)
    ans = Bloch()
    ans.alpha = u2.alpha / 2
    ans.n = u2.n.copy()
    ans.theta = u2.theta / 2
    return from_axis_angle(ans)
    # TODO: implement (hint: to_bloch, halve alpha and theta, from_axis_angle).
    # raise NotImplementedError("unitary2_sqrt is not implemented yet")

# ---------------------------------------------------------------------------
# H/T word machinery for approximating a 2x2 unitary in {H, T} (see cpp/src/HT.h).
#
# M1, M2 are short H/T words that realize rotations by THETA_M = 2*LAMBDA_PI about
# the axes a1, a2. A word is a flat string of 'H'/'T' characters, read left-to-right
# as a matrix product (leftmost char = leftmost/outermost factor).
# ---------------------------------------------------------------------------

# alternating (T-power, H-power, ...) exponents, starting with T
M1_WORD = [7, 1, 1, 1]
M2_WORD = [2, 1, 1, 1, 6, 1, 7, 1, 5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 7, 1, 6]


def expand_word(word: list[int]) -> str:
    """Flatten an alternating (T-power, H-power, ...) exponent list into a literal
    string of 'H'/'T' gates (left-to-right). Even indices are T, odd indices are H.
    """
    # TODO: implement.
    # raise NotImplementedError("expand_word is not implemented yet")
    out = []
    for i,pow in enumerate(word):
        gate = "T" if i%2 == 0 else "H"
        out.append(gate*pow)
    return "".join(out)

# flat H/T strings for the two building-block words (computed once expand_word works)
M1_STR = expand_word(M1_WORD)
M2_STR = expand_word(M2_WORD)


def gates_to_unitary(gates: str) -> np.ndarray:
    """The 2x2 unitary of a flat H/T gate string (left-to-right product)."""
    # TODO: implement (multiply H / T for each char, starting from I).
    # raise NotImplementedError(" gates_to_unitary is not implemented yet")
    U = np.eye(2, dtype=DTYPE)
    for g in gates:
        if g == "H":
            U = U@H
        else:
            U = U@T
    return U


def invert_gates(gates: str) -> str:
    """Inverse of a flat H/T word: reverse the gate order and invert each gate.
    H^-1 = H; the {H, T} basis has no T-dagger, so T^-1 must be spelled as T^7.
    """
    # TODO: implement.
    # raise NotImplementedError("invert_gates is not implemented yet")
    out = []
    for g in reversed(gates):
        if g=="H":
            out.append("H")
        else:
            out.append("T"*7)
    return "".join(out)



def power_gates(base: str, k: int) -> str:
    """The k-th power of a flat H/T word: base repeated k times. Negative k uses the
    inverse word (invert_gates).
    """
    # TODO: implement.
    # raise NotImplementedError("power_gates is not implemented yet")
    if k==0:
        return ""
    if k>0:
        return base*k
    else:
        return invert_gates(base)*(-k)


def approximate_in_ht(u: np.ndarray, error: float) -> str:
    """Approximate a 2x2 unitary `u` by a flat H/T word (up to global phase) to the
    angular tolerance `error` (smaller -> longer, more accurate).

    Use decompose_2x2 to get the powers (k, l, m) with u ~= M1^k M2^l M1^m, then
    assemble the word:

        power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m).
    """
    k,l,m = decompose_2x2(u,error)
    return (
        power_gates(M1_STR,k) + power_gates(M2_STR,l) + power_gates(M1_STR,m)
    )

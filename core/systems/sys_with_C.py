# core/system.py
import numpy as np
from numba import njit
from dataclasses import dataclass, fields, astuple

from core.physics import linear_law
from core.systems.sys_with_C import *

Y_ORDER = ("m01", "m12", "m13","p1")  # інтегровані (y)
AUX_ORDER = ("p0",)  # допоміжні (aux)

NY = len(Y_ORDER)
NAUX = len(AUX_ORDER)


@dataclass(frozen=True)
class Params:
    pf_tnk: float = 0.0
    po_tnk: float = 0.0
    rf_amp_2: float = 0.0

    rhoFu: float = 814.41
    C1: float = 1.27465e-9

    a01: float = 7.15292e8
    a12: float = 5.8531e12
    a13: float = 1.96184e12

    j01: float = 10115.2
    j12: float = 32228.9
    j13: float = 37852.4

    p2: float = 1e5
    p3: float = 1e5

    def as_tuple(self):
        # порядок = порядок полів dataclass
        return astuple(self)

# Автогенерація порядку і індексів
PARAM_ORDER = tuple(f.name for f in fields(Params))

def _declare_indices():
    for i, name in enumerate(Y_ORDER):
        globals()[f"I_{name}"] = i
    for i, name in enumerate(AUX_ORDER):
        globals()[f"A_{name}"] = i
    for _i, _name in enumerate(PARAM_ORDER):
        globals()[f"P_{_name}"] = _i

_declare_indices()
del _declare_indices


def initial_y():
    return np.array([0.0, 0.0, 0.0, 1e5], dtype=np.float64)


# @njit(cache=True)
@njit(cache=False)
def clamp_y_inplace(y):
    limit_m = 1e6
    limit_p = 300e5
    limit_pEnv = 1e5
    # обмеження Витрати
    if y[I_m01] > limit_m:  y[I_m01] = limit_m
    if y[I_m01] < -limit_m: y[I_m01] = -limit_m
    if y[I_m12] > limit_m:  y[I_m12] = limit_m
    if y[I_m12] < -limit_m: y[I_m12] = -limit_m
    if y[I_m13] > limit_m:  y[I_m13] = limit_m
    if y[I_m13] < -limit_m: y[I_m13] = -limit_m

    if y[I_p1] > limit_p: y[I_p1] = limit_p
    if y[I_p1] < limit_pEnv: y[I_p1] = limit_pEnv

# @njit(cache=True)
@njit(cache=False)
def rhs(t, y, p, dy, aux):
    clamp_y_inplace(y)

    rhoFu = p[P_rhoFu]
    C1 = p[P_C1]
    a01 = p[P_a01]
    a12 = p[P_a12]
    a13 = p[P_a13]
    j01 = p[P_j01]
    j12 = p[P_j12]
    j13 = p[P_j13]
    p2 = p[P_p2]
    p3 = p[P_p3]

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)
    aux[A_p0] = p0

    m01 = y[I_m01]
    m12 = y[I_m12]
    m13 = y[I_m13]
    p1 = y[I_p1]

    dy[I_m01] = (p0 - p1 - a01 / rhoFu * (abs(m01) * m01)) / j01
    dy[I_m12] = (p1 - p2 - a12 / rhoFu * (abs(m12) * m12)) / j12
    dy[I_m13] = (p1 - p3 - a13 / rhoFu * (abs(m13) * m13)) / j13
    dy[I_p1]  = (m01 - m12 - m13) / C1

__all__ = [
    "Y_ORDER", "AUX_ORDER", "NY", "NAUX",
    "PARAM_ORDER", "Params", "initial_y",
    "linear_law", "clamp_y_inplace", "rhs",
]

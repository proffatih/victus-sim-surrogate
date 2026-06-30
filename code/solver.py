"""
solver.py
=========
Parametric 2-D steady-state heat-conduction finite-element solver.

Governing equation (steady-state heat conduction with a volumetric source):

        -div( kappa(x; theta) grad u ) = f(x; theta)   in  Omega = (0,1)^2

with mixed boundary conditions:
        u = u_D                         on the left edge  (x = 0)   [Dirichlet]
        kappa du/dn = 0                 on top/bottom edges        [adiabatic / Neumann]
        kappa du/dn = -h (u - u_inf)    on the right edge (x = 1)  [Robin / convective]

The spatially-varying conductivity contains a Gaussian inclusion of higher (or
lower) conductivity centred at (cx, cy):

        kappa(x,y) = k0 * ( 1 + (C - 1) * exp( -((x-cx)^2 + (y-cy)^2) / (2 s^2) ) )

and a localised Gaussian heat source of magnitude Q centred at (0.5, 0.5):

        f(x,y) = Q * exp( -((x-0.5)^2 + (y-0.5)^2) / (2 sf^2) )

The design parameters theta = (C, cx, Q, u_D) are varied in the design study.
Fixed: k0 = 1.0, s = 0.15, sf = 0.10, cy = 0.5, h = 5.0, u_inf = 0.0.

Quantities of interest (QoI):
   * Tavg : area-averaged temperature over the whole domain
   * Tmax : maximum nodal temperature

Implemented with scikit-fem (P1 linear triangles).  No external mesh files.
Author: Fatih Gul (RTEU).  Reproducible: all randomness seeded by caller.
"""

import numpy as np
from skfem import (MeshTri, Basis, FacetBasis, ElementTriP1, asm,
                   BilinearForm, LinearForm, condense, solve)
from skfem.helpers import dot, grad

# -------------------------------------------------------------------------
# Fixed physical constants
# -------------------------------------------------------------------------
K0 = 1.0      # base conductivity
S_INC = 0.15  # inclusion width (std)
SF = 0.10     # source width (std)
CY = 0.5      # inclusion y-centre (fixed)
H_ROBIN = 5.0 # convective coefficient on right edge
U_INF = 0.0   # ambient temperature for Robin BC

# Parameter bounds: theta = (C, cx, Q, u_D)
PARAM_NAMES = ["C", "cx", "Q", "uD"]
PARAM_BOUNDS = np.array([
    [0.2, 5.0],   # C   : conductivity contrast (inclusion/base)
    [0.25, 0.75], # cx  : inclusion x-centre
    [0.0, 20.0],  # Q   : source magnitude
    [0.0, 2.0],   # uD  : Dirichlet temperature on left edge
])


def make_mesh(refine=5):
    """Uniformly refined triangulation of the unit square."""
    return MeshTri().refined(refine)


def _kappa_fn(C, cx):
    def kappa(x):
        xx, yy = x[0], x[1]
        g = np.exp(-((xx - cx) ** 2 + (yy - CY) ** 2) / (2 * S_INC ** 2))
        return K0 * (1.0 + (C - 1.0) * g)
    return kappa


def _source_fn(Q):
    def f(x):
        xx, yy = x[0], x[1]
        return Q * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2 * SF ** 2))
    return f


def solve_heat(theta, mesh=None, refine=5, return_field=False):
    """
    Solve the parametric heat problem for theta = (C, cx, Q, uD).

    Returns dict with QoI {'Tavg','Tmax', 'ndofs'} and optionally the field.
    """
    C, cx, Q, uD = [float(v) for v in theta]
    if mesh is None:
        mesh = make_mesh(refine)
    e = ElementTriP1()
    basis = Basis(mesh, e)

    kappa = _kappa_fn(C, cx)
    fsrc = _source_fn(Q)

    @BilinearForm
    def a_diff(u, v, w):
        return kappa(w.x) * dot(grad(u), grad(v))

    @LinearForm
    def l_src(v, w):
        return fsrc(w.x) * v

    A = asm(a_diff, basis)
    b = asm(l_src, basis)

    # Robin BC on right edge (x = 1):  kappa du/dn = -h (u - u_inf)
    # weak form adds  + h * int_{right} u v ds  to A  and  + h*u_inf * int v ds to b
    right = FacetBasis(mesh, e, facets=mesh.facets_satisfying(lambda x: np.abs(x[0] - 1.0) < 1e-9))

    @BilinearForm
    def robin_a(u, v, w):
        return H_ROBIN * u * v

    @LinearForm
    def robin_l(v, w):
        return H_ROBIN * U_INF * v

    A = A + asm(robin_a, right)
    b = b + asm(robin_l, right)

    # Dirichlet BC on left edge (x = 0): u = uD
    left_dofs = basis.get_dofs(lambda x: np.abs(x[0]) < 1e-9)
    x = basis.zeros()
    x[left_dofs] = uD
    sol = solve(*condense(A, b, x=x, D=left_dofs))

    # QoI
    @LinearForm
    def unit(v, w):
        return 1.0 * v
    ones = asm(unit, basis)
    area = ones.sum()              # = 1 for unit square
    Tavg = float(ones @ sol / area)
    Tmax = float(sol.max())

    out = {"Tavg": Tavg, "Tmax": Tmax, "ndofs": int(basis.N)}
    if return_field:
        out["mesh"] = mesh
        out["basis"] = basis
        out["u"] = sol
    return out


# -------------------------------------------------------------------------
# Verification utilities
# -------------------------------------------------------------------------
def manufactured_solution_test(refine_levels=(2, 3, 4, 5, 6)):
    """
    Method of Manufactured Solutions (MMS) for the CONSTANT-conductivity
    Poisson problem with pure Dirichlet BC, to verify spatial convergence.

        u_exact(x,y) = sin(pi x) sin(pi y)
        -Lap u = f  =>  f = 2 pi^2 sin(pi x) sin(pi y)
        u = 0 on all four edges.

    Returns (hs, l2errors, rate).
    """
    hs, errs = [], []
    for r in refine_levels:
        mesh = MeshTri().refined(r)
        e = ElementTriP1()
        basis = Basis(mesh, e)

        @BilinearForm
        def a(u, v, w):
            return dot(grad(u), grad(v))

        @LinearForm
        def l(v, w):
            return 2 * np.pi ** 2 * np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1]) * v

        A = asm(a, basis)
        b = asm(l, basis)
        D = basis.get_dofs()  # all boundary
        sol = solve(*condense(A, b, D=D))

        # L2 error via quadrature; interpolate uh at quadrature points
        uh_proj = basis.interpolate(sol)

        @LinearForm
        def err_form(v, w):
            ue = np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1])
            return (w["uh"] - ue) ** 2 * v
        e2 = asm(err_form, basis, uh=uh_proj)
        l2 = np.sqrt(e2.sum())
        h = 1.0 / (2 ** r)
        hs.append(h)
        errs.append(float(l2))
    hs = np.array(hs)
    errs = np.array(errs)
    # observed convergence rate (log-log slope)
    rate = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    return hs, errs, float(rate)


def mesh_convergence_qoi(theta, refine_levels=(2, 3, 4, 5, 6, 7)):
    """QoI convergence under uniform refinement for a fixed parameter set."""
    rows = []
    for r in refine_levels:
        res = solve_heat(theta, refine=r)
        rows.append((r, res["ndofs"], res["Tavg"], res["Tmax"]))
    return rows


if __name__ == "__main__":
    hs, errs, rate = manufactured_solution_test()
    print("MMS L2 convergence rate (P1, expected ~2):", round(rate, 3))
    for h, e in zip(hs, errs):
        print(f"  h={h:.4f}  L2err={e:.3e}")
    print()
    theta = [3.0, 0.5, 10.0, 1.0]
    print("QoI mesh convergence for theta=", theta)
    for r, nd, ta, tm in mesh_convergence_qoi(theta):
        print(f"  refine={r}  ndofs={nd:6d}  Tavg={ta:.6f}  Tmax={tm:.6f}")

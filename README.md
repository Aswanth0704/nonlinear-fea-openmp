# nonlinear-fea-openmp

[![CI](https://github.com/Aswanth0704/nonlinear-fea-openmp/actions/workflows/ci.yml/badge.svg)](https://github.com/Aswanth0704/nonlinear-fea-openmp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)]()

A 3D nonlinear finite element solver for **large-deformation, history-dependent solid
mechanics coupled to two reacting-diffusing scalar fields**, written from scratch in C++17
with OpenMP-parallel element assembly.

The physics application is soft-tissue remodeling after surgery, but the machinery is the
general one: finite-strain kinematics, a multiplicative split of the deformation gradient
into elastic and inelastic parts, per-integration-point evolution of internal state
variables, consistent linearization of that state into the global tangent, and a monolithic
Newton–Raphson solve of the coupled system with a Krylov linear solver at each iteration.

---

## What is actually in here

| Area | Implementation |
|---|---|
| **Geometric nonlinearity** | Total-Lagrangian finite strain. Right Cauchy–Green `C = Fᵀ F`, second Piola–Kirchhoff stress, full geometric + material tangent. No small-strain assumption anywhere. |
| **Material nonlinearity** | Multiplicative split `F = Fe · Fg`. `Fg = λa a₀⊗a₀ + λs s₀⊗s₀ + λn n₀⊗n₀` evolves on an orthonormal material triad — structurally the same object as a plastic/growth deformation. Anisotropic exponential fiber energy + neo-Hookean ground substance + volumetric penalty. |
| **History dependence** | Six internal variables per integration point (fiber volume fraction, triad orientation, dispersion, three inelastic stretches) integrated in time with a **local** solve nested inside each global Newton iteration. Explicit sub-stepped and fully implicit variants both implemented. |
| **Consistent tangent** | The local solve returns `∂Θ/∂C`, `∂Θ/∂ρ`, `∂Θ/∂c` — the linearization of the internal state with respect to the global unknowns — which is folded into the element tangent. This is what preserves Newton's convergence rate; a frozen-state tangent would degrade it to linear. |
| **Multiphysics coupling** | Monolithic 3-field system: displacement (3 dof/node) + cell density + chemical concentration. All nine tangent blocks `K_xx, K_xρ, K_xc, K_ρx, …` are assembled, not staggered or lagged. |
| **FEM discretization** | Isoparametric hex8 / hex20 / hex27 and tet4 / tet10, with matching Gauss and tetrahedral quadrature rules. Surface elements for Neumann and Robin boundary terms. |
| **Linear algebra** | Sparse triplet assembly into a row-major `Eigen::SparseMatrix`, solved with **BiCGSTAB** at every Newton step. Diagonal preconditioning by default; ILUT and direct (SparseLU / PardisoLU) paths are wired in for ill-conditioned cases. Optional Intel MKL backend for Eigen. |
| **Shared-memory parallelism** | `#pragma omp parallel for` over elements for residual/tangent evaluation, solution update, and post-processing. Element evaluation is fully reentrant — no shared mutable state — which the test suite verifies bit-for-bit across thread counts. |
| **Robustness** | Adaptive time-step reduction on Newton or linear-solver failure, with automatic state rollback and step retry. |

---

## Numerical design notes

**Why the local/global split.** The internal state at each integration point satisfies its
own ODE system in time. Solving it locally and returning its sensitivity to the global
unknowns keeps the global Newton iteration second-order convergent while keeping the global
unknown count at 5 dof/node instead of carrying ~6 more fields per integration point.

**Where the tangent comes from.** `evalWound` (`src/wound.cpp`) evaluates residual and tangent
together at every integration point. The stress pull-back through `Fg⁻¹` and the chain rule
through `∂Θ/∂C` are the two places where an inconsistency is easy to introduce and hard to
notice — the test suite targets both directly (see below).

**Known bottleneck.** Element evaluation parallelizes cleanly, but insertion into the global
triplet list is currently serialized behind `#pragma omp critical`. On high thread counts
this is the limiting factor, not the element work. Per-thread triplet buffers merged after
the parallel region is the obvious fix and is the first item on the roadmap.

---

## Testing

Correctness for a nonlinear solver is not "does it run" — it is whether the discrete
operators satisfy the identities they are supposed to. The suite (`tests/`, 400+ assertions)
checks properties that hold *exactly* in floating point:

**`test_kinematics`** — the constitutive core.
- The undeformed reference state carries **zero** stress: the neo-Hookean and volumetric
  contributions cancel identically, for any dispersion parameter and any fiber fraction.
- A **purely grown configuration** (`C = Fgᵀ Fg`, so `Fe = I`) also carries zero stress, for
  arbitrary growth stretches and arbitrary triad orientation. This is the invariant that the
  multiplicative split exists to guarantee, and it is the test that catches a wrong pull-back.
- **Objectivity**: `S(R C Rᵀ, R a₀, …) = R S(C, a₀, …) Rᵀ` for the passive, active, and
  volumetric parts independently.
- **Monotonicity** of the fiber-direction stress through a uniaxial stretch sweep, with the
  correct sign in tension and compression and an exact zero crossing at `λ = 1`.

**`test_element_functions`** — partition of unity and vanishing derivative sums for all five
element families, the nodal interpolation property, quadrature weights integrating unity over
each parent domain, Gauss exactness on `ξ²η²ζ²`, and Jacobian determinants plus element volume
recovery on cubes and under superposed rigid-body motion.

**`test_mesh_generator`** — connectivity validity, node extents, boundary-face tagging, and —
the one that matters — a **positive Jacobian determinant at every integration point of every
element**, with mesh-wide quadrature reproducing the analytic domain volume.

**`test_openmp_assembly`** — assembles a full mesh serially, then again at every available
thread count, and requires **bit-for-bit identical** residuals and tangent blocks. Element
evaluations are independent and reduce in a fixed order, so exact equality is the right
expectation; anything weaker would hide a genuine race.

The suite is verified non-vacuous by mutation: removing the `Fg⁻¹` pull-back from the
volumetric stress — a plausible, silent bug that leaves the reference state untouched —
turns 12 assertions red.

---

## Continuous integration

Every push and pull request runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **8-way build matrix** — {gcc, clang} × {Debug, Release} × {OpenMP on, off}, all running
  the full test suite. The OpenMP-off leg exists because the serial path is a real
  configuration, and it keeps the `#pragma omp` usage honest.
- **AddressSanitizer + UndefinedBehaviorSanitizer** — full suite under both, with
  `-fno-sanitize-recover=undefined` and leak detection on. Currently clean.
- **macOS / AppleClang** — cross-platform and cross-toolchain build with Homebrew `libomp`.

---

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Dependencies are Eigen 3.3+ (header-only) and Boost headers. CMake will use an installed
Eigen, fall back to a sibling-directory checkout, and finally fetch Eigen 3.4.0 if neither is
present.

```bash
# Ubuntu
sudo apt-get install libeigen3-dev libboost-dev libomp-dev
# macOS
brew install eigen boost libomp
```

### Options

| Option | Default | Effect |
|---|---|---|
| `WOUND_ENABLE_OPENMP` | `ON` | OpenMP-parallel assembly; degrades gracefully to serial if unavailable |
| `WOUND_BUILD_TESTS` | `ON` | Build and register the CTest suite |
| `WOUND_USE_MKL` | `OFF` | Use Intel MKL as the Eigen BLAS/LAPACK backend |

The solver driver is `surgerywoundcpp3D`. It was developed on the Purdue Negishi cluster
(Intel compilers 19.1.3, Boost 1.80, Eigen 3.4) and is submitted there with
`sbatch surgerywoundcpp3D.sub`.

---

## Layout

```
include/      public headers
src/
  solver.cpp            Newton–Raphson driver, sparse assembly, BiCGSTAB solve, time stepping
  wound.cpp             element residual and consistent tangent, constitutive response
  local_solver.cpp      integration-point internal-state solve (explicit and implicit)
  element_functions.cpp shape functions, quadrature rules, Jacobians
  file_io.cpp           Abaqus / COMSOL / ParaView mesh input, VTK output
  results_circle_wound.cpp  driver: problem setup, parameters, boundary conditions
meshing/      structured hex mesh generation
tests/        CTest suite
```

---

## Roadmap

Ordered by expected impact:

1. **Per-thread triplet buffers** to remove the `omp critical` serialization in assembly.
2. **GPU offload** of element evaluation (CUDA/HIP). The element loop is embarrassingly
   parallel with a fixed per-element working set, which is the favorable case; the local
   internal-state solve is the part that needs care because of its data-dependent iteration
   count. *Not implemented yet.*
3. **Preconditioning study** — the tangent becomes ill-conditioned as the inelastic stretches
   drift from unity; ILUT and block-diagonal field-split preconditioners are the candidates.
4. **Strong-scaling benchmark** in CI, so speedup numbers are measured rather than asserted.

---

## Citation

The underlying mechanobiological model builds on:

- A. Buganza Tepole et al., mechanobiological models of skin growth and remodeling.
- D. Sohutskay et al., computational models of wound healing and collagen remodeling.

## License

MIT — see [LICENSE](LICENSE).

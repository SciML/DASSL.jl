# API

The DASSL-owned API -- `dassl`, `dasslSolve`, `dasslSolve!`, `dasslIterator`,
`DASSLCache`, `alg_cache` -- is documented on the [Public API](index.md) page.

## Reexported SciML common interface

`using DASSL` also brings in the parts of the SciML common interface needed to build a
DAE problem, solve it, and inspect the result, so they do not have to be imported
separately. DASSL does not define these names -- they are owned and documented by
[SciMLBase](https://docs.sciml.ai/SciMLBase/stable/) and
[DiffEqBase](https://docs.sciml.ai/DiffEqDocs/stable/), and that is where their
documentation lives:

  - Problems: `DAEProblem`, `EnsembleProblem`
  - Functions: `DAEFunction`
  - Solutions: `DAESolution`, `EnsembleSolution`, `EnsembleSummary`, `DEStats`
  - Ensemble algorithms: `EnsembleSerial`, `EnsembleThreads`, `EnsembleDistributed`,
    `EnsembleSplitThreads`, and the `EnsembleAnalysis` module
  - Solving: `solve`, `remake`
  - Return status: `ReturnCode`, `successful_retcode`
  - Initialization algorithms, passed as the `initializealg` keyword to `solve`:
    `DefaultInit` (owned by DiffEqBase), `NoInit`, `CheckInit`, `OverrideInit`
  - `NullParameters`

Anything else from SciMLBase or DiffEqBase must be imported from SciMLBase or DiffEqBase
directly. Three groups are deliberately absent:

  - **Callbacks.** DASSL errors on a `callback` keyword ("DASSL is not compatible with
    callbacks"), so `ContinuousCallback` and friends are not part of its surface.
  - **The integrator interface** (`init`, `step!`, `solve!`, `reinit!`, ...). DASSL has
    no SciML integrator; its own iterator is [`dasslIterator`](index.md).
  - **`BrownFullBasicInit` and `ShampineCollocationInit`.** Unlike DASKR, `initialize_dae!`
    has no method for either, so passing one is an error.

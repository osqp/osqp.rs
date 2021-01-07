Version 0.6.2 (January 7, 2021)
-----------
- Update to OSQP 0.6.2 (see [OSQP changelog][osqp-clog] for details).

Version 0.6.1 (September 28, 2020)
-----------
- Fix `CscMatrix::from_row_iter_dense` incorrectly initialising non-square matrices.

Version 0.6.0 (September 5, 2019)
-----------
- Update to OSQP 0.6.0 (see [OSQP changelog][osqp-clog] for details).
- Add `CscMatrix::from_column_iter_dense` and `CscMatrix::from_row_iter_dense` to construct a dense CscMatrix with elements filled from column-major and row-major iterators.
- `Problem` now requires the `P` matrix to be structurally upper triangular. Two methods on `CscMatrix`, `is_structurally_upper_tri` and `into_upper_tri`, are added to assist with this requirement.
- Add variants to the `SetupError` enum returned by `Problem::new` explaining the cause of the error.

Version 0.5.0 (December 10, 2018)
-----------
- Update to OSQP 0.5.0 (see [OSQP changelog][osqp-clog] for details).
- `Problem::new` now returns a `Result` to indicate problem setup failure.

Version 0.4.1 (October 16, 2018)
-----------
- Update to OSQP 0.4.1 (see [OSQP changelog][osqp-clog] for details).

Version 0.4.0 (July 26, 2018)
-----------
- Update to OSQP 0.4.0 (see [OSQP changelog][osqp-clog] for details).
- Changes for QDLDL compatibility.  Removed SuiteSparse dependencies.
- Supports reporting of non-convex problems by OSQP.

Version <= 0.3.0
----------------
- See [OSQP changelog][osqp-clog] for details.

[osqp-clog]: https://github.com/oxfordcontrol/osqp/blob/master/CHANGELOG.md "OSQP changelog"

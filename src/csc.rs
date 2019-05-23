use osqp_sys as ffi;
use std::borrow::Cow;
use std::iter;
use std::slice;

use float;

macro_rules! check {
    ($check:expr) => {
        if !{ $check } {
            return false;
        }
    };
}

/// A matrix in Compressed Sparse Column format.
#[derive(Clone, Debug, PartialEq)]
pub struct CscMatrix<'a> {
    /// The number of rows in the matrix.
    pub nrows: usize,
    /// The number of columns in the matrix.
    pub ncols: usize,
    /// The CSC column pointer array.
    ///
    /// It contains the offsets into the index and data arrays of the entries in each column.
    pub indptr: Cow<'a, [usize]>,
    /// The CSC index array.
    ///
    /// It contains the row index of each non-zero entry.
    pub indices: Cow<'a, [usize]>,
    /// The CSC data array.
    ///
    /// It contains the values of each non-zero entry.
    pub data: Cow<'a, [float]>,
}

impl<'a> CscMatrix<'a> {
    /// Creates a dense CSC matrix with its elements filled with the components provided by an
    /// iterator in column-major order.
    ///
    /// Panics if `iter` contains fewer than `nrows * ncols` elements.
    pub fn from_column_iter_dense<I: IntoIterator<Item = float>>(
        nrows: usize,
        ncols: usize,
        iter: I,
    ) -> CscMatrix<'static> {
        CscMatrix::from_iter_dense_inner(nrows, ncols, |size| {
            let mut data = Vec::with_capacity(size);
            data.extend(iter.into_iter().take(size));
            assert_eq!(size, data.len());
            data
        })
    }

    /// Creates a dense CSC matrix with its elements filled with the components provided by an
    /// iterator in row-major order.
    ///
    /// The order of elements in the slice must follow the usual mathematic writing, i.e.,
    /// row-by-row.
    ///
    /// Panics if `iter` contains fewer than `nrows * ncols` elements.
    pub fn from_row_iter_dense<I: IntoIterator<Item = float>>(
        nrows: usize,
        ncols: usize,
        iter: I,
    ) -> CscMatrix<'static> {
        CscMatrix::from_iter_dense_inner(nrows, ncols, |size| {
            let mut iter = iter.into_iter();
            let mut data = vec![0.0; size];
            for r in 0..nrows {
                for c in 0..ncols {
                    data[c * ncols + r] = iter.next().expect("not enough elements in iterator");
                }
            }
            data
        })
    }

    fn from_iter_dense_inner<F: FnOnce(usize) -> Vec<float>>(
        nrows: usize,
        ncols: usize,
        f: F,
    ) -> CscMatrix<'static> {
        let size = nrows
            .checked_mul(ncols)
            .expect("overflow calculating matrix size");

        let data = f(size);

        CscMatrix {
            nrows,
            ncols,
            indptr: Cow::Owned((0..ncols + 1).map(|i| i * nrows).collect()),
            indices: Cow::Owned(iter::repeat(0..nrows).take(ncols).flat_map(|i| i).collect()),
            data: Cow::Owned(data),
        }
    }

    /// Returns `true` if the matrix is structurally upper triangular.
    ///
    /// A matrix is structurally upper triangular if, for each column, all elements below the
    /// diagonal, i.e. with their row number greater than their column number, are empty.
    ///
    /// Note that an element with an explicit value of zero is not empty. To be empty an element
    /// must not be present in the sparse encoding of the matrix.
    pub fn is_structurally_upper_tri(&self) -> bool {
        for col in 0..self.indptr.len().saturating_sub(1) {
            let col_data_start_idx = self.indptr[col];
            let col_data_end_idx = self.indptr[col + 1];

            for &row in &self.indices[col_data_start_idx..col_data_end_idx] {
                if row > col {
                    return false;
                }
            }
        }

        true
    }

    /// Extracts the upper triangular elements of the matrix.
    ///
    /// This operation performs no allocations if the matrix is already structurally upper
    /// triangular or if it contains only owned data.
    ///
    /// The returned matrix is guaranteed to be structurally upper triangular.
    pub fn into_upper_tri(self) -> CscMatrix<'a> {
        if self.is_structurally_upper_tri() {
            return self;
        }

        let mut indptr = self.indptr.into_owned();
        let mut indices = self.indices.into_owned();
        let mut data = self.data.into_owned();

        let mut next_data_idx = 0;

        for col in 0..indptr.len().saturating_sub(1) {
            let col_start_idx = indptr[col];
            let next_col_start_idx = indptr[col + 1];

            indptr[col] = next_data_idx;

            for data_idx in col_start_idx..next_col_start_idx {
                let row = indices[data_idx];

                if row <= col {
                    data[next_data_idx] = data[data_idx];
                    indices[next_data_idx] = row;
                    next_data_idx += 1;
                }
            }
        }

        if let Some(data_len) = indptr.last_mut() {
            *data_len = next_data_idx
        }
        indices.truncate(next_data_idx);
        data.truncate(next_data_idx);

        CscMatrix {
            indptr: Cow::Owned(indptr),
            indices: Cow::Owned(indices),
            data: Cow::Owned(data),
            ..self
        }
    }

    pub(crate) unsafe fn to_ffi(&self) -> ffi::csc {
        // Casting is safe as at this point no indices exceed isize::MAX and osqp_int is a signed
        // integer of the same size as usize/isize.
        ffi::csc {
            nzmax: self.data.len() as ffi::osqp_int,
            m: self.nrows as ffi::osqp_int,
            n: self.ncols as ffi::osqp_int,
            p: self.indptr.as_ptr() as *mut usize as *mut ffi::osqp_int,
            i: self.indices.as_ptr() as *mut usize as *mut ffi::osqp_int,
            x: self.data.as_ptr() as *mut float,
            nz: -1,
        }
    }

    pub(crate) unsafe fn from_ffi<'b>(csc: *const ffi::csc) -> CscMatrix<'b> {
        let nrows = (*csc).m as usize;
        let ncols = (*csc).n as usize;
        let nnz = (*csc).nzmax as usize;

        CscMatrix {
            nrows,
            ncols,
            indptr: Cow::Borrowed(slice::from_raw_parts((*csc).p as *const usize, ncols + 1)),
            indices: Cow::Borrowed(slice::from_raw_parts((*csc).i as *const usize, nnz)),
            data: Cow::Borrowed(slice::from_raw_parts((*csc).x as *const float, nnz)),
        }
    }

    pub(crate) fn assert_same_sparsity_structure(&self, other: &CscMatrix) {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);
        assert_eq!(&*self.indptr, &*other.indptr);
        assert_eq!(&*self.indices, &*other.indices);
        assert_eq!(self.data.len(), other.data.len());
    }

    pub(crate) fn is_valid(&self) -> bool {
        let max_idx = isize::max_value() as usize;
        check!(self.nrows <= max_idx);
        check!(self.ncols <= max_idx);
        check!(self.indptr.len() <= max_idx);
        check!(self.indices.len() <= max_idx);
        check!(self.data.len() <= max_idx);

        // Check row pointers
        check!(self.indptr.len() == self.ncols + 1);
        check!(self.indptr[self.ncols] == self.data.len());
        let mut prev_row_idx = 0;
        for &row_idx in self.indptr.iter() {
            // Row pointers must be monotonically nondecreasing
            check!(row_idx >= prev_row_idx);
            prev_row_idx = row_idx;
        }

        // Check index values
        check!(self.data.len() == self.indices.len());
        check!(self.indices.iter().all(|r| *r < self.nrows));
        for i in 0..self.ncols {
            let row_indices = &self.indices[self.indptr[i] as usize..self.indptr[i + 1] as usize];
            let mut row_indices = row_indices.iter();
            if let Some(&first_row) = row_indices.next() {
                let mut prev_row = first_row;
                for &row in row_indices {
                    // Row indices within each column must be monotonically increasing
                    check!(row > prev_row);
                    prev_row = row;
                }
                check!(prev_row < self.nrows);
            }
        }

        true
    }
}

// Any &CscMatrix can be converted into a CscMatrix without allocation due to the use of Cow.
impl<'a, 'b: 'a> From<&'a CscMatrix<'b>> for CscMatrix<'a> {
    fn from(mat: &'a CscMatrix<'b>) -> CscMatrix<'a> {
        CscMatrix {
            nrows: mat.nrows,
            ncols: mat.ncols,
            indptr: (*mat.indptr).into(),
            indices: (*mat.indices).into(),
            data: (*mat.data).into(),
        }
    }
}

// Enable creating a csc matrix from a slice of arrays for testing and small problems.
//
// let A: CscMatrix = (&[[1.0, 2.0],
//                       [3.0, 0.0],
//                       [0.0, 4.0]]).into;
//
impl<'a, I: 'a, J: 'a> From<I> for CscMatrix<'static>
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = &'a float>,
{
    fn from(rows: I) -> CscMatrix<'static> {
        let rows: Vec<Vec<float>> = rows
            .into_iter()
            .map(|r| r.into_iter().map(|&v| v).collect())
            .collect();

        let nrows = rows.len();
        let ncols = rows.iter().map(|r| r.len()).next().unwrap_or(0);
        assert!(rows.iter().all(|r| r.len() == ncols));
        let nnz = rows.iter().flat_map(|r| r).filter(|&&v| v != 0.0).count();

        let mut indptr = Vec::with_capacity(ncols + 1);
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);

        indptr.push(0);
        for c in 0..ncols {
            for r in 0..nrows {
                let value = rows[r][c];
                if value != 0.0 {
                    indices.push(r);
                    data.push(value);
                }
            }
            indptr.push(data.len());
        }

        CscMatrix {
            nrows,
            ncols,
            indptr: indptr.into(),
            indices: indices.into(),
            data: data.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    #[test]
    fn csc_from_array() {
        let mat = &[[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]];
        let csc: CscMatrix = mat.into();

        assert_eq!(3, csc.nrows);
        assert_eq!(2, csc.ncols);
        assert_eq!(&[0, 2, 4], &*csc.indptr);
        assert_eq!(&[0, 1, 0, 2], &*csc.indices);
        assert_eq!(&[1.0, 3.0, 2.0, 4.0], &*csc.data);
    }

    #[test]
    fn csc_from_ref() {
        let mat = &[[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]];
        let csc: CscMatrix = mat.into();
        let csc_ref: CscMatrix = (&csc).into();

        // csc_ref must be created without allocation
        if let Cow::Owned(_) = csc_ref.indptr {
            panic!();
        }
        if let Cow::Owned(_) = csc_ref.indices {
            panic!();
        }
        if let Cow::Owned(_) = csc_ref.data {
            panic!();
        }

        assert_eq!(csc.nrows, csc_ref.nrows);
        assert_eq!(csc.ncols, csc_ref.ncols);
        assert_eq!(csc.indptr, csc_ref.indptr);
        assert_eq!(csc.indices, csc_ref.indices);
        assert_eq!(csc.data, csc_ref.data);
    }

    #[test]
    fn csc_from_iter_dense() {
        let mat1 = CscMatrix::from_column_iter_dense(
            3,
            3,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                .iter()
                .cloned(),
        );
        let mat2 = CscMatrix::from_row_iter_dense(
            3,
            3,
            [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
                .iter()
                .cloned(),
        );
        let mat3: CscMatrix = (&[[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]).into();

        assert_eq!(mat1, mat3);
        assert_eq!(mat2, mat3);
    }

    #[test]
    fn same_sparsity_structure_ok() {
        let mat1: CscMatrix = (&[[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.0, 0.0]]).into();
        let mat2: CscMatrix = (&[[7.0, 8.0, 0.0], [9.0, 0.0, 0.0], [0.0, 10.0, 0.0]]).into();
        mat1.assert_same_sparsity_structure(&mat2);
    }

    #[test]
    #[should_panic]
    fn different_sparsity_structure_panics() {
        let mat1: CscMatrix = (&[[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.0, 6.0]]).into();
        let mat2: CscMatrix = (&[[7.0, 8.0, 0.0], [9.0, 0.0, 0.0], [0.0, 10.0, 0.0]]).into();
        mat1.assert_same_sparsity_structure(&mat2);
    }

    #[test]
    fn is_structurally_upper_tri() {
        let structurally_upper_tri: CscMatrix =
            (&[[1.0, 0.0, 5.0], [0.0, 3.0, 4.0], [0.0, 0.0, 2.0]]).into();
        let numerically_upper_tri: CscMatrix = CscMatrix::from_row_iter_dense(
            3,
            3,
            [1.0, 0.0, 5.0, 0.0, 3.0, 4.0, 0.0, 0.0, 2.0]
                .iter()
                .cloned(),
        );
        let not_upper_tri: CscMatrix =
            (&[[7.0, 2.0, 0.0], [9.0, 0.0, 5.0], [7.0, 10.0, 0.0]]).into();
        assert!(structurally_upper_tri.is_structurally_upper_tri());
        assert!(!numerically_upper_tri.is_structurally_upper_tri());
        assert!(!not_upper_tri.is_structurally_upper_tri());
    }

    #[test]
    fn into_upper_tri() {
        let mat: CscMatrix = (&[[1.0, 0.0, 5.0], [7.0, 3.0, 4.0], [6.0, 0.0, 2.0]]).into();
        let mat_upper_tri: CscMatrix =
            (&[[1.0, 0.0, 5.0], [0.0, 3.0, 4.0], [0.0, 0.0, 2.0]]).into();
        assert_eq!(mat.into_upper_tri(), mat_upper_tri);
    }
}

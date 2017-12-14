use osqp_sys as ffi;
use std::borrow::Cow;
use std::iter::once;
use std::slice;

use float;

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
    pub(crate) unsafe fn to_ffi(&self) -> ffi::csc {
        self.assert_valid();

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

    pub(crate) unsafe fn from_ffi(csc: *const ffi::csc) -> CscMatrix<'static> {
        let nrows = (*csc).m as usize;
        let ncols = (*csc).n as usize;
        let nnz = (*csc).nzmax as usize;

        CscMatrix {
            nrows,
            ncols,
            indptr: slice::from_raw_parts((*csc).p as *const usize, ncols + 1).into(),
            indices: slice::from_raw_parts((*csc).i as *const usize, nnz).into(),
            data: slice::from_raw_parts((*csc).x as *const float, nnz).into(),
        }
    }

    pub(crate) fn assert_same_sparsity_structure(&self, other: &CscMatrix) {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);
        assert_eq!(&*self.indptr, &*other.indptr);
        assert_eq!(&*self.indices, &*other.indices);
        assert_eq!(self.data.len(), other.data.len());
    }

    /// Assert `other` has the same upper triangle sparsity structure as `self`
    pub(crate) fn assert_same_upper_tri_sparsity_structure(&self, other: &CscMatrix) {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);
        assert_eq!(self.indptr.len(), other.indptr.len());
        assert_eq!(other.indices.len(), other.data.len());

        let mut col_start_idx = 0;
        let mut other_col_start_idx = 0;
        for (col_num, (&col_end_idx, &other_col_end_idx)) in self.indptr
            .iter()
            .zip(other.indptr.iter())
            .skip(1)
            .enumerate()
        {
            assert!(
                self.indices[col_start_idx..col_end_idx]
                    .iter()
                    .chain(once(&(self.nrows + 1)))
                    .zip(
                        other.indices[other_col_start_idx..other_col_end_idx]
                            .iter()
                            .chain(once(&(self.nrows + 1)))
                    )
                    .take_while(|&(&row_num, &other_row_num)| {
                        row_num <= col_num || other_row_num <= col_num
                    })
                    .all(|(&row_num, &other_row_num)| row_num == other_row_num)
            );
            col_start_idx = col_end_idx;
            other_col_start_idx = other_col_end_idx;
        }
    }

    pub(crate) fn assert_valid(&self) {
        use std::isize::MAX;
        let max_idx = MAX as usize;
        assert!(self.nrows <= max_idx);
        assert!(self.ncols <= max_idx);
        assert!(self.indptr.len() <= max_idx);
        assert!(self.indices.len() <= max_idx);
        assert!(self.data.len() <= max_idx);

        // Check row pointers
        assert_eq!(self.indptr[self.ncols], self.data.len());
        assert_eq!(self.indptr.len(), self.ncols + 1);
        self.indptr.iter().fold(0, |acc, i| {
            assert!(
                *i >= acc,
                "csc row pointers must be monotonically nondecreasing"
            );
            *i
        });

        // Check index values
        assert_eq!(
            self.data.len(),
            self.indices.len(),
            "csc row indices must be the same length as data"
        );
        assert!(self.indices.iter().all(|r| *r < self.nrows));
        for i in 0..self.ncols {
            let row_indices = &self.indices[self.indptr[i] as usize..self.indptr[i + 1] as usize];
            let first_index = *row_indices.get(0).unwrap_or(&0);
            row_indices.iter().skip(1).fold(first_index, |acc, i| {
                assert!(*i > acc, "csc row indices must be monotonically increasing");
                *i
            });
        }
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
        let rows: Vec<Vec<float>> = rows.into_iter()
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
    fn same_upper_tri_sparsity_structure_ok() {
        let mat1: CscMatrix = (&[[1.0, 2.0, 0.0], [3.0, 0.0, 5.0], [0.0, 6.0, 0.0]]).into();
        let mat2: CscMatrix = (&[[7.0, 2.0, 0.0], [9.0, 0.0, 5.0], [7.0, 10.0, 0.0]]).into();
        mat1.assert_same_upper_tri_sparsity_structure(&mat2);
    }

    #[test]
    #[should_panic]
    fn different_upper_tri_sparsity_structure_panics() {
        let mat1: CscMatrix = (&[[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.0, 6.0]]).into();
        let mat2: CscMatrix = (&[[7.0, 8.0, 0.0], [9.0, 0.0, 0.0], [0.0, 10.0, 0.0]]).into();
        mat1.assert_same_upper_tri_sparsity_structure(&mat2);
    }
}

use std::borrow::Cow;

use {float, osqp_sys as ffi};

pub struct CscMatrix<'a> {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Cow<'a, [usize]>,
    pub indices: Cow<'a, [usize]>,
    pub data: Cow<'a, [float]>,
}

impl<'a> CscMatrix<'a> {
    pub(crate) unsafe fn to_ffi(&self) -> ffi::csc {
        self.assert_valid();

        // Casting is safe as at this point no indices exceed isize::MAX and osqp_int is a signed integer
        // of the same size as usize/isize
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

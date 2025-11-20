# Preamble

I am learning C! This is one of my first libraries to contribute to Machine Learning using low-level programming languages. I accept any kind of suggestions. Be kind!

# MAT

Single-header C matrix library for ML/AI work.

## Usage
Very simple example
```
#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
    Mat m = mat_alloc(3, 3);
    mat_ones(m);
    mat_fill_diag(m, 5.0);
    
    Mat result = mat_alloc(3, 3);
    mat_dot(result, m, m);
    MAT_PRINT(result);
    
    mat_release(&m);
    mat_release(&result);
    return 0;
}Compile: `gcc -o program mat.c -lm`
```

## Core Functions

**Memory**: `mat_alloc()`, `mat_retain()`, `mat_release()`

**Init**: `mat_fill()`, `mat_rand()`, `mat_zeros()`, `mat_ones()`

**Operations**: `mat_dot()`, `mat_sum()`, `mat_nsum()`, `mat_prod_const()`, `mat_div_const()`, `mat_exp()`, `mat_sig()`

**Manipulation**: `mat_copy()`, `mat_trans()`, `mat_row()`, `mat_split_rows()`, `mat_split_cols()`

**Stats**: `mat_row_sum()`, `mat_col_sum()`, `mat_row_mean()`, `mat_col_mean()`, `mat_SS()`

**Access**: `MAT_AT(m, i, j)` for element at row i, column j

## Notes

- Uses reference counting—call `mat_release()` when done
- Most operations are in-place
- All values are `float`
- Override defaults with `MAT_MALLOC` and `MAT_ASSERT` before including

## License

Public domain.

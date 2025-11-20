#include <time.h>

#define MAT_IMPLEMENTATION
#include "mat.h"
    
#define ROWS 3
#define COLS 3

int main () {
    Mat m = mat_alloc(ROWS, COLS);
    MAT_PRINT(m);
    mat_ones(m);
    MAT_PRINT(m);
    mat_fill_diag(m, 5.0);
    MAT_PRINT(m);
    Mat m_dot = mat_alloc(ROWS, COLS);
    mat_dot(m_dot, m,m);
    MAT_PRINT(m_dot);
    Mat SS = mat_SS_const(m, ROWS);
    MAT_PRINT(SS);
    Mat m_trans = mat_trans(m);
    MAT_PRINT(m_trans);
    Mat m_row = mat_row_mean(m);
    MAT_PRINT(m_row);
    Mat m_col = mat_col_mean(m);
    MAT_PRINT(m_col);
    Mat *out_row = mat_split_rows(m, 0.7);
    MAT_PRINT(out_row[0]);
    MAT_PRINT(out_row[1]);
    Mat *out_col = mat_split_cols(m, 0.7);
    MAT_PRINT(out_col[0]);
    MAT_PRINT(out_col[1]);
    // This will also free
    Mat b = mat_alloc(ROWS, COLS);
    mat_zeros(b);
    Mat m_app_r = mat_append_rows(m, b);
    MAT_PRINT(m_app_r);
    Mat m_app_c = mat_append_cols(m, b);
    MAT_PRINT(m_app_c);

    mat_release(&m);
    mat_release(&m_app_r);

    return 0;
}
#include <time.h>

#define MAT_IMPLEMENTATION
#include "mat.h"
    
#define ROWS 3
#define COLS 3

int main () {
    Mat *m = mat_alloc(ROWS, COLS);
    MAT_PRINT(m);
    mat_ones(m);
    MAT_PRINT(m);
    mat_fill_diag(m, 5.0);
    MAT_PRINT(m);
    Mat *m_dot = mat_alloc(ROWS, COLS);
    mat_dot(m_dot, m, m);
    MAT_PRINT(m_dot);
    mat_release(m_dot);
    Mat *SS = mat_SS_const(m, ROWS);
    MAT_PRINT(SS);
    mat_release(SS);
    Mat *m_trans = mat_trans(m);
    MAT_PRINT(m_trans);
    mat_release(m_trans);
    Mat *m_row = mat_row_mean(m);
    MAT_PRINT(m_row);
    mat_release(m_row);
    Mat *m_col = mat_col_mean(m);
    MAT_PRINT(m_col);
    mat_release(m_col);
    Mat **out_row = mat_split_rows(m, 0.7);
    MAT_PRINT(out_row[0]);
    MAT_PRINT(out_row[1]);
    mat_release(out_row[0]);
    mat_release(out_row[1]);
    free(out_row);
    Mat **out_col = mat_split_cols(m, 0.7);
    MAT_PRINT(out_col[0]);
    MAT_PRINT(out_col[1]);
    mat_release(out_col[0]);
    mat_release(out_col[1]);
    free(out_col);
    Mat *b = mat_alloc(ROWS, COLS);
    mat_zeros(b);
    Mat *m_app_r = mat_append_rows(m, b);
    MAT_PRINT(m_app_r);
    mat_release(m_app_r);
    Mat *m_app_c = mat_append_cols(m, b);
    MAT_PRINT(m_app_c);
    mat_release(m_app_c);
    mat_release(b);
    Mat *try = mat_inverse_GJ(m);
    MAT_PRINT(try);
    mat_release(try);
    mat_swap_cols(m, 1, 2);
    MAT_PRINT(m);

    mat_release(m);

    return 0;
}
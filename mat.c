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

    // This will also free
    mat_release(&m);

    return 0;
}
#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef MAT_MALLOC
#include <stdlib.h>
#define MAT_MALLOC malloc
#endif 

#ifndef MAT_ASSERT
#include <assert.h>
#define MAT_ASSERT assert
#endif 
/*
    Structure for matrix allocation    
*/
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    size_t ref_count;
    float *es; // pointer to a continuous numbers of floats allocated with mat_alloc
} Mat;

// Access element at row i, column j. Requires Mat* pointer.
#define MAT_AT(m, i, j) ((m)->es[(i) * (m)->stride + (j)]) // Wrap every parameter in parentheses in case there is some complex expression like 'i+1'
                                                            // using stride to select only a given number of columns
#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])


float float_rand(void);

Mat *mat_alloc(size_t rows, size_t cols);
void mat_retain(Mat *m);
void mat_release(Mat *m);
void mat_free(Mat *m);
void mat_fill(Mat *m, float n);
void mat_fill_diag(Mat *m, float n);
void mat_rand(Mat *m, float low, float high);
Mat mat_row(const Mat *m, size_t row);
void mat_row_sum_const(Mat *m, size_t row, float val);
void mat_col_sum_const(Mat *m, size_t col, float val);
void mat_row_div_const(Mat *m, size_t row, float val);
void mat_col_div_const(Mat *m, size_t col, float val);
void mat_row_prod_const(Mat *m, size_t row, float val);
void mat_col_prod_const(Mat *m, size_t col, float val);
void mat_row_sum(Mat *a, Mat *b, size_t row_1, size_t row_2);
void mat_col_sum(Mat *a, Mat *b, size_t col_1, size_t col_2);
void mat_row_sub(Mat *a, Mat *b, size_t row_1, size_t row_2);
void mat_col_sub(Mat *a, Mat *b, size_t col_1, size_t col_2);
void mat_row_div(Mat *a, Mat *b, size_t row_1, size_t row_2);
void mat_col_div(Mat *a, Mat *b, size_t col_1, size_t col_2);
void mat_row_prod(Mat *a, Mat *b, size_t row_1, size_t row_2);
void mat_col_prod(Mat *a, Mat *b, size_t col_1, size_t col_2);
Mat **mat_split_rows(const Mat *m, float x);
Mat **mat_split_cols(const Mat *m, float x);
void mat_copy(Mat *dst, const Mat *src);
void mat_dot(Mat *dst, const Mat *a, const Mat *b);
void mat_sum(Mat *dst, const Mat *a);
void mat_nsum(Mat *dst, const Mat *a);
void mat_sig(Mat *m);
void mat_print(const Mat *m, const char* name, size_t padding);
void mat_prod_const(Mat *dst, float mult);
void mat_div_const(Mat *dst, float div);
void mat_sum_const(Mat *dst, float add);
void mat_exp(Mat *dst, float exp);
void mat_zeros(Mat *dst);
void mat_ones(Mat *dst);
void mat_swap_rows(Mat *m, size_t row_1, size_t row_2);
void mat_swap_cols(Mat *m, size_t col_1, size_t col_2);

Mat *mat_append_rows(const Mat *a, const Mat *b);
Mat *mat_append_cols(const Mat *a, const Mat *b);
Mat *mat_col_collapse_sum(const Mat *m);
Mat *mat_row_collapse_sum(const Mat *m);
Mat *mat_SS(const Mat *a, const Mat *b);
Mat *mat_SS_const(const Mat *a, float x);
Mat *mat_trans(const Mat *m);
Mat *mat_row_mean(const Mat *m);
Mat *mat_col_mean(const Mat *m);
Mat *mat_inverse_GJ(const Mat *m);


#define MAT_PRINT(m) mat_print(m, #m, 0) // this basically defines an alias to which the first argument is the one given
                                          // while the second is the one given stringified
#endif


#ifdef MAT_IMPLEMENTATION
/* ============= UTILS ============= */
float rand_float(void) {
    return(float) rand() / RAND_MAX;
}

float sigmoidf(float n) {
    return 1.f/(1.f+expf(-n));
}

/* ============= MAIN ============= */

/*
    Define a "malloc" cousin for memory allocation
*/
Mat *mat_alloc(size_t rows, size_t cols){
    Mat *m = MAT_MALLOC(sizeof(Mat) + sizeof(float) * rows * cols);
    MAT_ASSERT(m != NULL);
    m->rows = rows;
    m->cols = cols;
    m->stride = cols;
    m->ref_count = 1;
    // Points to memory right after the struct Mat
    m->es = (float *)(m + 1);
    return m;
} 

/*
    Define a "retain" cousin for increasing 'ref_count'.
*/
void mat_retain(Mat *m) {
    m -> ref_count ++;
}

/*
    Define a "release" cousin for releasing the  'ref_count' and freeing it from memory if it hits 0
*/
void mat_release(Mat *m){
    MAT_ASSERT(m->ref_count > 0);
    m -> ref_count --;
    if (m -> ref_count == 0) mat_free(m);
}

/*
    Define a "free" cousin for clearing the memory once the counter 'ref_count' reaches 0;
    Note: Since mat_alloc uses single allocation, we only free the struct pointer.
*/
void mat_free(Mat *m){
    if (!m) return;
    m->rows = 0;
    m->cols = 0;
    m->stride = 0;
    m->ref_count = 0;
    m->es = NULL;
    free(m);
} 

/*
    Randomize the matrix with values from 'low' to 'high'.
*/
void mat_rand(Mat *m, float low, float high) {
    for (size_t i = 0; i<m->rows; ++i) {
        for (size_t j = 0; j<m->cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }    
}
/*
    Fill the whole matrix with the value 'n'.
*/
void mat_fill(Mat *m, float n) {
    for (size_t i = 0; i<m->rows; ++i) {
        for (size_t j = 0; j<m->cols; ++j) {
            MAT_AT(m, i, j) = n;
        }
    }    
}

/*
    Fill the diagonal of the matrix 'm' with the value 'n'.
*/
void mat_fill_diag(Mat *m, float n) {
    MAT_ASSERT(m->rows == m->cols);
    for (size_t i = 0; i<m->rows; ++i) {
        MAT_AT(m, i, i) = n;
    }    
}


/*
    Extract the row corresponding to the index 'row' from the matrix
    Returns a view (Mat by value) that points into the original matrix.
*/
Mat mat_row(const Mat *m, size_t row) {
    MAT_ASSERT(row < m->rows);

    return (Mat) {
        .rows = 1,
        .cols = m->cols,
        .stride = m->stride,
        .es = &MAT_AT(m, row, 0),
        .ref_count = 0, // Views don't own memory
    };
}

/* 
    Adds a constant value to all elements in the specified row 
*/
void mat_row_sum_const(Mat *m, size_t row, float val){
    MAT_ASSERT(row < m->rows);
    for (size_t j = 0; j < m->cols; j++){
        MAT_AT(m, row, j) += val;
    }
}

/*
    Adds a constant value to all elements in the specified column
*/
void mat_col_sum_const(Mat *m, size_t col, float val){
    MAT_ASSERT(col < m->cols);
    for (size_t i = 0; i < m->rows; i++){
        MAT_AT(m, i, col) += val;
    }
}

/*
    Multiplies all elements in the specified row by a constant value
*/
void mat_row_prod_const(Mat *m, size_t row, float val){
    MAT_ASSERT(row < m->rows);
    for (size_t j = 0; j < m->cols; j++){
        MAT_AT(m, row, j) *= val;
    }
}

/*
    Multiplies all elements in the specified column by a constant value
*/
void mat_col_prod_const(Mat *m, size_t col, float val){
    MAT_ASSERT(col < m->cols);
    for (size_t i = 0; i < m->rows; i++){
        MAT_AT(m, i, col) *= val;
    }
}

/*
    Divides all elements in the specified row by a constant value
*/
void mat_row_div_const(Mat *m, size_t row, float val){
    MAT_ASSERT(val != 0.0);
    MAT_ASSERT(row < m->rows);
    for (size_t j = 0; j < m->cols; j++){
        MAT_AT(m, row, j) /= val;
    }
}

/*
    Divides all elements in the specified column by a constant value
*/
void mat_col_div_const(Mat *m, size_t col, float val){
    MAT_ASSERT(val != 0.0);
    MAT_ASSERT(val < m->cols);
    for (size_t i = 0; i < m->rows; i++){
        MAT_AT(m, i, col) /= val;
    }
}



/* 
    Adds a constant value to all elements in the specified row 
*/
void mat_row_sum(Mat *a, Mat *b, size_t row_1, size_t row_2){
    MAT_ASSERT(a->cols == b->cols);
    MAT_ASSERT(a->rows > row_1);
    MAT_ASSERT(b->rows > row_2);

    for (size_t j = 0; j < a->cols; j++){
        MAT_AT(a, row_1, j) += MAT_AT(b, row_2, j);
    }
}

/*
    Adds a constant value to all elements in the specified column
*/
void mat_col_sum(Mat *a, Mat *b, size_t col_1, size_t col_2){
    MAT_ASSERT(a->rows == b->rows);
    MAT_ASSERT(a->cols > col_1);
    MAT_ASSERT(b->cols > col_2);


    for (size_t i = 0; i < a->rows; i++){
        MAT_AT(a, i, col_1) += MAT_AT(b, i, col_2);
    }
}

/* 
    Adds a constant value to all elements in the specified row 
*/
void mat_row_sub(Mat *a, Mat *b, size_t row_1, size_t row_2){
    MAT_ASSERT(a->cols == b->cols);
    MAT_ASSERT(a->rows > row_1);
    MAT_ASSERT(b->rows > row_2);

    for (size_t j = 0; j < a->cols; j++){
        MAT_AT(a, row_1, j) -= MAT_AT(b, row_2, j);
    }
}

/*
    Adds a constant value to all elements in the specified column
*/
void mat_col_sub(Mat *a, Mat *b, size_t col_1, size_t col_2){
    MAT_ASSERT(a->rows == b->rows);
    MAT_ASSERT(a->cols > col_1);
    MAT_ASSERT(b->cols > col_2);


    for (size_t i = 0; i < a->rows; i++){
        MAT_AT(a, i, col_1) -= MAT_AT(b, i, col_2);
    }
}


/*
    Multiplies all elements in the specified row by a constant value
*/
void mat_row_prod(Mat *a, Mat *b, size_t row_1, size_t row_2){
    MAT_ASSERT(a->cols == b->cols);
    MAT_ASSERT(a->rows > row_1);
    MAT_ASSERT(b->rows > row_2);
    for (size_t j = 0; j < a->cols; j++){
        MAT_AT(a, row_1, j) *= MAT_AT(b, row_2, j);
    }
}

/*
    Multiplies all elements in the specified column by a constant value
*/
void mat_col_prod(Mat *a, Mat *b, size_t col_1, size_t col_2){
    MAT_ASSERT(a->rows == b->rows);
    MAT_ASSERT(a->cols > col_1);
    MAT_ASSERT(b->cols > col_2);
    for (size_t i = 0; i < a->rows; i++){
        MAT_AT(a, i, col_1) *= MAT_AT(b, i, col_2);
    }
}

/*
    Divides all elements in the specified row by a constant value
*/
void mat_row_div(Mat *a, Mat *b, size_t row_1, size_t row_2){
    MAT_ASSERT(a->cols == b->cols);
    MAT_ASSERT(a->rows > row_1);
    MAT_ASSERT(b->rows > row_2);
    for (size_t j = 0; j < a->cols; j++){
        MAT_AT(a, row_1, j) /= MAT_AT(b, row_2, j);
    }
}

/*
    Divides all elements in the specified column by a constant value
*/
void mat_col_div(Mat *a, Mat *b, size_t col_1, size_t col_2){
    MAT_ASSERT(a->rows == b->rows);
    MAT_ASSERT(a->cols > col_1);
    MAT_ASSERT(b->cols > col_2);

    for (size_t i = 0; i < a->rows; i++){
        MAT_AT(a, i, col_1) /= MAT_AT(b, i, col_2);
    }
}


/*
    Copy all of the matrix 'src' values to the 'dst' matrix in place. Both matrices must have the same dimensions.
*/
void mat_copy(Mat *dst, const Mat *src) {
    MAT_ASSERT(dst->rows == src->rows);
    MAT_ASSERT(dst->cols == src->cols);

    for(size_t i = 0; i< dst->rows; ++i) {
        for(size_t j = 0; j< dst->cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

/*
    Set all values of matrix 'dst' to zero.
*/
void mat_zeros(Mat *dst) {
    for (size_t i = 0; i<dst->rows; ++i) {
        for (size_t j = 0; j<dst->cols; ++j) {
            MAT_AT(dst, i, j) = 0;
        }
    }    
}


/*
    Set all values of matrix 'dst' to one.
*/
void mat_ones(Mat *dst) {
    for (size_t i = 0; i<dst->rows; ++i) {
        for (size_t j = 0; j<dst->cols; ++j) {
            MAT_AT(dst, i, j) = 1;
        }
    }       
}
/* ============= MATH BASIC OPERATIONS ============= */
/*
    Sum the values with the same entry indexes across matrices 'dst' and 'a' with the same dimensions. In place operation.
*/
void mat_sum(Mat *dst, const Mat *a) {
    MAT_ASSERT(dst->rows == a->rows);
    MAT_ASSERT(dst->cols == a->cols);

    for (size_t i = 0; i<a->rows; ++i) {
        for (size_t j = 0; j<a->cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }    
    
}

/*
    Subtract the values with the same entry given 'dst' and 'a', two matrices with same dimensions. In place operation.
    The same result can be achieved by using:

    <mat_sum(dst, mat_prod_const(a, -1))> 
*/
void mat_nsum(Mat *dst, const Mat *a) {
    MAT_ASSERT(dst->rows == a->rows);
    MAT_ASSERT(dst->cols == a->cols);

    for (size_t i = 0; i<a->rows; ++i) {
        for (size_t j = 0; j<a->cols; ++j) {
            MAT_AT(dst, i, j) -= MAT_AT(a, i, j);
        }
    }    
    
}

/*
    Sum a constant 'add' to all entries of matrix 'dst' in place.
    If subtraction is needed use a negative value as 'add'
*/
void mat_sum_const(Mat *dst, float add) {
    for (size_t i = 0; i<dst->rows; ++i) {
        for (size_t j = 0; j<dst->cols; ++j) {
            MAT_AT(dst, i, j) += add;
        }
    }    
    
}

/*
    Multiply a constant 'mult' to all entries of matrix 'dst' in place.
*/
void mat_prod_const(Mat *dst, float mult) {
    for (size_t i = 0; i<dst->rows; ++i) {
        for (size_t j = 0; j<dst->cols; ++j) {
            MAT_AT(dst, i, j) *= mult;
        }
    }    
}

/*
    Divide all entries by a constant 'div'
*/
void mat_div_const(Mat *dst, float div) {
    MAT_ASSERT(div!=0.0);
    for (size_t i = 0; i<dst->rows; ++i) {
        for (size_t j = 0; j<dst->cols; ++j) {
            MAT_AT(dst, i, j) /= div;
        }
    }    
}

/*
    Raise to a constant exponent 'exp' all entries of matrix 'dst' in place.
*/
void mat_exp(Mat *dst, float exp) {
    for (size_t i = 0; i<dst->rows; ++i) {
        for (size_t j = 0; j<dst->cols; ++j) {
            MAT_AT(dst, i, j) = powf(MAT_AT(dst, i, j), exp);
        }
    }    
}

/*
    Compute the sigmoid function for all the values in the matrix 'm' in place.
*/
void mat_sig(Mat *m) {
    for (size_t i = 0; i<m->rows; ++i) {
        for (size_t j = 0; j<m->cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

/*
    Split the matrix in two by rows
    Returns an array of 2 Mat* pointers. Caller must free the array and release the matrices.
*/
Mat **mat_split_rows(const Mat *m, float x){
    MAT_ASSERT(0.0 < x && x < 1.0);
    MAT_ASSERT(m->rows > 1);
    

    size_t size_1st = (size_t)(m->rows * x);
    size_t size_2nd = m->rows - size_1st;
    Mat *m_1st = mat_alloc(size_1st, m->cols);
    Mat *m_2nd = mat_alloc(size_2nd, m->cols);
    // Because 'out' will contain two pointers to two matrices
    Mat **out = MAT_MALLOC(2 * sizeof(Mat*));

    for(size_t i = 0; i < size_1st; i++){
        for(size_t j = 0; j < m->cols; j++){
            MAT_AT(m_1st,i,j) = MAT_AT(m,i,j);
        }
    }

    for(size_t i = size_1st; i < m->rows; i++){
        for(size_t j = 0; j < m->cols; j++){
            MAT_AT(m_2nd,i - size_1st,j) = MAT_AT(m,i,j);
        }
    }
    out[0] = m_1st;
    out[1] = m_2nd;
    return out;
}

/*
    Split the matrix in two by cols
    Returns an array of 2 Mat* pointers. Caller must free the array and release the matrices.
*/
Mat **mat_split_cols(const Mat *m, float x){
    MAT_ASSERT(0.0 < x && x < 1.0);
    MAT_ASSERT(m->cols > 1);
    size_t size_1st = (size_t)(m->cols * x);
    size_t size_2nd = m->cols - size_1st;
    Mat *m_1st = mat_alloc(m->rows, size_1st);
    Mat *m_2nd = mat_alloc(m->rows, size_2nd);
    Mat **out = MAT_MALLOC(2 * sizeof(Mat*));

    for(size_t i = 0; i < m->rows; i++){
        for(size_t j = 0; j < size_1st; j++){
            MAT_AT(m_1st,i,j) = MAT_AT(m,i,j);
        }
    }

    for(size_t i = 0; i < m->rows; i++){
        for(size_t j = size_1st; j < m->cols; j++){
            MAT_AT(m_2nd,i,j-size_1st) = MAT_AT(m,i,j);
        }
    }
    out[0] = m_1st;
    out[1] = m_2nd;
    return out;
}

/*
    It appends one matrix to another having the same cols
*/
Mat *mat_append_rows(const Mat *a, const Mat *b){
    MAT_ASSERT(a->cols == b->cols);
    size_t rows = a->rows + b->rows;
    size_t cols = a->cols;
    Mat *out = mat_alloc(rows, cols);

    for(size_t i = 0; i < a->rows; i++){
        for(size_t j = 0; j < cols; j++){
            MAT_AT(out,i,j) = MAT_AT(a,i,j);
        }
    }
    
    for(size_t i = 0; i < b->rows; i++){
        for(size_t j = 0; j < cols; j++){
            MAT_AT(out,i+a->rows,j) = MAT_AT(b,i,j);
        }
    }
    return out;
}

/*
    It appends one matrix to another having the same rows
*/
Mat *mat_append_cols(const Mat *a, const Mat *b){
    MAT_ASSERT(a->rows == b->rows);
    size_t rows = a->rows;
    size_t cols = a->cols + b->cols;
    Mat *out = mat_alloc(rows, cols);

    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < a->cols; j++){
            MAT_AT(out,i,j) = MAT_AT(a,i,j);
        }
    }
    
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < b->cols; j++){
            MAT_AT(out,i,j+a->cols) = MAT_AT(b,i,j);
        }
    }
    return out;
}

/*
    Swap two rows
*/
void mat_swap_rows(Mat *m, size_t row_1, size_t row_2){
    MAT_ASSERT(m->rows > row_1);
    MAT_ASSERT(m->rows > row_2);
    MAT_ASSERT(row_1 != row_2);


    for(size_t j = 0; j < m->cols; j++){
        float temp = MAT_AT(m, row_1, j);
        MAT_AT(m,row_1,j) = MAT_AT(m,row_2,j);
        MAT_AT(m,row_2,j) = temp;
    }

}

/*
    Swap two cols
*/
void mat_swap_cols(Mat *m, size_t col_1, size_t col_2){
    MAT_ASSERT(m->cols > col_1);
    MAT_ASSERT(m->cols > col_2);
    MAT_ASSERT(col_1 != col_2);

    for(size_t i = 0; i < m->rows; i++){
        float temp = MAT_AT(m, i, col_1);
        MAT_AT(m,i,col_1) = MAT_AT(m,i,col_2);
        MAT_AT(m,i,col_2) = temp;
    }
}
/* ============= ADVANCED MATH OPERATIONS ============= */
/*
    Compute matrix dot product in place.
*/
void mat_dot(Mat *dst, const Mat *a, const Mat *b) {
    MAT_ASSERT(a->cols == b->rows);
    size_t n = a->cols;
    MAT_ASSERT(dst->rows == a->rows);
    MAT_ASSERT(dst->cols == b->cols);

    for (size_t i = 0; i<dst->rows; ++i) {
        for (size_t j = 0; j<dst->cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k< n; ++k) {
                // Multiply each item of a row belonging to matrix 'a' with each item belonging to column 'b'
                // Sum the results to get the item in the new matrix.
                MAT_AT(dst, i, j) += MAT_AT(a, i, k)*MAT_AT(b, k, j);
            }
        }
    }    
    
}

/*
    Compute the transpose of matrix 'm'. Basically inverting indexes between rows and cols.
*/
Mat *mat_trans(const Mat *m) {
    Mat *o = mat_alloc(m->cols, m->rows);

    for (size_t i = 0; i<m->rows; ++i) {
        for (size_t j = 0; j<m->cols; ++j) {
            MAT_AT(o, j, i) = MAT_AT(m, i, j);
        }
    }
    return o;
}

/*
    A function that prints the structure of the matrix as usually represented in math textbooks.
*/
void mat_print(const Mat *m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i<m->rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j<m->cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}



/* 
    Given two matrices of the same size 'a' and 'b', return the Residual Sum Squared (RSS)
*/
Mat *mat_SS(const Mat *a, const Mat *b){
    Mat *res = mat_alloc(a->rows, a->cols);
    mat_copy(res, a);
    mat_nsum(res, b);
    mat_exp(res, 2);
    Mat *o = mat_row_collapse_sum(res);
    mat_release(res);
    return o;
}

/* 
    Given one matrix 'a' and one constant 'x', return the Residual Sum Squared (RSS) w.r.t. a given constant
*/
Mat *mat_SS_const(const Mat *a, float x){
    Mat *res = mat_alloc(a->rows, a->cols);
    mat_copy(res, a);
    mat_sum_const(res, -x);
    mat_exp(res, 2);
    Mat *o = mat_row_collapse_sum(res);
    mat_release(res);
    return o;
}

/*
    from RxC to 1xC for each col, sum across all rows.
*/
Mat *mat_col_collapse_sum(const Mat *m) {
    Mat *col_m = mat_alloc(1, m->cols);

    for (size_t j = 0; j < m->cols; j++){
        float temp = 0.0;
        for (size_t i = 0; i < m->rows; i ++){
            temp += MAT_AT(m, i, j);
        }
        MAT_AT(col_m, 0, j) = temp;
    }

    return col_m;
}

/*
    from RxC to Rx1 for each row, sum across all cols.
*/
Mat *mat_row_collapse_sum(const Mat *m) {
    Mat *row_m = mat_alloc(1, m->rows);

    for (size_t i = 0; i < m->rows; i ++){
        float temp = 0.0;
        for (size_t j = 0; j < m->cols; j++){
            temp += MAT_AT(m, i, j);
        }
        MAT_AT(row_m, 0, i) = temp;
    }
    return row_m;
}

/*
    Mean of each row.
*/
Mat *mat_row_mean(const Mat *m){
    Mat *o = mat_row_collapse_sum(m);
    MAT_ASSERT(m->rows != 0);
    mat_div_const(o, m->cols);
    return o;
}

/*
    Mean of each col
*/
Mat* mat_col_mean(const Mat *m){
    Mat *o = mat_col_collapse_sum(m);
    MAT_ASSERT(m->rows != 0);
    mat_div_const(o, m->rows);
    return o;
}

/*
    Computes the Inverse of a square matrix using the Gauss-Jordan Inversion
*/
Mat *mat_inverse_GJ(const Mat *m){
    MAT_ASSERT(m->rows == m->cols);
    size_t dim = m->rows;
    
    // Create Identity
    Mat *m_id = mat_alloc(dim, dim);
    mat_zeros(m_id);
    mat_fill_diag(m_id, 1.0f);
    
    // Create Augmented Matrix
    Mat *m_aug = mat_append_cols(m, m_id);
    mat_release(m_id); 

    for(size_t j = 0; j < dim; j++){
        
        // Move and select pivots
        float max_val = MAT_AT(m_aug, j, j);
        size_t idx = j;
        
        for(size_t i = j+1; i < dim; i++){
            float val = MAT_AT(m_aug, i, j);
            if(val > max_val){
                max_val = val;
                idx = i;
            }
        }
        
        MAT_ASSERT(max_val > 1e-6);
        
        if(idx != j){
            mat_swap_rows(m_aug, idx, j);
        }

        // Normalize the pivot
        float pivot = MAT_AT(m_aug, j, j);
        mat_row_div_const(m_aug, j, pivot);
        
        // Set to zero the values outside the diagonal
        for(size_t i = 0; i < dim; i++){
            // Skip the pivot row itself
            if(i == j) continue;
            
            float factor = MAT_AT(m_aug, i, j);
            
            if(fabs(factor) < 1e-6) continue; 

            // we perform: Row_i = Row_i - (factor * Row_j)
            for(size_t k = 0; k < m_aug->cols; k++){
                MAT_AT(m_aug, i, k) -= factor * MAT_AT(m_aug, j, k);
            }
        
        }
    }

    Mat *inv = mat_alloc(dim, dim);
    for(size_t i = 0; i < dim; i++){
        for(size_t k = 0; k < dim; k++){
            // The inverse is in the right half (columns dim to 2*dim)
            MAT_AT(inv, i, k) = MAT_AT(m_aug, i, k + dim);
        }
    }
    
    mat_release(m_aug);
    return inv;
}

#endif

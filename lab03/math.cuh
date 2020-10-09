#pragma once

__host__ __device__
void revert_sign(float *a, size_t dim1, size_t dim2);


__host__ __device__
void sum_vectors(
        float *a, 
        float *b, 
        float *result, 
        size_t dim1,
        size_t dim2
);



__device__
void sum_vectors_v(
        volatile float *a, 
        volatile float *b, 
        volatile float *result, 
        size_t dim1,
        size_t dim2
);


__host__ __device__
void mul_vectors(
        float *a, 
        float *b, 
        float *result, 
        size_t a_dim1,
        size_t a_dim2,
        size_t b_dim2
);


float matrix_norm(float *m, size_t dim1, size_t dim2);


void inverse_matrix(float *m_, float *result, size_t dim);

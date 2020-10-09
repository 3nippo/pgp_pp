#include <cmath>

__host__ __device__
void revert_sign(float *a, size_t dim1, size_t dim2)
{
    for (size_t i = 0; i < dim1 * dim2; ++i)
        a[i] = -a[i];
}


__host__ __device__
void sum_vectors(
        float *a, 
        float *b, 
        float *result, 
        size_t dim1,
        size_t dim2
)
{
    for (size_t i = 0; i < dim1; ++i)
        for (size_t j = 0; j < dim2; ++j)
        {
            size_t idx = i * dim2 + j;
            result[idx] = a[idx] + b[idx];
        }
}


__device__
void sum_vectors_v(
        volatile float *a, 
        volatile float *b, 
        volatile float *result, 
        size_t dim1,
        size_t dim2
)
{
    for (size_t i = 0; i < dim1; ++i)
        for (size_t j = 0; j < dim2; ++j)
        {
            size_t idx = i * dim2 + j;
            result[idx] = a[idx] + b[idx];
        }
}


__host__ __device__
void mul_vectors(
        float *a, 
        float *b, 
        float *result, 
        size_t a_dim1,
        size_t a_dim2,
        size_t b_dim2
)
{
    for (size_t i = 0; i < b_dim2; ++i)
        for (size_t j = 0; j < a_dim1; ++j)
        {
            float current_value = 0;

            for (size_t k = 0; k < a_dim2; ++k)
                current_value += a[j * a_dim2 + k] * b[k * b_dim2 + i];

            result[j * b_dim2 + i] = current_value;
        }
}


float matrix_norm(float *m, size_t dim1, size_t dim2) 
{ 
    float max_column_norm = 0;

    for (size_t i = 0; i < dim1; ++i)
    {
        float column_norm = 0;

        for (size_t j = 0; j < dim2; ++j)
            column_norm += std::fabs(m[i * dim2 + j]);

        max_column_norm = column_norm > max_column_norm ? column_norm : max_column_norm;
    }

    return max_column_norm;
}


void inverse_matrix(float *m_, float *result, size_t dim) 
{
    float inverse_matrix[dim * dim], m[dim * dim];
    
    // init accessory matrices
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
        {
            if (i == j)
                inverse_matrix[i * dim + j] = 1;
            else
                inverse_matrix[i * dim + j] = 0;

            m[i * dim + j] = m_[i * dim + j];
        }
    
    // forward stroke
    for (size_t i = 0; i < dim; ++i)
    {
        for (size_t j = i + 1; j < dim; ++j)
        {
            float multiplier = m[j * dim + i] / m[i * dim + i];

            for (size_t k = i; k < dim; ++k)
                m[j * dim + k] -= m[i * dim + k] * multiplier;
            
            for (size_t k = 0; k < dim; ++k)
                inverse_matrix[j * dim + k] -= inverse_matrix[i * dim + k] * multiplier;
        }

        float divisor = m[i * dim + i];

        for (size_t j = i; j < dim; ++j)
            m[i * dim + j] /= divisor;

        for (size_t j = 0; j < dim; ++j)
            inverse_matrix[i * dim + j] /= divisor;
    }

    // return stroke
    for (size_t i = 1; i <= dim; ++i)
        for (size_t j = i + 1; j <= dim; ++j)
        {
            float multiplier = m[(dim - j) * dim + dim - i];

            m[(dim - j) * dim + dim - i] -= m[(dim - i) * dim + dim - i] * multiplier;

            for (size_t k = 0; k < dim; ++k)
                inverse_matrix[(dim - j) * dim + k] -= inverse_matrix[(dim - i) * dim + k] * multiplier;
        }
    
    // copy inverse matrix to result
    for (size_t i = 0; i < dim*dim; ++i)
        result[i] = inverse_matrix[i];
}

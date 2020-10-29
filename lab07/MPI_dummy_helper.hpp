#pragma once

#include <iostream>
#include <string>
#include <type_traits>
#include <stdexcept>

#include <mpi.h>

template <typename T>
void read_in_container(T &container, size_t n=0)
{
    static_assert(
        std::is_same<
            typename std::iterator_traits<typename T::iterator>::iterator_category,
            std::random_access_iterator_tag
        >::value,
        "Wrong container"
    );

    if (n == 0)
        n = container.size();

    for (typename T::iterator it = container.begin(); n && it != container.end(); ++it, --n)
    {
        typename T::value_type a;

        std::cin >> a;

        *it = a;
    }
}

#define checkMPIErrors(val) _checkMPIErrors((val), #val, __FILE__, __LINE__)

template<typename T>
void _checkMPIErrors(
    T error_code, 
    const char* const func, 
    const char* const file, 
    const int line
) 
{
    if (error_code != MPI_SUCCESS) 
    {
        std::string error_message;

        error_message =
            std::string("Error occured: ")
            + file 
            + ":" 
            + std::to_string(line)
            + "\n";
        
        int error_class;
        
        MPI_Error_class(error_code, &error_class);
        
        char error_string_buffer[BUFSIZ];
        int error_string_size;

        MPI_Error_string(
            error_class, 
            error_string_buffer, 
            &error_string_size
        );

        error_message = error_message + "Error class:" + error_string_buffer + "\n";
        
        MPI_Error_string(
            error_code, 
            error_string_buffer, 
            &error_string_size
        );

        error_message = error_message + "Error text:" + error_string_buffer + "\n";

        throw std::runtime_error(error_message);
    }
}

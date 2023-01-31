## Introduction

Sparse matrices are a key component of the performance critical computations in many numerical simulations. A desire to represent sparse matrices efficiently in memory and optimise , in particular given the evolution of hardware architectures, has over the years led to the development of a plethora of sparse matrix storage formats. Each format is designed to exploit the particular strengths of an architecture or the specific sparsity pattern of a matrix. The choice of the format can be crucial in order to achieve optimal performance. Being able to dynamically select storage formats at runtime is therefore highly desirable.

`Morpheus`, is a library of sparse matrix storage formats that is designed for efficient and transparent format switching across architectures without introducing prohibitive overheads and that, at the same time, enables the straightforward addition of new storage formats without major application code changes. The library has a functional design which separates the containers that implement the storage formats from the algorithm that implement operations such as multiplication. Each container is aware of the memory space it resides in, and the algorithms require knowledge of the execution space that they will be running in. This design allows Morpheus to target multiple heterogeneous architectures. 

To enable efficient yet flexible dynamic switching, Morpheus uses the `std::variant` introduced in C++17 - hence it requires a **C++17** compliant compiler - as a type-safe union. It offers a polymorphic variable called `DynamicMatrix` which abstracts the notion of sparse matrix storage formats away from users and can hold any of the supported formats internally.

## Installation

A very basic installation using CMake is done with:

```sh
$ cmake ${srcdir} \
   -DCMAKE_CXX_COMPILER=g++ \
   -DCMAKE_INSTALL_PREFIX=${my_install_folder}
   -DKokkos_ROOT=${kokkos_install_prefix}
$ make
$ make install
```

which configures, builds and installs a default Morpheus. Note that `Morpheus` **REQUIRES** [Kokkos v.3.5.0](https://github.com/kokkos/kokkos/tree/3.5.00) to be installed. Note that `Morpheus` inherits the enabled devices and compiler optimization flags from `Kokkos`.

Once Morpheus is installed In your `CMakeLists.txt` simply use:

```cmake
find_package(Morpheus REQUIRED)
```

Then for every executable or library in your project:
```cmake
target_link_libraries(myTarget Morpheus::morpheus)
```

More information can be found at [BUILD.md](BUILD.md)

## Documentation

Documentation can be found [here](morpheus-org.github.io/morpheus).

### Supported Formats

| Format | Container | Serial | OpenMP | CUDA | HIP | Kokkos |
| ------ | --------- | ------ | ------ | ---- | --- | ------ |
| Coo    | CooMatrix | yes    | yes    | yes  | yes | no     |
| Csr    | CsrMatrix | yes    | yes    | yes  | yes | yes    |
| Dia    | DiaMatrix | yes    | yes    | yes  | yes | yes    |

### Specifying a container

To define a container we need to specify four template parameters:
- `ValueType`: The type of the values the container will hold. Valid types must satisfy `std::is_arithmetic` i.e to be an arithmetic type.
- `IndexType`: The type of the indices the container will hold. Valid types must satisfy `std::is_integral` i.e to be an integral type.
- `Layout`: Orientation of data in memory. Valid layouts are either  `Kokkos::LayoutLeft` (Column-Major) or `Kokkos::LayoutRight` (Row-Major).
- `Space`: A memory or execution space supported by Morpheus. 
  - Valid *Memory* Spaces are `HostSpace`, `CudaSpace` and `HIPSpace`. 
  - Valid *Execution* Spaces are `Serial`, `OpenMP`, `Cuda` and `HIP`. 
  - Spaces can be **Generic** or **Custom** by specifying the appropriate namespace eg: `Morpheus::Generic::HostSpace` and `Morpheus::Custom::HostSpace` represent the generic and custom Host Memory space respectively. 
  - Note that specifying the execution space will determine in which space each member function will be executed. By not providing a space parameter `Morpheus` will choose a default one for you.

Note that only `ValueType` is mandatory. For the rest of the arguments, if not provided, sensible defaults will be selected.

```cpp
#include <Morpheus_Core.hpp>

int main(){
    /* 
     * ValueType        : double
     * IndexType        : long long
     * Layout           : Kokkos::LayoutRight
     * MemorySpace      : Morpheus::HostSpace 
     */
    Morpheus::CooMatirx<double, long long, Kokkos::LayoutRight, Morpheus::HostSpace> A;  

    /* 
     * ValueType        : double
     * IndexType        : int (Default)
     * Layout           : Kokkos::LayoutRight (Default), 
     * MemorySpace      : Morpheus::DefaultSpace (Default) 
     */
    Morpheus::CsrMatirx<double> B; 
}
```

### Using an Algorithm

For each algorithm the same interface is used across different formats. Algorithms are aware of the execution space they will be executed in and dispatch depends on that too. Currently we support the following algorithms for each of the supported storage formats:
- Multiply (Sparse Matrix-Vector Multiplication)
- Copy
- Convert
- Copy by key
- Print
- Update diagonal
- Read/Write matrix market

```cpp
#include <Morpheus_Core.hpp>

int main(){
    // [ 3.5   *   * ]
    // [  *   1.5  * ]
    Morpheus::CooMatirx<double,  Kokkos::HostSpace> A(2, 3, 2);  
    Morpheus::DenseVector<double,  Kokkos::HostSpace> x(3, 0), y(2, 0); 
    
    // Initializing A
    A.row_indices(0) = 0;
    A.column_indices(0) = 0;
    A.values(0) = 3.5;

    A.row_indices(1) = 1;
    A.column_indices(1) = 1;
    A.values(1) = 1.5;

    // Initializing x
    x(0) = 1; x(1) = 2; x(2) = 3;

    // y = A * x
    Morpheus::Multiply<Morpheus::Serial>(A, x, y);
}
```

### Use of dynamic matrix

The dynamic matrix tries to absrtract away the different supported formats and provide an efficient yet flexible switching mechanism between them. The dynamic matrix follows the same interface as the other containers hence algorithms can be used in the same way.

```cpp
#include <Morpheus_Core.hpp>

int main(){
    Morpheus::DynamicMatirx<double> A;  // Default format is COO

    A.activate(Morpheus::CSR_FORMAT)    // Active type now is CSR

    A.activate(Morpheus::DIA_FORMAT)    // Active type now is DIA

    A.activate(0)    // Active type now is the first in the DynamicMatrix
}
```

## Building and Running the Tests and Examples

Building the tests and examples requires the [GTest](https://github.com/google/googletest) testing framework, which ships together with `Morpheus`.

To build and run the tests add the `-DMorpheus_ENABLE_TESTS=On` during configuration stage respectively. Note that tests for different devices are enabled based on how `Kokkos` was configured.

After configuration, to build and run the `Serial` tests do:
```sh
$ cd  ${srcdir}
$ make MorpheusCore_UnitTest_Serial
$ ${srcdir}/core/tests/MorpheusCore_UnitTest_Serial
```
Same process can be followed for `OpenMP`, `Cuda` and `HIP`.

To build and run the examples add the `-DMorpheus_ENABLE_EXAMPLES=On` during configuration. Similar process is followed as building the tests.

# License

This software is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
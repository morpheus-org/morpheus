# Changelog

## Current - Version 0.3.1
- Added updating main diagonal of sparse matrix in MatrixOperations. Update is only performed on non-zero elements of the main diagonal i.e doesn't change the sparsity pattern of the matrix.
- Added interface to inspect the sparse formats using their enum tag and facilitate format switching for dynamic matrix.
- Fixed errors in deep copy semantics for the `DynamicMatrix`.


## Version 0.3.0
- Build `morpheus` as a package with different subpackages.
- Enabled test environment for the core functionality using `googletest` framework.
- Enabled GPU support through the Host-Device model.
- Support for container mirroring through `create_mirror` and `create_mirror_container` as well as `HostMirror` trait.
- Efficient deep copies for containers across same/different memory spaces through the `Morpheus::copy` routine.
- Added `Dot` and `WAXPBY` alogirhtms for `DenseVector`.
- Added `Reduction` and `Scan` routines for `DenseVector`.
- Added `convert` algorithms for conversion across different containers on Host space.
- Algorithm dispatch through `Kokkos::Serial`, `Kokkos::OpenMP` and `Kokkos::Cuda` for the custom kernels.
- Added `Morpheus::Serial`, `Morpheus::OpenMP` and `Morpheus::Cuda` to be used for dispatching Kokkos kernels in algorithms.
- `SpMV` implementation for `CooMatrix`, `CsrMatrix` and `DiaMatrix` available for `Kokkos::Serial`, `Kokkos::OpenMP`, `Kokkos::Cuda` and Kokkos equivalent spaces - except for currently the Kokkos implementation for `CooMatrix`.
- Enabled shallow copy semantics between `DynamicMatrix` and concrete containers based on the active type of the `DynamicMatrix`.
- Enabled mirroring and deep copy semantics for the `DynamicMatrix`.

## Version 0.2.0
- Introduces `DynamicMatrix` container, a polymorphic container that holds a closed set of Sparse Matrix Storage Formats.
- Use of variadic templates to create a uniform interface under `DynamicMatrix` container for operations like `resize()`, which for different formats can have a different signature.
- Use of metaprogramming techniques for generalising the run-time switching across formats.
- Use of metaprogramming techniques to examine the traits of a vector and matrix.
- Added implementation of `DenseVector` and `DenseMatrix`
- Added implementation of `CooMatrix`, `CsrMatrix` and `DiaMatrix`.
- Kokkos integration to use the different memory and execution spaces.
- Interface for dispatching `multiply` and `print` operations using tag dispatching to distinguish amonst formats.
- Extended interface to dispatch operations to different architectures using SFINAE. Each container should reside in the same execution space.

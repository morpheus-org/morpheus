# Changelog

## Current

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

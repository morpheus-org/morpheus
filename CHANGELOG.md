# Changelog

## Current
- Introduces `DynamicMatrix` container, a polymorphic container that holds a closed set of Sparse Matrix Storage Formats.
- Use of variadic templates to create a uniform interface under `DynamicMatrix` container for operations like `resize()`, which for different formats can have a different signature.
- Use of metaprogramming techniques for generalising the switching across formats as well as providing template argument type checking.

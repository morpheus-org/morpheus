## Introduction

Intro about sparse matrix storage formats...

What `morpheus` provides...

`morpheus` requires a C++17 compliant compiler. The following C++ compilers are supported:

Compiler                | Version
------------------------|-------------------------------
ex1 | v1
ex2 | v2


## Installation

Although `morpheus` is a header-only library, we provide standardized means to install it with cmake.

Besides the `morpheus` headers, all these methods place the `CMake` project configuration file in the right location so that
third-party projects can use cmake's `find_package` to locate xsimd headers.

### Install from sources

You can directly install it from the sources with cmake:

```bash
mkdir build
cd build
cmake .. -D CMAKE_INSTALL_PREFIX=your_install_prefix
make install
```

## Documentation

To get started with using `morpheus`, check out the full documentation

http://link-to-documentation/

## Usage

### Use of dynamic matrix

## Building and Running the Tests

Building the tests requires the [Catch2](https://github.com/catchorg/Catch2) testing framework and [cmake](https://cmake.org).

Catch2 and cmake are available as a packages for most linux distributions. 

Once `Catch2` and `cmake` are installed, you can build and run the tests:

```bash
mkdir build
cd build
cmake ../ -DMorpheus_BUILD_TESTS=ON
make test
```

In the context of continuous integration(shall we?) with Travis CI...

## Building the HTML Documentation

`morpheus`' documentation is built with [doxygen](http://www.doxygen.org) which must be installed separately,.

Finally, build the documentation with

```bash
make html
```

from the `doc` subdirectory.

## License

This software is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
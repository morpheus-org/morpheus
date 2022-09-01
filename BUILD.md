# Installing and Using Morpheus

You can either use Morpheus as an installed package (encouraged) or use Morpheus in-tree in your project. Once Morpheus is installed In your `CMakeLists.txt` simply use:

```cmake
find_package(Morpheus REQUIRED)
```

Then for every executable or library in your project:

```cmake
target_link_libraries(myTarget Morpheus::morpheus)
```

## Configuring CMake

A very basic installation is done with:

```
cmake ${srcdir} \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_INSTALL_PREFIX=${my_install_folder}
 -DKokkos_ROOT=${kokkos_install_prefix}
```
which builds and installs a default Morpheus when you run `make install`. Note that `Morpheus` **REQUIRES** Kokkos to be installed. There is no checking Morpheus preprocessor, compiler, or linker flags. CMake propagates all the necesssary flags to your project. This means not only is linking to Morpheus easy, but Morpheus itself can actually configure compiler and linker flags for your project. If building in-tree, there is no find_package. In fact, you only ever need to link to Morpheus and not Kokkos! Linking to Morpheus transitively provides Kokkos.

There are also option Third-party Libraries (TPLs)
```
cmake ${srcdir} \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_INSTALL_PREFIX=${my_install_folder} \
 -DMorpheus_ENABLE_TPL_CUBLAS=ON
```
which, e.g. activates the CUBLAS dependency. The full keyword listing is below.

## Morpheus CMake Option Listing
<!-- TODO -->


## Archer 

### Setup environment - Cray
```sh
$ module restore PrgEnv-cray
$ module load cmake
$ export CRAYPE_LINK_TYPE=dynamic
$ CXX_COMPILER=/opt/cray/pe/craype/2.7.2/bin/CC
$ KOKKOS_INSTALL_DIR=/install/path/of/kokkos/with/cray
$ MORPHEUS_INSTALL_DIR=/install/path/of/morpheus/with/cray
```

### Setup environment - GNU
```sh
$ module restore PrgEnv-gnu
$ module load cmake
$ export CRAYPE_LINK_TYPE=dynamic
$ CXX_COMPILER=/opt/cray/pe/craype/2.7.2/bin/CC
$ KOKKOS_INSTALL_DIR=/install/path/of/kokkos/with/gnu
$ MORPHEUS_INSTALL_DIR=/install/path/of/morpheus/with/gnu
```

### Setup environment - AMD
```sh
$ module restore PrgEnv-aocc
$ module load cmake
$ export CRAYPE_LINK_TYPE=dynamic
$ CXX_COMPILER=/opt/cray/pe/craype/2.7.2/bin/CC
$ KOKKOS_INSTALL_DIR=/install/path/of/kokkos/with/aocc
$ MORPHEUS_INSTALL_DIR=/install/path/of/morpheus/with/aocc
```

### Installing Kokkos
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR} \
           -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON  -DKokkos_ENABLE_SERIAL=ON \
           -DKokkos_CXX_STANDARD=17 -DKokkos_ENABLE_COMPILER_WARNINGS=On -DKokkos_ARCH_ZEN2=On \
           -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=On
$ make
$ make install
```

## Cirrus - CPU

### Setup environment - Intel
```sh
$ module load cmake
$ module load intel-20.4/compilers
$ module swap gcc/6.3.0 gcc/10.2.0
$ CXX_COMPILER=/lustre/sw/intel/compilers_and_libraries_2020.4.304/linux/bin/intel64/icpc
$ KOKKOS_INSTALL_DIR=/install/path/of/kokkos/with/intel
$ MORPHEUS_INSTALL_DIR=/install/path/of/morpheus/with/intel
```
### Setup environment - GNU
```sh
$ module load cmake
$ module load gcc/10.2.0
$ CXX_COMPILER=/lustre/sw/gcc/10.2.0/bin/g++
$ KOKKOS_INSTALL_DIR=/install/path/of/kokkos/with/gnu
$ MORPHEUS_INSTALL_DIR=/install/path/of/morpheus/with/gnu
```

### Installing Kokkos
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR} \
           -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON  -DKokkos_ENABLE_SERIAL=ON \
           -DKokkos_CXX_STANDARD=17 -DKokkos_ENABLE_COMPILER_WARNINGS=On -DKokkos_ARCH_BDW=On \
           -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=On
$ make
$ make install
```

## Cirrus - GPU

### Setup environment
```sh
$ module load cmake
$ module load nvidia/cuda-11.0
$ module swap gcc/6.3.0 gcc/8.2.0 # otherwise complaints for not compatible with c++17 features
$ CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper
$ KOKKOS_INSTALL_DIR=/install/path/of/kokkos/with/cuda-11.0
$ MORPHEUS_INSTALL_DIR=/install/path/of/morpheus/with/cuda-11.0
```

## Local Machine
```sh
$ CXX_COMPILER=/usr/local/bin/g++-11
$ KOKKOS_INSTALL_DIR=/install/path/of/kokkos/with/gnu
$ MORPHEUS_INSTALL_DIR=/install/path/of/morpheus/with/gnu
```

### Installing Kokkos
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR} \
           -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON  -DKokkos_ENABLE_SERIAL=ON \
           -DKokkos_CXX_STANDARD=17 -DKokkos_ENABLE_COMPILER_WARNINGS=On -DKokkos_ARCH_VOLTA70=On -DKokkos_ARCH_SKX=On \
           -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=On -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On
$ make
$ make install
```
**Warning** For cmake to find the cuda drivers you need to have a visible NVIDIA GPU. See [here](#gpu_interactive) how to create an interactive GPU session on Cirrus.

**Warning** For C++17 support requires NVCC 11.0+ or Clang 4.0+

**Warning** For using lambdas on device `Kokkos_ENABLE_CUDA_CONSTEXPR` and `Kokkos_ENABLE_CUDA_LAMBDA` must be enabled.

## Installing Morpheus
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${MORPHEUS_INSTALL_DIR} \
           -DKokkos_ROOT=${KOKKOS_INSTALL_DIR} -DCMAKE_BUILD_TYPE=Release \
           -DMorpheus_ENABLE_EXAMPLES=On -DMorpheus_ENABLE_TESTS=On
$ make
$ make install
```

**Warning** The installed Kokkos configuration does not support CXX extensions when build with `CUDA` enabled. Add the `-DCMAKE_CXX_EXTENSIONS=Off` to surpress warning.

## P3 - MI100

### Setup environment
```sh
$ module use /lustre/projects/bristol/modules/modulefiles
$ module load cmake/3.23.2
$ module load amd/4.5.1
$ CXX_COMPILER=$(which hipcc)
$ KOKKOS_INSTALL_DIR="/lustre/home/ri-cstylianou/morpheus-benchmarks/tpl/kokkos/installs/release-mi100-off-hip-4.5-off-hip"
$ MORPHEUS_INSTALL_DIR="/lustre/home/ri-cstylianou/morpheus-benchmarks/tpl/morpheus/installs/release-mi100-off-hip-4.5-off-hip"
```

### Installing Kokkos
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR} \
           -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_HIP=On -DKokkos_ENABLE_CUDA=Off -DKokkos_ENABLE_OPENMP=Off  -DKokkos_ENABLE_SERIAL=ON \
           -DKokkos_CXX_STANDARD=17 -DKokkos_ENABLE_COMPILER_WARNINGS=On -DKokkos_ARCH_ZEN3=On -DKokkos_ARCH_VEGA908=On \
           -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=On 
           
$ make
$ make install
```

## Installing Morpheus
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${MORPHEUS_INSTALL_DIR} \
           -DKokkos_ROOT=${KOKKOS_INSTALL_DIR} -DCMAKE_BUILD_TYPE=Release \
           -DMorpheus_ENABLE_EXAMPLES=Off -DMorpheus_ENABLE_TESTS=On -DMorpheus_ENABLE_INDIVIDUAL_TESTS=On
$ make
$ make install
```

```sh
module use /lustre/projects/bristol/modules/modulefiles
module load cmake
module load amd/4.5.1
CXX_COMPILER=$(which hipcc)
KOKKOS_INSTALL_DIR="/lustre/home/ri-cstylianou/morpheus-benchmarks/tpl/kokkos/installs/release-mi100-off-hip-4.5-off-hip"
MORPHEUS_INSTALL_DIR="/lustre/home/ri-cstylianou/morpheus-benchmarks/tpl/morpheus/installs/release-mi100-off-hip-4.5-off-hip"

cmake ../.. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${MORPHEUS_INSTALL_DIR} \
           -DKokkos_ROOT=${KOKKOS_INSTALL_DIR} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_EXTENSIONS=Off \
           -DMorpheus_ENABLE_EXAMPLES=Off -DMorpheus_ENABLE_TESTS=On -DMorpheus_ENABLE_INDIVIDUAL_TESTS=On \
           -DGTest_ROOT=/lustre/home/ri-cstylianou/googletest/install
```

## Valgrind Memcheck
```sh
$  valgrind -s --tool=memcheck --leak-check=full --track-origins=yes /path/to/exe
```

### Installing Kokkos with Bound Checks
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR} \
           -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_OPENMP=ON  -DKokkos_ENABLE_SERIAL=ON \
           -DKokkos_CXX_STANDARD=17 -DKokkos_ENABLE_COMPILER_WARNINGS=On -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=On
$ make
$ make install
```

## Interactive Sessions on Cirrus
### GPU Session <a name="gpu_interactive"></a>
```sh
$ ACCOUNT_ID=YOUR_PROJECT_ID
$ srun --exclusive --nodes=1 --time=01:00:00 --gres=gpu:4 --partition=gpu-cascade --qos=gpu --account=${ACCOUNT_ID} --pty /usr/bin/bash --login
```

### CPU Session
```sh
$ ACCOUNT_ID=YOUR_PROJECT_ID
srun --exclusive --nodes=1 --time=01:00:00 --partition=standard --qos=standard --account=${ACCOUNT_ID} --pty /usr/bin/bash --login
```

# Morpheus Keyword Listing
## Enable Options
Options can be enabled by specifying `-DMorpheus_ENABLE_X`.
* Morpheus_ENABLE_DEBUG
    * Whether to activate extra debug features - may increase compile times
    * BOOL Default: OFF
* Morpheus_ENABLE_EXAMPLES
    * Whether to enable building examples
    * BOOL Default: OFF
* Morpheus_ENABLE_TESTS
    * Whether to enable building tests
    * BOOL Default: OFF
* Morpheus_ENABLE_BENCHMARKS
    * Whether to enable building benchmarks
    * BOOL Default: OFF
* Morpheus_INSTALL_TESTING
    * Whether to build tests and examples against installation
    * BOOL Default: OFF
  
## Third-party Libraries (TPLs)
The following options control enabling TPLs:
* Morpheus_ENABLE_TPL_CUBLAS: BOOL
    * Whether to enable CUBLAS
    * Default: ON if CUDA-enabled Kokkos, otherwise OFF
* Morpheus_ENABLE_TPL_HIPBLAS: BOOL
    * Whether to enable HIPBLAS
    * Default: ON if HIP-enabled Kokkos, otherwise OFF
* Morpheus_ENABLE_TPL_MPARK_VARIANT
    * Whether to enable the Mpark Variant library
    * BOOL Default: OFF

The following options control `find_package` paths for CMake-based TPLs:
* Morpheus_CUBLAS_ROOT: PATH
    * Location of CUBLAS install root.
    * Default: None or the value of the environment variable CUBLAS_ROOT if set
* GTest_ROOT: PATH
    * Location of GoogleTest install root.
    * Default: None or the value of the environment variable GTest_ROOT if set
* CUBLAS_LIBRARIES: STRING
    * Optional override for the libraries that comprise TPL CUBLAS.
    * Default: None. Default common library names will be searched
* CUBLAS_LIBRARY_DIRS: STRING
    * Optional override for the library directories that comprise TPL CUBLAS.
    * Default: None. Default common library locations will be searched
* MPARK_VARIANT_DIR or MPARK_VARIANT_ROOT
    * Location of MPARK_VARIANT prefix (ROOT) or CMake config file (DIR)
    * PATH Default:
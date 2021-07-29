# Installing and Using Morpheus

You can either use Morpheus as an installed package (encouraged) or use Morpheus in-tree in your project. Once Morpheus is installed In your `CMakeLists.txt` simply use:

```
find_package(Morpheus REQUIRED)
```

Then for every executable or library in your project:

```
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
 -DMorpheus_ENABLE_TPL_BLAS=ON
```
which, e.g. activates the BLAS dependency. The full keyword listing is below.

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
$ CXX_COMPILER=/lustre/sw/intel/compilers_and_libraries_2020.4.304/linux/bin/intel64/icpc
$ KOKKOS_INSTALL_DIR=/install/path/of/kokkos/with/intel
$ MORPHEUS_INSTALL_DIR=/install/path/of/morpheus/with/intel
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

### Installing Kokkos
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR} \
           -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON  -DKokkos_ENABLE_SERIAL=ON \
           -DKokkos_CXX_STANDARD=17 -DKokkos_ENABLE_COMPILER_WARNINGS=On -DKokkos_ARCH_VOLTA70=On -DKokkos_ARCH_SKX=On \
           -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=On
$ make
$ make install
```
**Warning** For cmake to find the cuda drivers you need to have a visible NVIDIA GPU. See [here](#gpu_interactive) how to create an interactive GPU session on Cirrus.

**Warning** For C++17 support requires NVCC 11.0+ or Clang 4.0+

## Installing Morpheus
```sh
$ cmake .. -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=${MORPHEUS_INSTALL_DIR} \
           -DKokkos_ROOT=${KOKKOS_INSTALL_DIR} -DCMAKE_BUILD_TYPE=Release \
           -DMorpheus_ENABLE_EXAMPLES=On -DMorpheus_ENABLE_TESTS=On
$ make
$ make install
```

**Warning** The installed Kokkos configuration does not support CXX extensions when build with `CUDA` enabled. Add the `-DCMAKE_CXX_EXTENSIONS=Off` to surpress warning.

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
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
#!/bin/sh

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -p|--prefix)
        # (/Volumes/PhD/Code/Libraries/install/morpheus)
            IPREFIX="$2"
            shift # past argument
            shift # past value
            ;;
        -k|--kokkos)
            KROOT="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            printf "Invalid arguments..\nExiting..\n"
            return
    esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters
cd ..

cmake -E make_directory "build"
CC=gcc CXX=g++ cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=/home/e609/e609/cstyl/libs/kokkos -DMorpheus_BUILD_EXAMPLES=Off -DMorpheus_BUILD_BENCHMARKS=On ..
# cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${IPREFIX} -DKokkos_ROOT=${KROOT} -DMorpheus_BUILD_EXAMPLES=Off -DMorpheus_BUILD_BENCHMARKS=On ..
cmake --build "build" --config Release --target install 
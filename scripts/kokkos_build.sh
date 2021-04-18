#!/bin/sh

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -p|--prefix)
        # (/Volumes/PhD/Code/Libraries/install/kokkos)
            IPREFIX="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            printf "Invalid arguments..\nExiting..\n"
            return
    esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters


git clone https://github.com/kokkos/kokkos.git
cd kokkos

cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${IPREFIX} -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_CXX_STANDARD=17 ..
cmake --build "build" --config Release --target install 
#!/bin/sh

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -p|--prefix)
        # (/Volumes/PhD/Code/Libraries/install/benchmark)
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

git clone https://github.com/google/benchmark.git
# Benchmark requires Google Test as a dependency. Add the source tree as a subdirectory.
git clone https://github.com/google/googletest.git benchmark/googletest
# Go to the library root directory
cd benchmark

cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${IPREFIX} ..
cmake --build "build" --config Release --target install 

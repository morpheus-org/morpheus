#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

BUILD_PATH="$SCRIPT_PATH/../build"
VERSIONS=("cusp" "dynamic_1" "dynamic_6" "dynamic_12" "dynamic_20")
CCOMP="gcc"
CPPCOMP="g++"

module load cmake/3.16.0
module unload gcc
module load gcc/7.3.0

# Create build directory and compile code to generate the binaries
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"
CC="$CCOMP" CXX="$CPPCOMP" cmake "$BUILD_PATH/.."
make

# Print information about binaries
for version in "${VERSIONS[@]}"
do
  BINARY="$BUILD_PATH/$version"
  ls -l "$BINARY"
done

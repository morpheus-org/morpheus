#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

MACHINE="$1"

BUILD_PATH="$SCRIPT_PATH/../build"
VERSIONS=("cusp" "dynamic_1" "dynamic_6" "dynamic_12" "dynamic_20")
CCOMP="gcc"
CPPCOMP="g++"

if [ "$MACHINE" = "local" ]; then
  echo "Running locally"
  BUILD_FLAGS="-DLOCAL=ON"
elif [ "$MACHINE" = "archer" ]; then
  echo "Running on ARCHER"
  module load cmake/3.16.0
  module unload gcc
  module load gcc/7.3.0
  BUILD_FLAGS="-DARCHER=ON"
elif [ "$MACHINE" = "cirrus" ]; then
  echo "Running on CIRRUS"
  module load gcc/8.2.0
  module load cmake/3.17.3
  BUILD_FLAGS="-DCIRRUS=ON"
else
  echo "Invalid inpug argument."
  echo "Usage:"
  echo -e "\t/path/to/script/build.sh [local|archer|cirrus]"
  exit -1
fi

# Create build directory and compile code to generate the binaries
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"
CC="$CCOMP" CXX="$CPPCOMP" cmake "$BUILD_FLAGS" "$BUILD_PATH/.."
make

# Print information about binaries
for version in "${VERSIONS[@]}"
do
  BINARY="$BUILD_PATH/$version"
  ls -l "$BINARY"
done
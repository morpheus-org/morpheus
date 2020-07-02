#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

MACHINE="$1"
COMPILER="$2"
COMP_VERSION="$3"

if [ "$COMPILER" == "" ]; then
  COMPILER="gcc"
fi

if [ "$COMP_VERSION" == "" ]; then
  COMP_VERSION="6.3.0"
fi

if [ "$MACHINE" = "local" ]; then
  echo "Running locally"
  BUILD_FLAGS="-DLOCAL=ON"
elif [ "$MACHINE" = "archer" ]; then
  echo "Running on ARCHER"
  module load cmake/3.16.0
  module unload gcc
  module load "$COMPILER/$COMP_VERSION"
  BUILD_FLAGS="-DARCHER=ON"
elif [ "$MACHINE" = "cirrus" ]; then
  echo "Running on CIRRUS"
  module load "$COMPILER/$COMP_VERSION"
  module load cmake/3.17.3
  BUILD_FLAGS="-DCIRRUS=ON"
else
  echo "Invalid input argument."
  echo "Usage:"
  echo -e "\t/path/to/script/run.sh [local|archer|cirrus] [compiler] [version]\n\n"
  echo -e "\t\t local|archer|cirrus: Select for which machine the code is build for."
  echo -e "\t\t compiler: Compiler used (default is gcc)."
  echo -e "\t\t version: Version of the compiler (default is 6.3.0)."
  exit -1
fi

echo "Parameters:"
echo -e "\t MACHINE = $MACHINE"
echo -e "\t COMPILER = $COMPILER"
echo -e "\t VERSION = $COMP_VERSION"

BUILD_PATH="$SCRIPT_PATH/../build/$COMPILER/$COMP_VERSION"
VERSIONS=("cusp" "dynamic_1" "dynamic_6" "dynamic_12" "dynamic_20")
CCOMP="gcc"
CPPCOMP="g++"

# Create build directory and compile code to generate the binaries
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"
CC="$CCOMP" CXX="$CPPCOMP" cmake "$BUILD_FLAGS" "$BUILD_PATH/../../.."
make

# Print information about binaries
for version in "${VERSIONS[@]}"
do
  BINARY="$BUILD_PATH/$version"
  ls -l "$BINARY"
done

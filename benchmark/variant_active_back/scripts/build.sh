#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

. $SCRIPT_PATH/../../../scripts/bash/machine.sh
. $SCRIPT_PATH/../../../scripts/bash/cmake.sh
. $SCRIPT_PATH/../../../scripts/bash/compilers.sh

MACHINE=$(echo $1 | tr A-Z a-z)
COMPILER=$(echo $2 | tr A-Z a-z)
COMP_VERSION="$3"

if $(check_supported_machines $MACHINE); then
  echo "Running on $MACHINE"
else
  echo "Invalid input argument."
  echo "Usage:"
  echo -e "\t/path/to/script/build.sh [local|archer|cirrus] [compiler] [version]\n\n"
  echo -e "\t\t local|archer|cirrus: Select for which machine the code is build for."
  echo -e "\t\t compiler: Compiler used (default is gcc)."
  echo -e "\t\t version: Version of the compiler (default is 6.3.0)."
  exit -1
fi

load_cmake $MACHINE
load_compiler $MACHINE $COMPILER $COMP_VERSION

BUILD_PATH="$SCRIPT_PATH/../build/$COMPILER/$COMP_VERSION"
BUILD_FLAGS="-D$(echo $MACHINE | tr a-z A-Z)=ON"
VERSIONS=("static" "dynamic_01" "dynamic_06" "dynamic_12" "dynamic_20")

# Create build directory and compile code to generate the binaries
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"
CC=$(set_CC $COMPILER) CXX=$(set_CXX $COMPILER) cmake "$BUILD_FLAGS" "$BUILD_PATH/../../.."
make

# Print information about binaries
for version in "${VERSIONS[@]}"
do
  BINARY="$BUILD_PATH/$version"
  ls -l "$BINARY"
done

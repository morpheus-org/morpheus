#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MORPHEUS_PATH="$SCRIPT_PATH/../../.."

. $MORPHEUS_PATH/scripts/bash/machine.sh
. $MORPHEUS_PATH/scripts/bash/cmake.sh
. $MORPHEUS_PATH/scripts/bash/compilers.sh

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

load_cmake $MORPHEUS_PATH $MACHINE
load_compiler $MORPHEUS_PATH $MACHINE $COMPILER $COMP_VERSION

BUILD_PATH="$SCRIPT_PATH/../build/benchmark/$COMPILER/$COMP_VERSION"
BUILD_FLAGS="-D$(echo $MACHINE | tr a-z A-Z)=ON"

VERSIONS=("static" "dynamic_01" "dynamic_06" "dynamic_12" "dynamic_20"
          "dynamic_01_boost" "dynamic_06_boost" "dynamic_12_boost" "dynamic_20_boost"
          "static_O2" "dynamic_01_O2" "dynamic_06_O2" "dynamic_12_O2" "dynamic_20_O2"
          "dynamic_01_boost_O2" "dynamic_06_boost_O2" "dynamic_12_boost_O2" "dynamic_20_boost_O2")

# Create build directory and compile code to generate the binaries
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"

RESULTS_FILE="$SCRIPT_PATH/../results/processed_data/build/$MACHINE"_"$COMPILER"_"$COMP_VERSION.csv"

mkdir -p $(dirname "$RESULTS_FILE")

CC=$(set_CC $COMPILER) CXX=$(set_CXX $COMPILER) cmake "$BUILD_FLAGS" "$BUILD_PATH/../../../.."

# CSV Header
echo "Machine,Version,Build Time,Binary Size KB" 2>&1 | tee "$RESULTS_FILE"

TIMEFORMAT=(%U + %S)
for version in "${VERSIONS[@]}"
do
    exec 3>&1 4>&2
    build_time=$( { time make $version 1>&3 2>&4; } 2>&1 )  # Captures time only.
    exec 3>&- 4>&-
    bin_sz=$(ls -l "$BUILD_PATH/$version" | awk '{ sz_kb = $5 / 1024; print sz_kb }')

    echo "$MACHINE,$version,$build_time,$bin_sz" 2>&1 | tee -a "$RESULTS_FILE"

done

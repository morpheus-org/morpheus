#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

INSTALL_PATH="$SCRIPT_PATH/../../.."

THRUST="https://github.com/thrust/thrust/archive/1.8.1.tar.gz"

cd "$INSTALL_PATH"
git clone https://github.com/thrust/thrust.git

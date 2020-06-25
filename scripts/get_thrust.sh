#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

INSTALL_PATH="$SCRIPT_PATH/../.."

THRUST="https://github.com/thrust/thrust/archive/1.8.1.tar.gz"

wget -P "$INSTALL_PATH" "$THRUST"
tar -xvf "$INSTALL_PATH/1.8.1.tar.gz"
rm "$INSTALL_PATH/1.8.1.tar.gz"
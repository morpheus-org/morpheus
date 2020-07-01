#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

INSTALL_PATH="$SCRIPT_PATH/../../.."

BOOST="https://dl.bintray.com/boostorg/release/1.73.0/source/boost_1_73_0.tar.gz"

wget -P "$INSTALL_PATH" "$BOOST"
cd "$INSTALL_PATH"
tar -xvf "$INSTALL_PATH/boost_1_73_0.tar.gz"
rm "$INSTALL_PATH/boost_1_73_0.tar.gz"

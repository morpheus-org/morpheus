#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

INSTALL_PATH="$SCRIPT_PATH/../.."

CUSP="https://github.com/cusplibrary/cusplibrary/archive/v0.5.1.zip"

wget -P "$INSTALL_PATH" "$CUSP"
gunzip "$INSTALL_PATH/v0.5.1.zip"
rm "$INSTALL_PATH/v0.5.1.zip"
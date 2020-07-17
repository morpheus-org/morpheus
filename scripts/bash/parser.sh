#!/bin/bash

parse_arg()
{
    local __PARSED_ARG="$1"
    local __DEFAULT_ARG="$2"

    if [ "$__PARSED_ARG" == "" ]; then
        __PARSED_ARG=$__DEFAULT_ARG
    fi

    echo "$__PARSED_ARG"
}
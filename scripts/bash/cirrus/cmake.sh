#!/bin/bash

load_cmake_cirrus()
{
    module load cmake/3.17.3
    module unload gcc
}
#!/bin/bash

rm -R build
rm -R lib
rm -R bin
mkdir build
cd build
cmake ../
make -j4

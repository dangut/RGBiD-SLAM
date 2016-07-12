#!/bin/bash

for i in $(ls -d */)

do 
  cd $i
  ../build_dir.sh
  cd ..

 done

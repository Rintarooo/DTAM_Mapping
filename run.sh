#!/bin/bash

DIR="build"
if [ -d "$DIR" ]; then
  echo "rm -rf ${DIR}"
  rm -rf ${DIR}
fi

mkdir ${DIR}
cd ${DIR}
cmake ..
# cmake CMAKE_BUILD_TYPE=DEBUG ..
# cmake -DCMAKE_BUILD_TYPE=DEBUG ..

cmake --build .
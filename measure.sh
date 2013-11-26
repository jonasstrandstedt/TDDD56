#!/bin/bash

clang++ -O0 -DNDEBUG main.cpp -std=c++11

#blur
CMD="./a.out"
echo ${CMD}
#for i in {1..10}; do ${CMD} 2>/dev/null; echo -n "+"; done;
(echo -n "("; for i in {1..5}; do ${CMD} ; echo -n "+"; done; echo "0)/5.0") | bc -l
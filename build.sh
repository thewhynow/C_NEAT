#!/bin/bash

flags=""
lib_path=""
run_cmd=""
  
if [[ "$(uname)" == "Darwin" ]]; then
       lib_path="/opt/homebrew" 
else
       lib_path="/"
fi

if [[ "$1" == "debug" ]]; then
       run_cmd="lldb ./main"
       flags+="-g "
else
       flags+="-O3 "
       run_cmd="./main"
fi

gcc main.c NEAT.c C_Vector/C_Vector.c \
       -I$lib_path/include           \
       -L$lib_path/lib               \
       -lraylib                      \
       -o main                       \
       $flags

$run_cmd

rm main

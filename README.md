personal cuda playground:

```bash
$ nvcc -o kernelProfile kernelProfile.cu -std=c++20
nsys profile --stats=true ./kernelProfile
# then open the .nsys-rep file with nvidia nsight compute
```

```bash
# replace the architecture and toolkit path with the
#                   appropriate values for your system
clang++ -o kernelProfile kernelProfile.cu --offload-arch=sm_75 \
    -L/usr/local/cuda-13.4/lib64/ -lcudart_static -ldl -lrt -pthread
# for more details, see: https://llvm.org/docs/CompileCudaWithLLVM.html
```



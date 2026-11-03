personal cuda playground:

some commands:

```bash
$ nvcc -o kernelProfile kernelProfile.cu -std=c++20
nsys profile --stats=true ./kernelProfile
# open the .nsys-rep file with nvidia nsight compute
```



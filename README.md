# Laboratory work for University

## CUDA basics

In this laboratory work we have to learn **CUDA** basics, create one simple algorithm and test
the computational speed, by launching our algorithm on **CPU** and **GPU**.

### Compilation

Here is nvcc command that we used to compile and run our program on **GPU**

```console
nvcc main.cu -rdc=true -lcurand && ./a.out
```

**where:**

- `-rdc` - parameter that allow to use relocatable device code, to call kernels **inside** each other.
- `-lcurand` - links `cuRAND` library that is used to generate matrixes of any sizes.

>STATUS: Reports are done, waiting for an acceptance.

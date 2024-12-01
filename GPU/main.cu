#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 16

__global__ void add_to_row(float* dev_matrix, int row, int diag, float k, int size) {
    int col = diag + (blockIdx.x * blockDim.x + threadIdx.x);

	if (col < size) {
		dev_matrix[row*size+col] += k * dev_matrix[diag*size+col];
	}
}

void triangular_matrix(float* hst_matrix, float* dev_matrix, int size) {
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

	for (int diag = 0; diag < size-1; diag++) {
		for (int row = diag+1; row < size; row++) {
			cudaMemcpy(hst_matrix, dev_matrix, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

			float k = (- 1.0) * hst_matrix[row*size+diag] / hst_matrix[diag*size+diag];
			add_to_row<<<blocksPerGrid, threadsPerBlock>>>(dev_matrix, row, diag, k, size);
		}
	}
}

void show_matrix(float* hst_matrix, int size) {
    for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%7.2f", hst_matrix[i*size+j]);
		}
		printf("\n");
	}
	printf("\n");
}

void fill_matrix(float* dev_matrix, int size) {
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock());
    curandGenerateUniform(gen, dev_matrix, (size*size));

    curandDestroyGenerator(gen);
}

int main(void) {	
    int size;

	printf("Enter size of square matrix, which will be transformed into a triangular >>> ");
	fflush(stdin);
	fscanf(stdin, "%d", &size);

	float *hst_matrix;
	float *dev_matrix;

	clock_t start = clock(); //! START

	printf("allocation...");
    hst_matrix = (float*)calloc(size * size, sizeof(float));
    cudaMalloc((void **)&dev_matrix, sizeof(float) * size * size);
	printf(" - allocated.\n");

	printf("filling...");
    fill_matrix(dev_matrix, size);
	printf(" - filled.\n");

	if (size <= 20) {
		cudaMemcpy(hst_matrix, dev_matrix, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
		printf("Generated matrix:\n");
		show_matrix(hst_matrix, size);
	}

	printf("transforming...");
	triangular_matrix(hst_matrix, dev_matrix, size);
	printf(" - transformed\n");

    if (size <= 20) {
		cudaMemcpy(hst_matrix, dev_matrix, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
		printf("Transformed matrix:\n");
        show_matrix(hst_matrix, size);
    }

	clock_t end = clock(); //! STOP

	double ms_duration = (double)(end - start) / CLOCKS_PER_SEC * 1000;
	printf("Time to execute - %f ms\n", ms_duration);

	free(hst_matrix);
    cudaFree(dev_matrix);

	return 0;
}
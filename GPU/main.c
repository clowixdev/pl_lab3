#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>

#define RANGE 1000
#define GRID_SIZE 1
#define BLOCK_SIZE 16
#define DEPTH 4

/*
	for (int diag =0; diag<size-1;diag++)
	{ 
		for (int str = diag+1; str < size; str++)
		{
			float k = (- 1.0) * matrix[str][diag] / matrix[diag][diag];
			for (int column = diag; column < size; column++)
			{
				matrix[str][column] += k * matrix[diag][column] ;
			}
		}
	}
*/

// __global__ triangular_matrix(int* matrix, int size) {

// }

void show_matrix(float* hst_matrix, int size) {
    for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%7.2f", hst_matrix[i*size+j]);
		}
		printf("\n");
	}
}

void fill_matrix(float* dev_matrix, int size) {
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 0);
    curandGenerateUniform(gen, dev_matrix, size * size);

    curandDestroyGenerator(gen);
}

int main(void) {	
    int size;

	printf("Enter size of square matrix, which will be transformed into a triangular >>> ");
	fflush(stdin);
	fscanf(stdin, "%d", &size);

	float *hst_output;
	float *dev_input;
	float *dev_output;

	clock_t start = clock(); //! START

    hst_output = (float*)calloc(size * size, sizeof(float));
    cudaMalloc((void **)&dev_input, sizeof(float) * size * size);
    cudaMalloc((void **)&dev_output, sizeof(float) * size * size);

    fill_matrix(dev_input, size);

    // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	// triangular_matrix<<<dimGrid, dimBlock>>>(dev_input, size);

    cudaMemcpy(hst_output, dev_output, size * size, cudaMemcpyDeviceToHost);

	clock_t end = clock(); //! STOP

    if (size <= 20) {
        show_matrix(hst_output, size);
    }

	double ms_duration = (double)(end - start) / CLOCKS_PER_SEC * 1000;
	printf("Time to execute - %f ms\n", ms_duration);

	free(hst_output);
    cudaFree(dev_input);
    cudaFree(dev_output);

	return 0;
}
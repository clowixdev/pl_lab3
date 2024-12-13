#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdbool.h>

#define BLOCK_SIZE 16
#define FILE_INPUT

__global__ void add_to_row(float* dev_matrix, int row, int diag, float k, int size) {
	int col = diag + (blockIdx.x * blockDim.x + threadIdx.x);

	if (col < size) {
		dev_matrix[row * size + col] += k * dev_matrix[diag * size + col];
	}
}

__global__ void compute_k(float* dev_matrix, int diag, int size) {
	dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
	dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
	int row = diag + 1 + (blockIdx.x * blockDim.x + threadIdx.x);

	if (row < size) {
		float k = (-1.0) * dev_matrix[row * size + diag] / dev_matrix[diag * size + diag];
		__syncthreads();
		add_to_row << <blocksPerGrid, threadsPerBlock >> > (dev_matrix, row, diag, k, size);	
	}
}

void triangular_matrix(float* dev_matrix, int size) {
	for (int diag = 0; diag < size - 1; diag++) {

		int rows = size - (diag + 1);
		dim3 threads( BLOCK_SIZE  * BLOCK_SIZE);
		dim3 blocks((rows + threads.x - 1) / threads.x);

		compute_k << <blocks, threads >> > (dev_matrix, diag, size); 
		//cudaDeviceSynchronize();
	}
}


void show_matrix(float* hst_matrix, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%8.2f", hst_matrix[i * size + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void fill_matrix(float* dev_matrix, int size) {
	curandGenerator_t gen;

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock());
	curandGenerateUniform(gen, dev_matrix, (size * size));

	curandDestroyGenerator(gen);
}

void print_matrix(float* matrix, int size)
{
	FILE* file;
	file = fopen("../../gpu_output.txt", "w");
	
	if (file == NULL)
	{
		printf("Failed to open file\n");
		return;
	}

	fprintf(file, "Transformed matrix:\n");
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			fprintf(file, "%14.2f ", matrix[i * size + j]);
			if (j == 1024) fprintf(file," | ");// just for clarity
		}
		fprintf(file, "\n");
	}

	fclose(file);
}

void file_fill_matrix(float* hst_matrix, int  size, FILE* file)
{
	int ch;
	while ((ch = fgetc(file)) != '\n' && ch != EOF);

	printf("reading matrix from file...");

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			int result = fscanf(file, "%f", &hst_matrix[i * size + j]);
			if (result != 1) {
				fprintf(stderr, "Error reading matrix data at position (%d, %d)\n", i, j);
				exit(EXIT_FAILURE);
			}
		}
	}
}


int main(void) {
	int size;

	printf("Enter size of square matrix, which will be transformed into a triangular >>> ");
	fflush(stdin);
	fscanf(stdin, "%d", &size);

	float* hst_matrix;
	float* dev_matrix;


	printf("allocation...");
	hst_matrix = (float*)calloc(size * size, sizeof(float));
	cudaMalloc((void**)&dev_matrix, sizeof(float) * size * size);
	printf(" - allocated.\n");


	printf("filling...");
#ifdef FILE_INPUT
	FILE* file_input = fopen("../../../CPU/generated_matrix.txt", "r");
	if (file_input == NULL) {
		fprintf(stderr, "Failed to open file 'generated_matrix.txt'\n");
		free(hst_matrix);
		cudaFree(dev_matrix);
		return -1;
	}
	printf("filling matrix from file...");
	file_fill_matrix(hst_matrix, size, file_input);
	fclose(file_input);

	cudaMemcpy(dev_matrix, hst_matrix, sizeof(float) * size * size, cudaMemcpyHostToDevice);
#else
	fill_matrix(dev_matrix, size);
#endif
	printf(" - filled.\n");
	

	if (size <= 20) {
		cudaMemcpy(hst_matrix, dev_matrix, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
		printf("Generated matrix:\n");
		show_matrix(hst_matrix, size);
	}

	printf("transforming...");
	clock_t start = clock(); //! START
	triangular_matrix(dev_matrix, size);
	clock_t end = clock(); //! STOP
	printf(" - transformed\n");

	cudaMemcpy(hst_matrix, dev_matrix, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
	
	if (size <= 20) {
		printf("Transformed matrix:\n");
		show_matrix(hst_matrix, size);
	}

	print_matrix(hst_matrix, size);


	double ms_duration = (double)(end - start) / CLOCKS_PER_SEC * 1000;
	printf("Time to execute - %f ms\n", ms_duration);

	free(hst_matrix);
	cudaFree(dev_matrix);

	return 0;
}
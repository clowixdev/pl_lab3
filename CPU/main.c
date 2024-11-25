
/*	This program creates a square matrix of size AxA, which is filled with random values.
*	Then the matrix will be transformed to a triangular matrix using the Gauss method.	*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define RANGE 1000		/* [-RANGE;RANGE] for generating random values*/
#define _CRT_SECURE_NO_WARNINGS
#define FILE_INPUT


float** matrix_create(int size);
void matrix_fill(float** matrix, int size);
void matrix_clean_up(float** matrix, int size);
void print_matrix(float** matrix, int size, bool flag);
void triangular_matrix(float** matrix, int size);


float** matrix_create(int size)
{
	float **matrix = (float **)malloc(size * sizeof(float *));
	if (matrix == NULL)
	{
		fprintf(stderr, "Memory allocation failed for matrix rows.\n");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < size; i++)
	{
		matrix[i] = (float*)malloc(size * sizeof(float));
		if (matrix[i] == NULL)
		{
			fprintf(stderr, "Memory allocation failed for matrix columns.\n");
			exit(EXIT_FAILURE);
		}
	} 
	return matrix;
}

void matrix_fill(float** matrix, int size)
{
	srand(time(NULL));

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			float value = (float)(rand() % (RANGE*2+1) - RANGE);
			matrix[i][j] = value;
		}
	}
}

void matrix_clean_up(float** matrix, int size)
{
	for (int i = 0; i < size; i++) {
		free(matrix[i]);
	}
	free(matrix);
}

void print_matrix(float** matrix, int size, bool flag)
{
	FILE* file;
	if (flag == true)
	{
		file = fopen("output.txt", "w");
		fprintf(file, "Generated matrix:\n");
	}
	else
	{
		file = fopen("output.txt", "a");
		fprintf(file, "Transformed matrix:\n");
	}
	
	if (file == NULL)
	{
		printf("Failed to open file\n");
		return;
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			fprintf(file, "%8.2f", matrix[i][j]);
		}
		fprintf(file, "\n");
	}

	fclose(file);
}

void triangular_matrix(float** matrix, int size)
{
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
}

int main(void)
{

	int size;
	float** matrix;
#ifdef FILE_INPUT
	FILE* input_file = fopen("input.txt", "r");
	if (input_file == NULL)
	{
		printf("Failed to open file\n");
		return;
	}
	fscanf(input_file, "%d", &size);
#else
	printf("Enter size of square matrix, which will be transformed into a triangular >>>");
	fflush(stdin);
	fscanf(stdin, "%d", &size);
#endif // FILE_INPUT


	matrix = matrix_create(size);
	matrix_fill(matrix, size);
	
	print_matrix(matrix, size, 1);

	triangular_matrix(matrix, size);

	print_matrix(matrix, size, 0);
	matrix_clean_up(matrix, size);
	return 0;
}
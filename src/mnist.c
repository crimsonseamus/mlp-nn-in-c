#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include "../include/mnist.h"

static uint32_t read_be_u32 (FILE* file) {
	uint8_t b[4];

	if(fread(b, 1, 4, file) != 4){
		printf("Error reading buffer\n");
		exit(1);
	}

	uint32_t curr_chunk = (uint32_t)b[0] << 24 
			| (uint32_t)b[1] << 16 
			| (uint32_t)b[2] << 8 
			| (uint32_t)b[3];

	return curr_chunk;
}

float* mnist_load_images (
	const char* file_path,
	int* out_count,
	int* out_rows,
	int* out_cols
) {
	FILE* f = fopen(file_path, "rb");
	
	if(!f) {
		printf("Error opening file!\n");
		exit(1);
	}

	uint32_t magic = read_be_u32(f);
	if(magic != 2051) {
		printf("not a valid magic number %u\n", magic);
		exit(1);
	}
	
	uint32_t count = read_be_u32(f);
	uint32_t rows = read_be_u32(f);
	uint32_t cols = read_be_u32(f);

	printf("Mnist images: \n");
	printf(" count: %u\n", count);
	printf(" rows: %u\n", rows);
	printf(" cols: %u\n", cols);

	size_t image_size = rows * cols;
	size_t total_pixels = count * image_size;


	uint8_t* raw = malloc(total_pixels * sizeof(uint8_t));

	if(!raw) {
		printf("Memory allocation failed for raw\n");
		exit(1);
	}
	
	if(fread(raw, 1, total_pixels, f) != total_pixels) {
		printf("Error reading pixels for raw\n");
		exit(1);
	}
	
	fclose(f);

	float* images = malloc(sizeof(float) * total_pixels);

	if(!images) {
		printf("Error allocating memory for images\n");
		exit(1);
	}

	for(size_t i = 0; i < total_pixels; i++) {
		images[i] = raw[i] / 255.0f;
	}

	free(raw);

	*out_count = count;
	*out_rows = rows;
	*out_cols = cols;
	
	return images;
}

uint8_t* mnist_load_labels (
	const char* file_path,
	int* out_count
) {
	FILE* fl = fopen(file_path, "rb");

	if(!fl) {
		printf("Error opening label file!\n");
		exit(1);
	}

	uint32_t magic = read_be_u32(fl);

	if(magic != 2049) {
		printf("Wrong magic number for label!%u\n", magic);
		exit(1);
	}

	uint32_t count = read_be_u32(fl);

	uint8_t* labels = malloc(count * sizeof(uint8_t));
	
	printf("Labels count: %u\n", count);


	if(!labels) {
		printf("Error allocating memory for lables\n");
		exit(1);
	}
	
	if(fread(labels, 1, count, fl) != count) {
		printf("Erorr reading lables maybe wrong file!\n");
		exit(1);
	}
	
	fclose(fl);

	*out_count = (int)count;

	return labels;
}

/* SPDX-License-Identifier: MIT
 * Copyright (c) 2021 Ravi Kumar
 */

#include <iostream>
#include <random>
#include <time.h>
#include <string.h>
#include "utils.h"

#define FILTER_KERNEL_SOURCE		"filter.cl"
#define FILTER_1_KERNEL				"filter_1"
#define FILTER_2_KERNEL				"filter_2"
#define LOCAL_XRES					256
#define UPSCAL_FACTOR				2
#define SAMPLE_SIZE					2560
#define TAP_SIZE_1					335
#define TAP_SIZE_2					29
#define BATCH_KERNELS				2
#define WARMUP_RUNS					100

// Helper function to generate test data for this exercise
void GenerateTestData(size_t const numElements,
				std::vector<float> const& filterWeights_1,
				std::vector<float> const& filterWeights_2,
				float **testData_ret,
				std::vector<float> reference, bool verify)
{
	int filterLength = filterWeights_1.size();
	int filterLength_2 = filterWeights_2.size();
	long const halfFilterLength = filterLength / 2;
	size_t paddedLength = numElements + filterLength - 1;

    //testData.resize(2 * paddedLength); //Contrains both real and imaginary parts
	float *testData = new float[2 * paddedLength];
	*testData_ret = testData;
	
	std::vector<float> tmp_reference_r;
	std::vector<float> tmp_reference_i;

	tmp_reference_r.resize(UPSCAL_FACTOR*(numElements + filterLength_2 - 1));
	tmp_reference_i.resize(UPSCAL_FACTOR*(numElements + filterLength_2 - 1));

	reference.resize(2 * UPSCAL_FACTOR * numElements);
	//reference = new float[2 * UPSCAL_FACTOR * numElements];
		
    std::default_random_engine            generator;
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    for(size_t i = halfFilterLength; i < numElements + halfFilterLength; i++)
    {
		testData[2*i+0] = uniform(generator);
		testData[2*i+1] = uniform(generator);
	}

	if (verify) {
		int filterLength = filterWeights_1.size();
		
		for (size_t i = 0; i < numElements; i++)
		{
			long windowIdx = i;// -halfFilterLength;
			float  value_r = 0.0f;
			float  value_i = 0.0f;
			float sample_r = 0.0f;
			float sample_i = 0.0f;

			for (int filterIdx = 0; filterIdx < filterLength; filterIdx++, windowIdx++)
			{
				sample_r = testData[2 * windowIdx + 0];
				value_r += sample_r * filterWeights_1[filterIdx];

				sample_i = testData[2 * windowIdx + 1];
				value_i += sample_i * filterWeights_1[filterIdx];
			}

			tmp_reference_r[2 * i + 0] = value_r;
			tmp_reference_r[2 * i + 1] = value_r;

			tmp_reference_i[2 * i + 0] = value_i;
			tmp_reference_i[2 * i + 1] = value_i;
		}

		filterLength = filterWeights_2.size();
		
		for (size_t i = 0; i < 2 * numElements; i++)
		{
			long windowIdx = i;// -halfFilterLength;
			float  value_r = 0.0f;
			float  value_i = 0.0f;
			float sample_r = 0.0f;
			float sample_i = 0.0f;

			for (int filterIdx = 0; filterIdx < filterLength; filterIdx++, windowIdx++)
			{
				sample_r =  tmp_reference_r[windowIdx];
				value_r += sample_r * filterWeights_2[filterIdx];

				sample_i = tmp_reference_i[windowIdx];
				value_i += sample_i * filterWeights_2[filterIdx];
			}

			reference[2*i+0] = value_r;
			reference[2*i+1] = value_i;			
		}

	}
}

bool AlmostEqual(float ref, float value, size_t ulp)
{
    return std::abs(ref-value) <= std::numeric_limits<float>::epsilon() * std::abs(ref) * ulp;
}

// Helper function to compare test data for this exercise
void CompareData(std::vector<float> const& expected, 
				float *actual, int toleranceInUlp)
{
    size_t const numElements = expected.size();	//will match both real and imaginary data

    bool pass = true;
    for(size_t i = 0; i < numElements; i++)
    {
        if(!AlmostEqual(expected[i], actual[i], toleranceInUlp))
        {
            std::cout << "Mismatch at index " << i << "!\nExpected value: " << expected[i]
                      << "\nActual value: " << actual[i] << std::endl;
            pass = false;
            break;
        }

    }

    if(pass)
        printf("Test Passed! All values match.\n");
}

bool buildKernels(cl_context oclContext, cl_device_id oclDevice, cl_kernel *firFilterKernel_1, cl_kernel *firFilterKernel_2, cl_uint filterLength_1, cl_uint filterLength_2, int batch_kernels)
{
	cl_int err = CL_SUCCESS;

	/**************************************************************************
	* Read kernel source file into buffer.                                    *
	**************************************************************************/
	const char *filename = FILTER_KERNEL_SOURCE;
	char *source = NULL;
	size_t sourceSize = 0;
	err = convertToString(filename, &source, &sourceSize);
	CHECK_RESULT(err != CL_SUCCESS, "Error reading file %s ", filename);

	/**************************************************************************
	* Create kernel program.                                                  *
	**************************************************************************/
	cl_program programFirFilter = clCreateProgramWithSource(oclContext, 1, (const char **)&source, (const size_t *)&sourceSize, &err);
	CHECK_RESULT(err != CL_SUCCESS, "clCreateProgramWithSource failed with Error code = %d", err);

	/**************************************************************************
	* Build the kernel and check for errors. If errors are found, it will be  *
	* printed to console                                                      *
	**************************************************************************/
	char option[256];
	sprintf(option, "-DLOCAL_XRES=%d -DTAP_SIZE_1=%d -DTAP_SIZE_2=%d -cl-mad-enable", LOCAL_XRES, filterLength_1, filterLength_2);

	err = clBuildProgram(programFirFilter, 1, &(oclDevice), option, NULL, NULL);
	free(source);
	if (err != CL_SUCCESS)
	{
		char *buildLog = NULL;
		size_t buildLogSize = 0;
		clGetProgramBuildInfo(programFirFilter, oclDevice,
			CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog,
			&buildLogSize);
		buildLog = (char *)malloc(buildLogSize);
		clGetProgramBuildInfo(programFirFilter, oclDevice,
			CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
		printf("%s\n", buildLog);
		free(buildLog);
		clReleaseProgram(programFirFilter);
		CHECK_RESULT(true, "clCreateProgram failed with Error code = %d", err);
	}

	/**************************************************************************
	* Create kernel                                                           *
	**************************************************************************/

	for (int i = 0; i < batch_kernels; i++) {
		firFilterKernel_1[i] = clCreateKernel(programFirFilter, FILTER_1_KERNEL, &err);
		CHECK_RESULT(err != CL_SUCCESS, "clCreateKernel failed with Error code = %d", err);

		firFilterKernel_2[i] = clCreateKernel(programFirFilter, FILTER_2_KERNEL, &err);
		CHECK_RESULT(err != CL_SUCCESS, "clCreateKernel failed with Error code = %d", err);
	}

	clReleaseProgram(programFirFilter);
	return true;
}

void usage(const char *prog)
{
	printf("Usage: %s \n\t", prog);
	printf("\n\t[-zeroCopy (0 | 1)] //0 (default) - Device buffer, 1 - zero copy buffer\n\t");
	printf("[-numElements <int value> (default 245760)]\n\t");
	printf("[-f1 <int value> (filter1 TAP size - default 335)]\n\t");
	printf("[-f2 <int value> (filter2 TAP size - default 29)]\n\t");
	printf("[-iter <int value> (Number of times the piepeline shold be run)]\n\t");
	printf("[-verify (0 | 1]\n\t");
	printf("[-gpu (0 | 1) [in mGPU case, 0 = dGPU, 1 = iGPU]\n\t");
	printf("[-h (help)]\n\n");
	printf("Example: \n\n\t\tfirFilter.exe -numElements 256000 -f1 335 -f2 29 -verify 1 -iter 10\n\n");
}

int main(int argc, char **argv)
{
	DeviceInfo infoDeviceOcl;

	cl_mem device_input_cl[BATCH_KERNELS];
	cl_mem device_tmp_input_cl[BATCH_KERNELS];
	cl_mem device_output_cl[BATCH_KERNELS];

	cl_mem filter1_coeff_cl;
	cl_mem filter2_coeff;
	cl_int err;

	cl_kernel kernel_1[BATCH_KERNELS];
	cl_kernel kernel_2[BATCH_KERNELS];
	
	bool zeroCopy = false;
	bool verify = true;
	bool dataTransfer;
	int iteration = 10;
	int gpu = 0;

	size_t filterLength_1 = TAP_SIZE_1;
	size_t filterLength_2 = TAP_SIZE_2;
	size_t numElements = SAMPLE_SIZE;

	float *data;
	float *result;
	float *pinned_input[BATCH_KERNELS];

	cl_mem host_input_cl[BATCH_KERNELS];
	cl_mem host_output_cl[BATCH_KERNELS];

	std::vector<float> reference;
	float *pinned_output[BATCH_KERNELS];


	/***************************************************************************
	* Processing the command line arguments                                   *
	**************************************************************************/
	int tmpArgc = argc;
	char **tmpArgv = argv;

	while (tmpArgv[1] && tmpArgv[1][0] == '-')
	{
		if (strncmp(tmpArgv[1], "-f1", 3) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			filterLength_1 = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-f2", 3) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			filterLength_2 = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-zeroCopy", 9) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			zeroCopy = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-numElements", 10) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			numElements = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-verify", 7) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			verify = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-iter", 5) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			iteration = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-gpu", 4) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			gpu = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-h", 2) == 0)
		{
			usage(argv[0]);
			exit(1);
		}
		else
		{
			printf("Illegal option %s ignored\n", tmpArgv[1]);
			usage(argv[0]);
			exit(1);
		}
		tmpArgv++;
		tmpArgc--;
	}

	if (tmpArgc > 1)
	{
		usage(argv[0]);
		exit(1);
	}

	dataTransfer = !zeroCopy;

	std::vector<float> filterWeights_1(filterLength_1, 1.0 / filterLength_1);
	std::vector<float> filterWeights_2(filterLength_2, 1.0 / filterLength_2);

	printf("Running filters in pipeline. NumElements: %d, FilterLength_1: %d, filterLength_2: %d\n", numElements, filterLength_1, filterLength_2);
	printf("===========\n");

	size_t numBytes = numElements * sizeof(float);
	size_t paddedNumBytes = (numElements + filterLength_1 - 1) * sizeof(float);
	size_t tmpOutputpaddedNumBytes = (numElements + filterLength_2 - 1) * sizeof(float);

	size_t paddedLength = numElements + filterLength_1 - 1;

	GenerateTestData(numElements, filterWeights_1, filterWeights_2, &data, reference, verify);

	result = new float[2 * numElements * UPSCAL_FACTOR]; //Contains both real and imaginary upscaled data 

	if (initOpenCl(&infoDeviceOcl, gpu) == false)
	{
		printf("Error in initOpenCl.\n");
		return false;
	}

	if (zeroCopy)
	{
		for (int i = 0; i < BATCH_KERNELS; i++) {
			device_input_cl[i] = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				2 * paddedNumBytes,
				data, &err);
			CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

			device_tmp_input_cl[i] = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_WRITE,
				2 * tmpOutputpaddedNumBytes,
				NULL, &err);
			CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

			device_output_cl[i] = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
				2 * numBytes * UPSCAL_FACTOR,
				result, &err);
			CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);
		}
	}
	else
	{
		for (int i = 0; i < BATCH_KERNELS; i++) {
			device_input_cl[i] = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_ONLY,
				2 * paddedNumBytes,
				NULL, &err);
			CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

			host_input_cl[i] = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_ALLOC_HOST_PTR,
				2 * paddedNumBytes,
				NULL, &err);
			CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

			device_tmp_input_cl[i] = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_WRITE,
				2 * tmpOutputpaddedNumBytes,
				NULL, &err);
			CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

			device_output_cl[i] = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_WRITE_ONLY,
				2 * numBytes * UPSCAL_FACTOR,
				NULL, &err);
			CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

			host_output_cl[i] = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_ALLOC_HOST_PTR,
				2 * numBytes * UPSCAL_FACTOR,
				NULL, &err);
			CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);
		}
	}

	//Filter data doesn't change. So create read-only device memory
	filter1_coeff_cl = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_ONLY,
		filterLength_1 * sizeof(float), NULL, &err);
	CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

	filter2_coeff = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_ONLY,
		filterLength_2 * sizeof(float), NULL, &err);
	CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);


	//Build OpenCL kernels
	if (!buildKernels(infoDeviceOcl.mCtx, infoDeviceOcl.mDevice, kernel_1, kernel_2, filterLength_1, filterLength_2, BATCH_KERNELS))
	{
		printf("Error in buildKernels.\n");
		return false;
	}

	//Set kernel arguments
	int cnt = 0;
	for (int i = 0; i < BATCH_KERNELS; i++) {
		cnt = 0;
		err = clSetKernelArg(kernel_1[i], cnt++, sizeof(cl_mem), &(device_input_cl[i]));
		err |= clSetKernelArg(kernel_1[i], cnt++, sizeof(cl_mem), &(device_tmp_input_cl[i]));
		err |= clSetKernelArg(kernel_1[i], cnt++, sizeof(cl_mem), &(filter1_coeff_cl));
		CHECK_RESULT(err != CL_SUCCESS, "clSetKernelArg failed with Error code = %d", err);

		cnt = 0;
		err = clSetKernelArg(kernel_2[i], cnt++, sizeof(cl_mem), &(device_tmp_input_cl[i]));
		err |= clSetKernelArg(kernel_2[i], cnt++, sizeof(cl_mem), &(device_output_cl[i]));
		err |= clSetKernelArg(kernel_2[i], cnt++, sizeof(cl_mem), &(filter2_coeff));
		CHECK_RESULT(err != CL_SUCCESS, "clSetKernelArg failed with Error code = %d", err);
	}


	size_t localWorkSize[3] = { LOCAL_XRES, 1, 1 };
	size_t globalWorkSize[3] = { 1, 1, 1 };

	globalWorkSize[0] = numElements;

	cl_int status;

	// FIlter coefficients do not change. So send them once in the beginning of the pipeline
	status = clEnqueueWriteBuffer(infoDeviceOcl.mQueue,
		filter1_coeff_cl, CL_TRUE, 0, filterLength_1 * sizeof(float),
		&filterWeights_1[0], 0, NULL, NULL);
	CHECK_RESULT(status != CL_SUCCESS,
		"Error in clEnqueueWriteBuffer. Status: %d\n", status);

	status = clEnqueueWriteBuffer(infoDeviceOcl.mQueue,
		filter2_coeff, CL_TRUE, 0, filterLength_2 * sizeof(float),
		&filterWeights_2[0], 0, NULL, NULL);
	CHECK_RESULT(status != CL_SUCCESS,
		"Error in clEnqueueWriteBuffer. Status: %d\n", status);

	cl_mem *input_cl, *output_cl, *tmpInput_cl;
	cl_event input_event[BATCH_KERNELS];
	cl_event output_event[BATCH_KERNELS];
	cl_event kernel2_event[BATCH_KERNELS];
	cl_event kernel1_event[BATCH_KERNELS];

	if (dataTransfer) {
		//Getting host pointers for all host pinned buffers
		for (int i = 0; i < BATCH_KERNELS; i++) {
			pinned_input[i] = (float *)clEnqueueMapBuffer(infoDeviceOcl.mQueue, host_input_cl[i], CL_TRUE, CL_MAP_WRITE, 0, 2 * paddedNumBytes, 0, NULL, NULL, &status);
			CHECK_RESULT(status != CL_SUCCESS, "Error in clEnqueueMapBuffer. Status: %d\n", status);

			//My input data doesn't change. In the real application, this memcpy should happen by the secondary thread based on the event callback function or other such means so the main thread is not blocked
			memcpy(pinned_input[i], data, 2 * paddedNumBytes);

			pinned_output[i] = (float *)clEnqueueMapBuffer(infoDeviceOcl.mQueue, host_output_cl[i], CL_FALSE, CL_MAP_READ, 0, 2 * numBytes * UPSCAL_FACTOR, 0, NULL, NULL, &status);
			CHECK_RESULT(status != CL_SUCCESS, "Error in clEnqueueMapBuffer. Status: %d\n", status);

		}
	}

	//Send first set of data in
	if (dataTransfer) {
		status = clEnqueueWriteBuffer(infoDeviceOcl.mQueue, device_input_cl[0], CL_TRUE, 0, 2 * paddedNumBytes, pinned_input[0], 0, NULL, &input_event[0]);
		CHECK_RESULT(status != CL_SUCCESS, "Error in clEnqueueWriteBuffer. Status: %d\n", status);
	}
	
	timer t_timer;

	if (iteration < WARMUP_RUNS) {
		printf("starting timer at position A....");
		timerStart(&t_timer);
	}
	
	//pipeline - will use dual set of images that will be sent through a secondary queue
	//The kernel launch call will wait for the data transfer event to be triggerred
	//Kernels would be launched on the second queue
	int set = 0;
	int iter;
	
	for (iter = 0; iter < iteration; iter++)
	{
		//Not keeping first WARMUP_RUNS iterations data in the performance measurement
		if (iteration >= WARMUP_RUNS && iter == WARMUP_RUNS) {
			clFinish(infoDeviceOcl.kQueue);
			printf("starting timer at position B....");
			timerStart(&t_timer);
			
		}
		
		if (dataTransfer)
		{
			if (iter < (BATCH_KERNELS)) {	//This is required as the first BATCH_KERNELS number of kernels would trigger all the required events for second set run and onwards
				
				set = iter % BATCH_KERNELS;
				input_cl = &device_input_cl[set];
				output_cl = &device_output_cl[set];
				
				//Work on the previously sent data
				err = clEnqueueNDRangeKernel(infoDeviceOcl.kQueue, kernel_1[set], 1, NULL, globalWorkSize, localWorkSize, 1, &input_event[set], &kernel1_event[set]);
				CHECK_RESULT(err != CL_SUCCESS, "clEnqueueNDRangeKernel failed with Error code = %d", err);

				err = clEnqueueNDRangeKernel(infoDeviceOcl.kQueue, kernel_2[set], 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernel2_event[set]);
				CHECK_RESULT(err != CL_SUCCESS, "clEnqueueNDRangeKernel failed with Error code = %d", err);
				clFlush(infoDeviceOcl.kQueue);

				//In my case, the host buffer is pinned, so MapBuffer step is required only once at the beginning, already done that

				//reading half the output as the final stage in the pipeline is supposed to be a compression kernel with 2:1 compression factor
				status = clEnqueueReadBuffer(infoDeviceOcl.mReadQueue, *output_cl, CL_FALSE, 0, 2 * numBytes, pinned_output[set], 1, &kernel2_event[set], &output_event[set]);
				CHECK_RESULT(status != CL_SUCCESS, "Error in clEnqueueWriteBuffer. Status: %d\n", status);	

				//Send next set of data 
				//In my case, the host buffer is pinned, so MapBuffer step is required only once at the beginning, already done that
				status = clEnqueueWriteBuffer(infoDeviceOcl.mQueue, *input_cl, CL_FALSE, 0, 2 * paddedNumBytes, pinned_input[(set + 1) % BATCH_KERNELS], 0, NULL, &input_event[(set + 1) % BATCH_KERNELS]);
				CHECK_RESULT(status != CL_SUCCESS, "Error in clEnqueueWriteBuffer. Status: %d\n", status);

				clFlush(infoDeviceOcl.mReadQueue);
			}
			else {
				set = iter % BATCH_KERNELS;

				input_cl = &device_input_cl[(iter + 1) % BATCH_KERNELS];
				output_cl = &device_output_cl[set];
			
				//Work on the previously sent data
				//kernel1 can start execution only after the input is available
				err = clEnqueueNDRangeKernel(infoDeviceOcl.kQueue, kernel_1[set], 1, NULL, globalWorkSize, localWorkSize, 1, &input_event[set], &kernel1_event[set]);
				CHECK_RESULT(err != CL_SUCCESS, "clEnqueueNDRangeKernel failed with Error code = %d", err);
				
				//kernel2 can start execution only after the previous set output has been read so it has to wait for output_event[set] event
				err = clEnqueueNDRangeKernel(infoDeviceOcl.kQueue, kernel_2[set], 1, NULL, globalWorkSize, localWorkSize, 1, &output_event[set], &kernel2_event[set]);
				CHECK_RESULT(err != CL_SUCCESS, "clEnqueueNDRangeKernel failed with Error code = %d", err);

				clFlush(infoDeviceOcl.kQueue);
				
				//In my case, the host buffer is pinned, so MapBuffer step is required only once at the beginning, already done that

				//reading half the output as the final stage in the pipeline is supposed to be a compression kernel with 2:1 compression factor
				status = clEnqueueReadBuffer(infoDeviceOcl.mReadQueue, *output_cl, CL_FALSE, 0, 2 * numBytes, pinned_output[set], 1, &kernel2_event[set], &output_event[set]);
				CHECK_RESULT(status != CL_SUCCESS, "Error in clEnqueueWriteBuffer. Status: %d\n", status);		

				//Send next set of data 
				//In my case, the host buffer is pinned, so MapBuffer step is required only once at the beginning, already done that
				status = clEnqueueWriteBuffer(infoDeviceOcl.mQueue, *input_cl, CL_FALSE, 0, 2 * paddedNumBytes, pinned_input[(iter + 1) % BATCH_KERNELS], 1, &kernel1_event[(iter + 1) % BATCH_KERNELS], &input_event[(iter + 1) % BATCH_KERNELS]);
				CHECK_RESULT(status != CL_SUCCESS, "Error in clEnqueueWriteBuffer. Status: %d\n", status);

				clFlush(infoDeviceOcl.mReadQueue);
			}

		}
		
		else {	//zero-copy case no data transfer required, this can also be improved by removing clFinish and using events to syncronize a batch of kernels
			err = clEnqueueNDRangeKernel(infoDeviceOcl.kQueue, kernel_1[0], 1, NULL,
				globalWorkSize, localWorkSize, 0, NULL, NULL);
			CHECK_RESULT(err != CL_SUCCESS,
				"clEnqueueNDRangeKernel failed with Error code = %d", err);
			err = clEnqueueNDRangeKernel(infoDeviceOcl.kQueue, kernel_2[0], 1, NULL,
				globalWorkSize, localWorkSize, 0, NULL, NULL);
			CHECK_RESULT(err != CL_SUCCESS,
				"clEnqueueNDRangeKernel failed with Error code = %d", err);
			clFinish(infoDeviceOcl.kQueue);
		}		
	}

	clFinish(infoDeviceOcl.mReadQueue);

	double time_ms = timerCurrent(&t_timer);
	time_ms = 1000 * time_ms;

	iteration = iteration >= WARMUP_RUNS ? iteration - WARMUP_RUNS : iteration;

	printf("iteration: %d\n", iteration);

	if (dataTransfer)
		printf("Average time taken per iteration with data transfer: %f msec\n", time_ms/iteration);
	else
		printf("Average time taken per iteration using zero-copy buffer: %f msec\n", time_ms/iteration);

	if (verify) {
		if (dataTransfer)
			CompareData(reference, pinned_output[0], filterWeights_1.size());
		else 
			CompareData(reference, result, filterWeights_1.size());
	}

	//Clean up memory
	for (int i = 0; i < BATCH_KERNELS; i++) {
		if (device_input_cl[i])
			clReleaseMemObject(device_input_cl[i]);

		if (device_tmp_input_cl[i])
			clReleaseMemObject(device_tmp_input_cl[i]);
		
		if (device_output_cl[i])
			clReleaseMemObject(device_output_cl[i]);	
	}
	clReleaseMemObject(filter1_coeff_cl);
	clReleaseMemObject(filter2_coeff);
	
    return 0;
}

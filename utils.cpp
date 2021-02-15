/* SPDX-License-Identifier: MIT
* Copyright (c) 2021 Ravi Kumar
*/

/**
********************************************************************************
* @file <utils.cpp>
*
* @brief Contains utility functions used by all the samples
*
********************************************************************************
*/
#include "utils.h"

/**
*******************************************************************************
*  @fn     timerStart
*  @brief  Starts the timer 
*
*  @param[in] mytimer  : pointer to the timer
*
*  @return void 
*******************************************************************************
*/
void timerStart(timer* mytimer)
{
#ifdef _WIN32
    QueryPerformanceCounter(&mytimer->start);
#elif defined __MACH__
    mytimer->start = mach_absolute_time();
#else
    struct timespec s;
    clock_gettime(CLOCK_MONOTONIC, &s);
    mytimer->start = (long long)s.tv_sec * (long long)1.0E6 + (long long)s.tv_nsec / (long long)1.0E3;
#endif
}
/**
*******************************************************************************
*  @fn     timerCurrent
*  @brief  Get current time 
*
*  @param[in] mytimer  : pointer to the timer
*
*  @return double : current time
*******************************************************************************
*/
double timerCurrent(timer* mytimer)
{
#ifdef _WIN32
    LARGE_INTEGER stop, frequency;
    QueryPerformanceCounter(&stop);
    QueryPerformanceFrequency(&frequency);
    double time = ((double)(stop.QuadPart - mytimer->start.QuadPart) / frequency.QuadPart);
#elif defined __MACH__
    static mach_timebase_info_data_t info = { 0, 0 };
    if (info.numer == 0)
        mach_timebase_info(&info);
    long long stop = mach_absolute_time();
    double time = ((stop - mytimer->start) * (double) info.numer / info.denom) / 1.0E9;
#else
    struct timespec s;
    long long stop;
    clock_gettime(CLOCK_MONOTONIC, &s);
    stop = (long long)s.tv_sec * (long long)1.0E6 + (long long)s.tv_nsec / (long long)1.0E3;
    double time = ((double)(stop - mytimer->start) / 1.0E6);
#endif
    return time;
}
/**
*******************************************************************************
*  @fn     initOpenCl
*  @brief  This function creates the opencl context and command queue
*
*  @param[in/out] infoDeviceOcl  : pointer to structure
*  @param[in] deviceNum          : pointer to the structure containing opencl
*                                  device information
*
*  @return bool : true if successful; otherwise false.
*******************************************************************************
*/
bool initOpenCl(DeviceInfo *infoDeviceOcl, cl_uint deviceNum)
{
    cl_int err;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = infoDeviceOcl->mCtx;
    cl_command_queue queue = 0;
    cl_event event = NULL;
    
    /**************************************************************************
    * Setup OpenCL environment.                                               *
    **************************************************************************/
    /* Get platform */
    cl_platform_id platform = NULL;    
    size_t param_size = 0;
    char *platformVendor = NULL;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_RESULT(err != CL_SUCCESS, "clGetPlatformIDs failed. Error code = %d", err);
    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, NULL, &param_size);
    CHECK_RESULT(err != CL_SUCCESS, "clGetPlatformInfo failed. Error code = %d", err);
    platformVendor = (char*)malloc(param_size);
    CHECK_RESULT(platformVendor == NULL, "Memory allocation failed: platformVendor");
    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, param_size, platformVendor, NULL);
    CHECK_RESULT(err != CL_SUCCESS, "clGetPlatformInfo failed. Error code = %d", err);
    printf("Found and using platform from vendor: %s\n", platformVendor);

    //CHECK_RESULT(platform == NULL, "AMD platform not found");
    infoDeviceOcl->mPlatform = platform;

#ifdef USE_OPENCL_CPU
    /* Get first CPU on platform */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &infoDeviceOcl->mDevice, 0);
    CHECK_RESULT(err != CL_SUCCESS, "clGetDeviceIDs failed. Error code = %d", err);
#else
    /* Get num GPUs on platform */
    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (err != CL_SUCCESS)
	{
		printf("GPU device not found. Trying to use CPU...\n");
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &infoDeviceOcl->mDevice, 0);
		CHECK_RESULT(err != CL_SUCCESS, "clGetDeviceIDs failed. Error code = %d", err);
		printf("Compatible CPU device found. Using it to run the OpenCL kernel.\n");
	}
	else
	{
	/* Get list of GPUs on platform */
		cl_device_id *devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
		CHECK_RESULT(devices == NULL, "Memory allocation failed: devices");
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, 0);
		CHECK_RESULT(err != CL_SUCCESS, "clGetDeviceIDs failed. Error code = %d", err);
		infoDeviceOcl->mDevice = devices[deviceNum];
		free(devices);
	}
#endif

    props[1] = (cl_context_properties)platform;
    infoDeviceOcl->mCtx = clCreateContext(props, 1, &infoDeviceOcl->mDevice, NULL, NULL, &err);
    CHECK_RESULT(err != CL_SUCCESS, "clCreateContext failed. Err code = %d", err);

    //infoDeviceOcl->mQueue = clCreateCommandQueue(infoDeviceOcl->mCtx, infoDeviceOcl->mDevice, CL_QUEUE_PROFILING_ENABLE, &err);
	infoDeviceOcl->mQueue = clCreateCommandQueue(infoDeviceOcl->mCtx, infoDeviceOcl->mDevice, 0, &err);
    CHECK_RESULT(err != CL_SUCCESS, "clCreateCommandQueue mQueue failed. Err code = %d", err);

	infoDeviceOcl->mReadQueue = clCreateCommandQueue(infoDeviceOcl->mCtx, infoDeviceOcl->mDevice, 0, &err);
	CHECK_RESULT(err != CL_SUCCESS, "clCreateCommandQueue mQueue failed. Err code = %d", err);

	infoDeviceOcl->kQueue = clCreateCommandQueue(infoDeviceOcl->mCtx, infoDeviceOcl->mDevice, 0, &err);
	CHECK_RESULT(err != CL_SUCCESS, "clCreateCommandQueue kQueue failed. Err code = %d", err);

    return true;
}

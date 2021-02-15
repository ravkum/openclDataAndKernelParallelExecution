/* SPDX-License-Identifier: MIT
* Copyright (c) 2021 Ravi Kumar
*/

#ifndef __UTILS__H
#define __UTILS__H
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include "CL/cl.h"
#include "macros.h"

#ifdef _WIN32
#include <windows.h>
#elif defined __MACH__
#include <mach/mach_time.h>
#else
#include <sys/time.h>
#include <linux/limits.h>
#include <unistd.h>
#endif

/******************************************************************************
* Timer structure                                                             *
******************************************************************************/
typedef struct timer
{
#ifdef _WIN32
    LARGE_INTEGER start;
#else
    long long start;
#endif
} timer;

/******************************************************************************
* Structure to hold opencl device information                                 *
******************************************************************************/
typedef struct DeviceInfo
{
    cl_platform_id mPlatform;
    cl_device_id mDevice;
    cl_context mCtx;
    cl_command_queue mQueue;
	cl_command_queue mReadQueue;
	cl_command_queue kQueue;
} DeviceInfo;

void timerStart(timer* mytimer);
double timerCurrent(timer* mytimer);
bool initOpenCl(DeviceInfo *infoDeviceOcl, cl_uint deviceNum);

#endif
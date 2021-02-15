/* SPDX-License-Identifier: MIT
* Copyright (c) 2021 Ravi Kumar
*/


#ifndef __MACROS__H
#define __MACROS__H
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_CL_EVENTS
#define CL_EVENT(x)	x
#else
#define CL_EVENT(x)	NULL
#endif

/******************************************************************************
* Error handling macro.                                                       *
******************************************************************************/
#ifdef _MSC_VER
#define snprintf sprintf_s
#endif
#define CHECK_RESULT(test, msg,...)                                     \
    if ((test))                                                         \
    {                                                                   \
        char *buf = (char*)malloc(4096);                                \
        int rc = snprintf(buf, 4096, msg,  ##__VA_ARGS__);              \
        printf("%s:%d - %s\n", __FILE__, __LINE__, buf);                \
        free(buf);                                                      \
        return false;                                                   \
    }

/**
*******************************************************************************
*  @fn     convertToString
*  @brief  convert the kernel file into a string
*
*  @param[in] filename          : Kernel file name to read
*  @param[out] kernelSource     : Buffer containing the contents of kernel file
*  @param[out] kernelSourceSize : Size of the buffer containing the source
*
*  @return int : 0 if successful; otherwise 1.
*******************************************************************************
*/
inline int convertToString(const char *filename, char **kernelSource, size_t *kernelSourceSize)
{
    FILE *fp = NULL;
    *kernelSourceSize = 0;
    *kernelSource = NULL;
    fp = fopen(filename, "r");
    if (fp != NULL)
    {
        fseek(fp, 0L, SEEK_END);
        *kernelSourceSize = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        *kernelSource = (char*)malloc(*kernelSourceSize);
        if (*kernelSource != NULL)
            *kernelSourceSize = fread(*kernelSource, 1, *kernelSourceSize, fp);
        fclose(fp);
        return 0;
    }
    return 1;
}
#endif

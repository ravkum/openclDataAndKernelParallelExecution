/* SPDX-License-Identifier: MIT
 * Copyright (c) 2021 Ravi Kumar
 */

__kernel 
__attribute__((reqd_work_group_size(LOCAL_XRES, 1, 1)))
void filter_1(
				    __global float2 *input,
					__global float2 *output,
					__constant float *cFilterWeights)

{
    __local float2 local_Input[(LOCAL_XRES + TAP_SIZE_1 - 1)];
	
    size_t threadIdx = get_local_id(0);
	long idx = get_global_id(0);
	size_t numElements = get_global_size(0);

	int local_samples_to_Read = LOCAL_XRES + TAP_SIZE_1 - 1;
	int for_loop_iter = (local_samples_to_Read/LOCAL_XRES);
	int extra_reads = local_samples_to_Read - (for_loop_iter * LOCAL_XRES);

	#pragma unroll 2 //This is 2 for 335 filter size, will be less for small sizes. Curently hardcoding
	for (int i = 0; i < for_loop_iter; i++)	{
		local_Input[threadIdx + (i * LOCAL_XRES)] = input[idx + (i * LOCAL_XRES)];
	}
	
	if (threadIdx < extra_reads)
    {
		local_Input[threadIdx + (for_loop_iter * LOCAL_XRES)] = input[idx + (for_loop_iter * LOCAL_XRES)];
    }
	
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx >= numElements)
		return;

    float2 value = 0.0f;
	float fw = 0.0f;
	
	#pragma unroll
    for(int sIdx = 0; sIdx < TAP_SIZE_1; sIdx++)
    {
		fw = cFilterWeights[sIdx];
		value = mad(local_Input[threadIdx + sIdx], fw, value);		
    }

	output[idx] = value;	
}


__kernel 
__attribute__((reqd_work_group_size(LOCAL_XRES, 1, 1)))
void filter_2(
				    __global float2 *input,
				    __global float4 *output,
					__constant float *cFilterWeights)
{
    
	__local float2 local_Input[LOCAL_XRES + (TAP_SIZE_2 >> 2)];
	
	size_t threadIdx = get_local_id(0);
	long idx = get_global_id(0);
	size_t numElements = get_global_size(0);

	//Up-scale and apply fir filter
	int local_samples_to_Read = LOCAL_XRES + (TAP_SIZE_2 >> 1);
	int extra_reads = local_samples_to_Read - LOCAL_XRES;

	//Hard-coding for TAP_SIZE 29. loop is not required
	local_Input[threadIdx] = input[idx];
	
	if (threadIdx < extra_reads)
	{
		local_Input[threadIdx + LOCAL_XRES] = input[LOCAL_XRES + idx];
    }
	
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx >= numElements)
		return;

    float4 t_output = 0.0f;
	float4 val = 0.0f;

	float2 input_cur = 0.0f;
	float2 input_next = 0.0f;
	
	float fw = 0.0f;

	input_cur  = local_Input[threadIdx];
	int sIdx;

	#pragma unroll
    for(sIdx = 0; sIdx < (TAP_SIZE_2/2); sIdx++)
    {
		val.xy = input_cur;
		val.zw = input_cur;
		
		fw = cFilterWeights[sIdx*2 + 0];
		t_output += fw * val;

		input_next = local_Input[threadIdx + sIdx + 1];
		input_cur = input_next;
		
		val.zw = input_next;
		
		fw = cFilterWeights[sIdx*2 + 1];
		t_output += fw * val;
	}

	val.xy = input_cur;
	val.zw = input_cur;//local_Input[threadIdx + sIdx + 1];	
		
	fw = cFilterWeights[sIdx*2];
	t_output += fw*val;
	
	output[idx] = t_output;
}

This sample shows how to hide data transfer behind kernel run time using OpenCL while executing workload in dGPU. 

Prerequisite:
1) OpenCL header and lib
2) export AMDAPPSDKROOT=/opt/rocm 
    or to any path where the OpenCL SDK is present.

Build:
1) Linux:
  mkdir build
  cd build
  cmake ../
  make clean all
  
2) Windows:
  Open the .sln fileon Visusl Studio and build. 
  
 
Run:
Usage: firFilter.exe

        [-zeroCopy (0 | 1)] //0 (default) - Device buffer, 1 - zero copy buffer
        [-numElements <int value> (default 245760)]
        [-f1 <int value> (filter1 TAP size - default 335)]
        [-f2 <int value> (filter2 TAP size - default 29)]
        [-iter <int value> (Number of times the piepeline shold be run)]
        [-verify (0 | 1)]
	[-gpu (0 | 1) (in mGPU case, 0 = dGPU, 1 = iGPU)]
        [-h (help)]

Example:

       firFilter.exe -numElements 256000 -verify 1 -iter 1000 -zeroCopy 0
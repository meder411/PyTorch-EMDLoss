#ifndef EMD_CUH_
#define EMD_CUH_

#include "cuda_helper.h"

#define BLOCK_SIZE 512

template <typename T>
__global__ void approx_match_kernel(
	const int64_t b, const int64_t n, const int64_t m, const int64_t d, 
	const T * __restrict__ xyz1,
	const T * __restrict__ xyz2,
	T * __restrict__ match,
	T * temp)
{
	// Pointers to temporary storage for this current block
	// Starting point for batch = blockIdx.x * (n + m)
	T * remainL = temp + blockIdx.x * (n + m) * 2; // Start of set 1
	T * remainR = temp + blockIdx.x * (n + m) * 2 + n; // Start of set 2
	T * ratioL = temp + blockIdx.x * (n + m) * 2 + n + m; // Start of set 1
	T * ratioR = temp + blockIdx.x * (n + m) * 2 + n + m + n; // Start of set 2
	
	// Ratio of two point sets
	T multiL;
	T multiR;
	if (n >= m)
	{
		multiL = 1;
		multiR = n / m;
	}
	else
	{
		multiL = m / n;
		multiR = 1;
	}

	// Dynamic shared memory buffer for templated function
	// https://stackoverflow.com/a/27570775/3427580
	extern __shared__ __align__(sizeof(T)) unsigned char my_buf[];
	T *buf = reinterpret_cast<T*>(my_buf);

	// For each batch
	for (int64_t i = blockIdx.x; i < b; i += gridDim.x)
	{
		// Initialize match values
		for (int64_t j = threadIdx.x; j < n*m; j += blockDim.x) 
		{
			match[i*n*m+j] = 0;
		}
		for (int64_t j = threadIdx.x; j < n; j += blockDim.x) 
		{ 
			remainL[j] = multiL;
		}
		for (int64_t j = threadIdx.x; j < m; j += blockDim.x)
		{
			remainR[j] = multiR;
		}
		__syncthreads();


		for (int64_t j = 7; j >= -2; j--)
		{
			T level = -pow(4.0, j);
			if (j == -2) { level = 0; }

			// Iterate over blocks
			for (int64_t k0 = 0; k0 < n; k0 += blockDim.x)
			{
				// Current thread linear index
				int64_t k = k0 + threadIdx.x;

				// Initialize a really small, non-zero sum
				T suml = T(1e-9);

				// Iterate over grid
				for (int64_t l0 = 0; l0 < m; l0 += BLOCK_SIZE)
				{
					// End of the block or m, whichever comes first
					int64_t lend = min(m, l0 + BLOCK_SIZE) - l0;

					// Put points from the second set into the shared buffer 
					for (int64_t l = threadIdx.x; l < lend; l += blockDim.x)
					{
						for(int64_t z = 0; z < d; z++)
						{
							buf[l*(d+1)+z] = xyz2[i*m*d+l0*d+l*d+z];;
							
						}
						buf[l*(d+1)+d] = remainR[l0+l];
					}
					__syncthreads();

					for (int64_t l = 0; l < lend; l++)
					{
						T v = 0;
						for (int64_t z = 0; z < d; z++)
						{
							if (k < n)
							{
								v += (buf[l*(d+1)+z] - xyz1[i*n*d+k*d+z]) * 
									(buf[l*(d+1)+z] - xyz1[i*n*d+k*d+z]);
							}
							else
							{
								v += buf[l*(d+1)+z] * buf[l*(d+1)+z];
							}
						}
						v *= level;
						suml += exp(v)*buf[l*(d+1)+d];
					}
					__syncthreads();
				}
				if (k < n) { ratioL[k] = remainL[k] / suml; }
			}
			__syncthreads();

			// Iterate over blocks again (now for second point set)
			for (int64_t l0 = 0; l0 < m; l0 += blockDim.x)
			{
				int64_t l = l0 + threadIdx.x;
				T sumr = 0;
				for (int64_t k0 = 0; k0 < n; k0 += BLOCK_SIZE)
				{
					int64_t kend = min(n, k0 + BLOCK_SIZE) - k0;
					for (int64_t k = threadIdx.x; k < kend; k += blockDim.x)
					{
						for (int64_t z = 0; z < d; z++)
						{
							buf[k*(d+1)+z] = xyz1[i*n*d+k0*d+k*d+z];
						}
						buf[k*(d+1)+d] = ratioL[k0+k];
					}
					__syncthreads();

					for (int64_t k = 0; k < kend; k++)
					{
						T v = 0;
						for (int64_t z = 0; z < d; z++)
						{
							if (l < m)
							{
								v += (xyz2[i*m*d+l*d+z] - buf[l*(d+1)+z]) * 
									(xyz2[i*m*d+l*d+z] - buf[l*(d+1)+z]);
							}
							else
							{
								v += buf[l*(d+1)+z] * buf[l*(d+1)+z];
							}
						}
						v *= level;
						sumr += exp(v)*buf[k*(d+1)+d];
					}
					__syncthreads();
				}

				if (l < m)
				{
					sumr *= remainR[l];
					T consumption = fmin(remainR[l] / (sumr + 1e-9), 1.0);
					// ******************************
					// SOURCE OF THE ISSUE: sumr
					// Any variable that is a function of sumr causes an error
					// Specifically the assignments below
					// It's an issue only with large m and n. Maybe it's a 
					// overflow issue?
					// ******************************
					ratioR[l] = consumption * remainR[l];
					remainR[l] = fmax(0.0, remainR[l] - sumr);
				}
			}
			__syncthreads();

			for (int64_t k0 = 0; k0 < n; k0 += blockDim.x)
			{
				int64_t k = k0 + threadIdx.x;
				T suml=0;

				for (int64_t l0 = 0; l0 < m; l0 += BLOCK_SIZE)
				{
					int64_t lend = min(m, l0 + BLOCK_SIZE) - l0;
					for (int64_t l = threadIdx.x; l < lend; l += blockDim.x)
					{

						for(int64_t z = 0; z < d; z++)
						{
							buf[l*(d+1)+z] = xyz2[i*m*d+l0*d+l*d+z];;
							
						}
						buf[l*(d+1)+d] = ratioR[l0+l];
					}
					__syncthreads();

					T rl = ratioL[k];
					if (k < n)
					{
						for (int64_t l = 0; l < lend; l++)
						{

							T v = 0;
							for (int64_t z = 0; z < d; z++)
							{
								if (k < n)
								{
									v += (buf[l*(d+1)+z] - xyz1[i*n*d+k*d+z]) *
										(buf[l*(d+1)+z] - xyz1[i*n*d+k*d+z]);
								}
								else
								{
									v += buf[l*(d+1)+z] * buf[l*(d+1)+z];
								}
							}
							v *= level;

							T w = __expf(v)*buf[l*(d+1)+d]*rl;
							match[i*n*m+(l0+l)*n+k] += w;
							suml += w;
						}
					}
					__syncthreads();
				}

				if (k < n) { remainL[k] = fmaxf(0.0f, remainL[k] - suml); }
			}
			__syncthreads();
		}
	}
}

void approx_match(
	const int64_t b, const int64_t n, 
	const int64_t m, const int64_t d,
	const at::Tensor xyz1,
	const at::Tensor xyz2,
	at::Tensor match, 
	at::Tensor temp)
{
	AT_DISPATCH_FLOATING_TYPES(match.type(), "approx_match_kernel", ([&] {
		approx_match_kernel
			<<<32, 512, BLOCK_SIZE*(d+1)*sizeof(scalar_t)>>>(
			b, n, m, d, 
			xyz1.data<scalar_t>(),
			xyz2.data<scalar_t>(),
			match.data<scalar_t>(),
			temp.data<scalar_t>());
	}));
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError())
}



template <typename T>
__global__ void match_cost_kernel(
	const int64_t b, const int64_t n, const int64_t m, const int64_t d, 
	const T * __restrict__ xyz1,
	const T * __restrict__ xyz2,
	const T * __restrict__ match,
	T * __restrict__ out)
{
	// First 512 elements is used for sum computation
	// Remaining buffer is a general buffer
	extern __shared__ __align__(sizeof(T)) unsigned char my_buf[];
	T *buf = reinterpret_cast<T*>(my_buf);

	for (int64_t i=blockIdx.x;i<b;i+=gridDim.x)
	{
		T subsum=0;
		for (int64_t k0 = 0; k0 < n; k0 += blockDim.x)
		{
			int64_t k=k0+threadIdx.x;
			for (int64_t l0 = 0; l0 < m; l0 += BLOCK_SIZE)
			{
				int64_t lend = min(m, l0 + BLOCK_SIZE) - l0;
				for (int64_t l = threadIdx.x; l < lend * d; l += blockDim.x) 
				{
					buf[512+l]=xyz2[i*m*d+l0*d+l];
				}
				__syncthreads();

				if (k < n)
				{
					for (int64_t l = 0; l < lend; l++)
					{
						T v = 0;
						for (int64_t z = 0; z < d; z++)
						{
							v += (buf[512+l*d+z] - xyz1[i*n*d+k*d+z]) * 
								(buf[512+l*d+z] - xyz1[i*n*d+k*d+z]);
						}
						subsum += sqrtf(v)*match[i*n*m+(l0+l)*n+k];
					}
				}
				__syncthreads();
			}
		}
		buf[threadIdx.x] = subsum;
		for (int64_t j = 1; j < blockDim.x; j <<= 1)
		{
			__syncthreads();
			if ((threadIdx.x & j) == 0 && threadIdx.x + j <blockDim.x)
			{
				buf[threadIdx.x] += buf[threadIdx.x+j];
			}
		}
		if (threadIdx.x == 0) { out[i] = buf[0]; }
		__syncthreads();
	}
}


void match_cost(
	const int64_t b, const int64_t n,const int64_t m, const int64_t d,
	const at::Tensor xyz1,
	const at::Tensor xyz2,
	const at::Tensor match,
	at::Tensor out)
{
	AT_DISPATCH_FLOATING_TYPES(xyz1.type(), "match_cost_kernel", ([&] {
		unsigned shared_mem_size = (512+BLOCK_SIZE*d)*sizeof(scalar_t);
		match_cost_kernel<<<32, 512, shared_mem_size>>>(
			b, n, m, d, 
			xyz1.data<scalar_t>(),
			xyz2.data<scalar_t>(),
			match.data<scalar_t>(),
			out.data<scalar_t>());
	}));
	CUDA_CHECK(cudaGetLastError())
}



template <typename T>
__global__ void match_cost_grad2_kernel(
	const int64_t b, const int64_t n, 
	const int64_t m, const int64_t d, 
	const T * __restrict__ xyz1,
	const T * __restrict__ xyz2,
	const T * __restrict__ match,
	T * __restrict__ grad2)
{

	extern __shared__ __align__(sizeof(T)) unsigned char my_buf[];
	T *sum_grad = reinterpret_cast<T*>(my_buf);

	for (int64_t i = blockIdx.x; i < b; i += gridDim.x)
	{
		int64_t kbeg = m*blockIdx.y / gridDim.y;
		int64_t kend = m*(blockIdx.y+1) / gridDim.y;
		for (int64_t k = kbeg; k < kend; k++)
		{
			for (int64_t j = threadIdx.x; j < n; j += blockDim.x)
			{
				T v = 0;
				for (int64_t z = 0; z < d; z++)
				{
					v += (xyz2[(i*m+k)*d+z] - xyz1[(i*n+j)*d+z]) * 
						(xyz2[(i*m+k)*d+z] - xyz1[(i*n+j)*d+z]);
				}
				T w = match[i*n*m+k*n+j] * rsqrtf(fmaxf(v, 1e-20f));

				for (int64_t z = 0; z < d; z++)
				{
					sum_grad[threadIdx.x*d+z] += xyz1[(i*n+j)*d+z] * w;
				}
			}
			for (int64_t j = 1; j < blockDim.x; j <<= 1)
			{
				__syncthreads();
				int64_t j1 = threadIdx.x;
				int64_t j2 = threadIdx.x + j;
				if ((j1 & j) == 0 && j2 < blockDim.x)
				{
					for (int64_t z = 0; z < d; z++)
					{
						sum_grad[j1*d+z] += sum_grad[j2*d+z];
					}
				}
			}
			if (threadIdx.x == 0)
			{
				for (int64_t z = 0; z < d; z++)
				{
					grad2[(i*m+k)*d+z] = sum_grad[z];
				}
			}
			__syncthreads();
		}
	}
}


template <typename T>
__global__ void match_cost_grad1_kernel(
	const int64_t b, const int64_t n, 
	const int64_t m, const int64_t d, 
	const T * __restrict__ xyz1,
	const T * __restrict__ xyz2,
	const T * __restrict__ match,
	T * __restrict__ grad1)
{
	for (int64_t i = blockIdx.x; i < b; i += gridDim.x)
	{
		for (int64_t l = threadIdx.x; l < n; l += blockDim.x)
		{
			for (int64_t k = 0; k < m; k++)
			{
				T v = 0;
				for (int64_t z = 0; z < d; z++)
				{
					v += (xyz1[i*n*d+l*d+z] - xyz2[i*m*d+k*d+z]) * 
						(xyz1[i*n*d+l*d+z] - xyz2[i*m*d+k*d+z]);
				}
				T w = match[i*n*m+k*n+l] * rsqrtf(fmaxf(v, 1e-20f));

				for (int64_t z = 0; z < d; z++)
				{
					grad1[i*n*d+l*d+z] += 
						(xyz1[i*n*d+l*d+z] - xyz2[i*m*d+k*d+z]) * w;
				}				
			}
		}
	}
}

void match_cost_grad(
	const int64_t b, const int64_t n, 
	const int64_t m, const int64_t d, 
	const at::Tensor xyz1,
	const at::Tensor xyz2,
	const at::Tensor match,
	at::Tensor grad1,
	at::Tensor grad2)
{
	AT_DISPATCH_FLOATING_TYPES(xyz1.type(), "match_cost_grad1_kernel", ([&] {
		match_cost_grad1_kernel<<<32,512>>>(
			b, n, m, d, 
			xyz1.data<scalar_t>(),
			xyz2.data<scalar_t>(),
			match.data<scalar_t>(),
			grad1.data<scalar_t>());
	}));
	CUDA_CHECK(cudaGetLastError())

	AT_DISPATCH_FLOATING_TYPES(xyz1.type(), "match_cost_grad2_kernel", ([&] {
		match_cost_grad2_kernel<<<dim3(32,32),512,(512*d)*sizeof(scalar_t)>>>(
			b, n, m, d, 
			xyz1.data<scalar_t>(),
			xyz2.data<scalar_t>(),
			match.data<scalar_t>(),
			grad2.data<scalar_t>());
	}));
	CUDA_CHECK(cudaGetLastError())
}


#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <chrono>
#include <algorithm>

#ifdef _WIN32
#include <Windows.h>

struct HighResClock
{
	typedef long long                               rep;
	typedef std::nano                               period;
	typedef std::chrono::duration<rep, period>      duration;
	typedef std::chrono::time_point<HighResClock>   time_point;
	static const bool is_steady = true;

	static time_point now();
};

namespace
{
	const long long g_Frequency = []() -> long long
	{
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		return frequency.QuadPart;
	}();
}

HighResClock::time_point HighResClock::now()
{
	LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return time_point(duration(count.QuadPart * static_cast<rep>(period::den) / g_Frequency));
}
#endif

#include <pthread.h>
#include <math.h>
#define NB_THREADS 4
#define MAX_WORKLOAD 200
      
pthread_attr_t attr;
pthread_t thread[NB_THREADS];
struct merge_work {
	int begin;
	int end;
};
typedef struct merge_work merge_work_t;

struct thread_sort_arg
{
	int *arr;
	int begin[MAX_WORKLOAD];
	int end[MAX_WORKLOAD];
	int workload;
};
typedef struct thread_sort_arg thread_sort_arg_t;


struct thread_sort_improved_arg
{
	int *arr;
	int begin[MAX_WORKLOAD];
	int end[MAX_WORKLOAD];
	int workload;
};
typedef struct thread_sort_improved_arg thread_sort_improved_arg_t;

struct thread_mergesort_merge_arg
{
	int *arr;
	int *tmp;
	int toLevel;
	int work_count;
	merge_work_t *merge_works;
	int id;
};
typedef struct thread_mergesort_merge_arg thread_mergesort_merge_arg_t;

thread_sort_arg_t arg[NB_THREADS]; 
thread_sort_improved_arg_t arg_improved[NB_THREADS]; 
thread_mergesort_merge_arg_t arg_merge[NB_THREADS]; 



inline void swap(int & a, int & b)
{
	int tmp = a;
	a = b;
	b = tmp;
}

void insertionSort(int * arr, int left, int right)
{
	int j;
	for (int i = left + 1; i < right; ++i)
	{
		int tmp = arr[i];
		for (j = i; j > left && tmp < arr[j - 1]; --j)
			arr[j] = arr[j - 1];
		arr[j] = tmp;
	}
}

inline int selectPivot(int * arr, int left, int right)
{
	int center = (left + right) / 2;
	if (arr[center] < arr[left])
		swap(arr[left], arr[center]);
	if (arr[right] < arr[left])
		swap(arr[left], arr[right]);
	if (arr[right] < arr[center])
		swap(arr[center], arr[right]);

	swap(arr[center], arr[right - 1]);
	return arr[right - 1];
}

void quicksort(int * arr, int begin, int end)
{
	if (end - begin < 50)
	{
		insertionSort(arr, begin, end);
		return;
	}

	int pivot = selectPivot(arr, begin, end-1);
	int l = begin;
	int u = end - 2;

	while (1)
	{
		while (arr[++l] < pivot);
		while (arr[--u] > pivot);

		if (l < u)
			swap(arr[l], arr[u]);
		else
			break;
	}

	swap(arr[l], arr[end - 1]);

	quicksort(arr, begin, l);
	quicksort(arr, l, end);
}

void merge(	int * arr, int * tmp,
			int left_begin, int left_end,
			int right_begin, int right_end)
{
	int l = left_begin;
	int r = right_begin;
	int i = left_begin;

	while (l <= left_end && r <= right_end)
	{
		if (arr[l] < arr[r])
			tmp[i++] = arr[l++];
		else
			tmp[i++] = arr[r++];
	}
/*
	while (l <= left_end)
		tmp[i++] = arr[l++];

	while (r <= right_end)
		tmp[i++] = arr[r++];
*/
	if(l <= left_end) 
		memcpy(tmp+i, arr+l, (left_end-l+1)*sizeof(int));
	else if(r <= right_end) 
		memcpy(tmp+i, arr+r, (right_end-r+1)*sizeof(int));

}

void*
thread_sort(void* arg)
{
	thread_sort_arg_t *a = (thread_sort_arg_t *) arg;

	for (int i = 0; i < a->workload; ++i)
	{
		quicksort(a->arr, a->begin[i], a->end[i]);
	}
	return NULL;
}

void*
thread_mergesort(void* arg)
{
	thread_sort_improved_arg_t *a = (thread_sort_improved_arg_t *) arg;

	int i;
	for (i = 0; i < a->workload; ++i)
	{
		quicksort(a->arr, a->begin[i], a->end[i]);
	}
	return NULL;
}


void*
thread_mergesort_merge(void* arg)
{
	thread_mergesort_merge_arg_t *a = (thread_mergesort_merge_arg_t *) arg;

	int toLevel = a->toLevel;
	int work_count = a->work_count;
	merge_work_t *merge_works = a->merge_works;

	int i;
	for(i = 0; i < toLevel; ++i) {
		int d = pow(2, i +1);
		int chunk_count = (work_count / d);

		int use_threads = NB_THREADS;
		if(use_threads % 2 == 1)
			use_threads--;

		int sep = chunk_count / use_threads;
		int k_start = a->id * sep;
		int k_stop = ((a->id+1))*sep;

		//printf("[%i] %i -> %i \n", a->id, k_start, k_stop);
		// printf("usingk[%i] \n", use_threads);
		int k;
		for(k = k_start; k < k_stop; ++k) {
			int left_begin_id = k * d;
			int right_end_id = (k+1) * d-1;
			int center = (right_end_id + left_begin_id) / 2;
			int left_end_id = center;
			int right_begin_id = center +1;

			int left_begin_index = merge_works[left_begin_id].begin;
			int left_end_index = merge_works[left_end_id].end -1;
			int right_begin_index = merge_works[right_begin_id].begin;
			int right_end_index = merge_works[right_end_id].end -1;

			merge(a->arr, a->tmp, left_begin_index, left_end_index, right_begin_index, right_end_index);
			memcpy(a->arr+left_begin_index, a->tmp+left_begin_index, (right_end_index-left_begin_index+1)*sizeof(int));
		}
	}
	return NULL;
}


#define MIN_WORK 10000
#define MAX_WORKGROUP_SIZE 100000
void mergesort_flat(int * arr, int size)
{
	int * tmp = (int*)malloc(size * sizeof(int));
	
	if(size <= MIN_WORK || NB_THREADS == 1)
	{
		quicksort(arr, 0, size);
		return;
	}
	
	int i=0;
	int workgroup_count = ceil(size / MAX_WORKGROUP_SIZE);
//	if(NB_THREADS % 2 == 0 && workgroup_count % 2 != 0)
//		workgroup_count++;
	int work_count = workgroup_count * NB_THREADS;
	work_count = pow(2, ceil(log(work_count)/log(2)));
	int work_size = size / work_count;
	
	//printf("WG_COUNT %i, W_COUNT %i, W_SIZE %i\n", workgroup_count, work_count, work_size);



	merge_work_t * merge_works = (merge_work_t*)malloc(work_count * sizeof(merge_work_t));
	
	// The work should be done in parallel
	for(i=0; i<work_count; ++i)
	{
		int id = i%NB_THREADS;
		int workload = arg_improved[id].workload++;
		arg_improved[id].arr = arr;
		arg_improved[id].begin[workload] = 0+i*work_size;
		arg_improved[id].end[workload] = arg_improved[id].begin[workload] + work_size;

		if(i == work_count-1) {
			arg_improved[id].end[workload] = size;
		}

		merge_works[i].begin = arg_improved[id].begin[workload];
		merge_works[i].end = arg_improved[id].end[workload];
		
	//	printf("WORK: t_id %i, workload %i, begin %i, end %i \n", id, workload, arg_improved[id].begin[workload], arg_improved[id].end[workload]); 
				
	}

	int levels = ceil(log2(work_count));
	//printf("levels: %i\n", levels);
	
	// do the partial sorting
	
	for (i = 0; i < NB_THREADS; i++)
		pthread_create(&thread[i], &attr, &thread_mergesort, (void*) &arg_improved[i]);
	for (i = 0; i < NB_THREADS; i++)
		pthread_join(thread[i], NULL);

	int use_threads = NB_THREADS;
	if(use_threads % 2 == 1)
		use_threads--;

	int split_level = levels - ceil(log2(use_threads));
	//printf("split_level: %i\n", split_level);

	for(i=0; i<use_threads; ++i)
	{
		arg_merge[i].arr = arr;
		arg_merge[i].tmp = tmp;
		arg_merge[i].merge_works = merge_works;
		arg_merge[i].work_count = work_count;
		arg_merge[i].toLevel = split_level;
		arg_merge[i].id = i;

		pthread_create(&thread[i], &attr, &thread_mergesort_merge, (void*) &arg_merge[i]);	
		//pthread_join(thread[i], NULL);
	}
	//
	for (i = 0; i < use_threads; i++)
		pthread_join(thread[i], NULL);

	
	// merge
	

	int start_level = split_level;
	if(split_level < 0)
		start_level = 0;
	for(i = start_level; i < levels; ++i) {
		int d = pow(2, i +1);
		int chunk_count = (work_count / d);
		int k;
		for(k = 0; k < chunk_count; ++k) {

			int left_begin_id = k * d;
			int right_end_id = (k+1) * d-1;
			int center = (right_end_id + left_begin_id) / 2;
			int left_end_id = center;
			int right_begin_id = center +1;

			int left_begin_index = merge_works[left_begin_id].begin;
			int left_end_index = merge_works[left_end_id].end -1;
			int right_begin_index = merge_works[right_begin_id].begin;
			int right_end_index = merge_works[right_end_id].end -1;
			
			//printf("left_begin_id: %i, left_end_id: %i, right_begin_id:%i, right_end_id:%i\n",left_begin_id, left_end_id, right_begin_id, right_end_id);

			//int left_index = merge_works[left_chunk].begin;
			//int right_index = merge_works[right_chunk].end;

			merge(arr, tmp, left_begin_index, left_end_index, right_begin_index, right_end_index);
			
			//printf("left_begin_index: %i, left_end_index: %i, right_begin_index:%i, right_end_index:%i\n",left_begin_index, left_end_index, right_begin_index, right_end_index);
			memcpy(arr+left_begin_index, tmp+left_begin_index, (right_end_index-left_begin_index+1)*sizeof(int));


			//printf("Level: %i, left_begin_id: %i, left_end_id:%i \n", i, left_begin_id, left_end_id);
			//printf("Level: %i, right_begin_id: %i, right_end_id:%i \n", i, right_begin_id, right_end_id);

			//printf("\n");
			//printf("Idn: %i, left: %i, right:%i \n", i, left_index, right_index);
		}
	}
	



}

int main()
{
	int size = 10000000;
	int *data = (int*)malloc(size * sizeof(int));

	int *ref  = (int*)malloc(size * sizeof(int));

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 

	srand(time(0));
	for (int i = 0; i < size; ++i)
	{
		data[i] = rand()%56;
		ref[i] = data[i];
	}

	//printf("\n");

	#ifdef _WIN32
	HighResClock::time_point start = HighResClock::now();
	#else
	auto start = std::chrono::high_resolution_clock::now();
	#endif
	//quicksort(data, 0, size);
	//std::sort(data, data+size);
	//mergesort(data, size);
	mergesort_flat(data, size);
	#ifdef _WIN32
	HighResClock::time_point stop = HighResClock::now();
	#else
	auto stop = std::chrono::high_resolution_clock::now();
	#endif


	std::chrono::duration<double> dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
	

#ifndef NDEBUG
	printf("Using %i threads\n", NB_THREADS);
	printf("time taken %f \n", dur.count() * 1.0e3);
	printf("expected time %f \n", 10000000 / size * dur.count() * 1.0e3);

	std::sort(ref, ref+size);

/*
	printf("OUR: ");
	for(int i=0; i< size; ++i)
		printf("%i ",data[i]);


	printf("\nSTD: ");
	for(int i=0; i< size; ++i)
		printf("%i ",ref[i]);

	printf("\n");
*/
	for (int i = 0; i < size; ++i)
	{
		if (data[i] != ref[i])
		{
			printf("ERROR at [%i] \n", i);
			break;
		}
	}
#else
	printf("%f ", dur.count() * 1.0e3);
#endif

	//printf("\n");

	free(data);
	free(ref);

	return 0;
}
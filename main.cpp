#include <cstdio>
#include <cstdlib>

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

inline void swap(int & a, int & b)
{
	int tmp = a;
	a = b;
	b = tmp;
}

void insertionSort(int * arr, int left, int right)
{
	int j;
	for (int i = left + 1; i <= right; ++i)
	{
		int tmp = arr[i];
		for (j = i; j > 0 && tmp < arr[j - 1]; --j)
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

void quicksort(int * arr, int left, int right)
{
	if (right - left <= 48)
	{
		insertionSort(arr, left, right);
		return;
	}

	int pivot = selectPivot(arr, left, right);
	int l = left;
	int u = right - 1;

	while (1)
	{
		while (arr[++l] < pivot);
		while (arr[--u] > pivot);

		if (l < u)
			swap(arr[l], arr[u]);
		else
			break;
	}

	swap(arr[l], arr[right - 1]);

	quicksort(arr, left, l - 1);
	quicksort(arr, l + 1, right);
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

	while (l <= left_end)
		tmp[i++] = arr[l++];

	while (r <= right_end)
		tmp[i++] = arr[r++];
}

void mergesort_recursive(int * arr, int * tmp, int left, int right)
{
	/*
	printf("S: ");
	for(int i=left; i<=right; ++i)
		printf("%i ",arr[i]);
	printf("\n");
*/
	if (right - left < 1)
	{
		//quicksort(arr, left, right);
		return;
	}

	int center = (left + right) / 2;
	mergesort_recursive(arr, tmp, left, center);
	mergesort_recursive(arr, tmp, center+1, right);

	merge(arr, tmp, left, center, center+1, right);

	memcpy(arr+left, tmp+left, (right-left+1)*sizeof(int));
/*
	printf("T: ");
	for(int i=left; i<=right; ++i)
		printf("%i ",tmp[i]);
	printf("\n");

	printf("M: ");
	for(int i=left; i<=right; ++i)
		printf("%i ",arr[i]);
	printf("\n");
	*/
}

void mergesort(int * arr, int size)
{
	int * tmp = (int*)malloc(size * sizeof(int));

	mergesort_recursive(arr, tmp, 0, size - 1);

	free(tmp);
}

int main()
{
	int size = 10000;
	int *data = (int*)malloc(size * sizeof(int));
	int *ref  = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; ++i)
	{
		data[i] = rand()%100;
		ref[i] = data[i];
	}

	printf("\n");

	#ifdef _WIN32
	HighResClock::time_point start = HighResClock::now();
	#else
	auto start = std::chrono::high_resolution_clock::now();
	#endif
	//quicksort(data, 0, size-1);
	//std::sort(data, data+size);
	mergesort(data, size);
	#ifdef _WIN32
	HighResClock::time_point stop = HighResClock::now();
	#else
	auto stop = std::chrono::high_resolution_clock::now();
	#endif

	std::sort(ref, ref+size);

	printf("\n");

	std::chrono::duration<double> dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

	printf("time taken %f \n", dur.count() * 1.0e3);
	printf("expected time %f \n", 10000000 / size * dur.count() * 1.0e3);

	for (int i = 0; i < size; ++i)
	{
		if (data[i] != ref[i])
		{
			printf("ERROR at [%i] \n", i);
			break;
		}
	}

	printf("\n");

	free(data);
	free(ref);

	return 0;
}
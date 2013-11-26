/*
 * sort.c
 *
 *  Created on: 5 Sep 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

// Do not touch or move these lines
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "disable.h"

#ifndef DEBUG
#define NDEBUG
#endif

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"

 pthread_attr_t attr;
pthread_t thread[NB_THREADS];

struct thread_sort_arg
{
	int *arr;
	int begin;
	int end;
};
typedef struct thread_sort_arg thread_sort_arg_t;

thread_sort_arg_t arg[NB_THREADS]; 

inline void swap(int *a, int *b)
{
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

void insertionSort(int * arr, int left, int right)
{
	int j,i;
	for (i = left + 1; i < right; ++i)
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
		swap(arr + left, arr+ center);
	if (arr[right] < arr[left])
		swap(arr + left, arr + right);
	if (arr[right] < arr[center])
		swap(arr + center, arr + right);

	swap(arr +center, arr+right - 1);
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
			swap(arr+l, arr+u);
		else
			break;
	}

	swap(arr+l, arr+end - 1);

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
	quicksort(a->arr, a->begin, a->end);
	return NULL;
}

void mergesort_recursive(int * arr, int * tmp, int left, int right)
{
	/*
	printf("S: ");
	for(int i=left; i<=right; ++i)
		printf("%i ",arr[i]);
	printf("\n");
*/
	if (right - left < 250000)
	{
		int size = right - left + 1;
		int tsize = size / NB_THREADS;
		int i;
		//printf("'size' %i \n", size);
		for (i = 0; i < NB_THREADS; i++)
		{
			arg[i].arr = arr;
			arg[i].begin = left+i*tsize;
			arg[i].end = arg[i].begin + tsize;
			if(i == NB_THREADS-1)
				arg[i].end = right+1;
			//printf("%i : %i -> %i\n", i, arg[i].begin, arg[i].end);
			pthread_create(&thread[i], &attr, &thread_sort, (void*) &arg[i]);
		}
		for (i = 0; i < NB_THREADS; i++)
		{
			pthread_join(thread[i], NULL);
		}

		i = 1;
		merge(arr, tmp, arg[i-1].begin, arg[i].begin-1, arg[i].begin, arg[i].end-1);
		//memcpy(arr+arg[i-1].begin, tmp+arg[i-1].begin, (arg[i].end-arg[i-1].begin)*sizeof(int));

		i = 3;
		merge(arr, tmp, arg[i-1].begin, arg[i].begin-1, arg[i].begin, arg[i].end-1);
		//memcpy(arr+arg[i-1].begin, tmp+arg[i-1].begin, (arg[i].end-arg[i-1].begin)*sizeof(int));

		//memcpy(arr+left, tmp+left, (size)*sizeof(int));
		merge(tmp,arr , arg[0].begin, arg[1].end-1, arg[2].begin, arg[3].end-1);
		
		//memcpy(arr+left, tmp+left, (size)*sizeof(int));
/*
		merge(arr, tmp, arg[0].begin, arg[1].begin-1, arg[1].begin, arg[1].end-1);
		merge(arr, tmp, arg[2].begin, arg[3].begin-1, arg[3].begin, arg[3].end-1);

		merge(tmp, arr, arg[0].begin, arg[1].end-1, arg[2].begin, arg[3].end-1);
*/
		//merge(tmp, arr, left, arg[2].begin-1, arg[2].begin, right);

		/*for (int i = 1; i < NB_THREADS; i++)
		{
			merge(arr, tmp, left, arg[i].begin-1, arg[i].begin, arg[i].end-1);
			memcpy(arr+left, tmp+left, (arg[i].end-left)*sizeof(int));
		}*/
		//memcpy(arr+left, tmp+left, (size)*sizeof(int));

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

int
sort(struct array * array)
{
	//simple_quicksort_ascending(array);
	mergesort(array->data, array->length);

	return 0;
}


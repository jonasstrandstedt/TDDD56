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
#include "disable.h"

#ifndef DEBUG
#define NDEBUG
#endif

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"

inline void swap(int *a, int *b)
{
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

void insertionSort(int * arr, int left, int right)
{
	int i, j;
	for (i = left + 1; i <= right; ++i)
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
		swap(&arr[left], &arr[center]);
	if (arr[right] < arr[left])
		swap(&arr[left], &arr[right]);
	if (arr[right] < arr[center])
		swap(&arr[center], &arr[right]);

	swap(&arr[center], &arr[right - 1]);
	return arr[right - 1];
}

void quicksort(int * arr, int left, int right)
{
	if (right - left <= 10)
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
			swap(&arr[l], &arr[u]);
		else
			break;
	}

	swap(&arr[l], &arr[right - 1]);

	quicksort(arr, left, l - 1);
	quicksort(arr, l + 1, right);
}

int
sort(struct array * array)
{
	quicksort(array->data, 0, array->length - 1);

	return 0;
}


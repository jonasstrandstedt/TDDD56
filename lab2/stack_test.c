/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
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

#ifndef DEBUG
#define NDEBUG
#endif

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

pthread_mutex_t *aba_locks[4];
node_t *pool[NB_THREADS];

stack_t *stack;
data_t data;

// functionsint
int test_push_safe();
int test_pop_safe();

void
test_init()
{
  // Initialize your test batch
  stack = stack_alloc();
  stack_init(stack, 0);

}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  size_t i,j;
	for(i=0; i<NB_THREADS; ++i)
		{
		pool[i] = NULL;
		for(j=0; j<MAX_PUSH_POP; ++j)
		{
		  node_t *n = (node_t*)malloc(sizeof(node_t));
		  n->next = pool[i];
		  n->data =j;
		  pool[i] = n;
		}
	}

#if MEASURE == 2

test_push_safe();

#endif
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks as now
  size_t i;
	for(i=0; i<NB_THREADS; ++i) {  
    node_t *n = pool[i];
		while(n != NULL) {
		  pool[i] = n->next;
		  free(n);
      n = pool[i];
		}
	}
}

void
test_finalize()
{
  // Destroy properly your test batch
  stack_deinit(stack);
}

void*
thread_test_push_safe(void* arg)
{
	stack_measure_arg_t* args = (stack_measure_arg_t*) arg;
	int id = args->id;
	int i;
  for(i = 0; i < MAX_PUSH_POP; ++i) {
		node_t * n = pool[id];
		pool[id] = n->next;
    stack_push(stack, n);
  }

  return NULL;
}

void*
thread_test_pop_safe(void* arg)
{
	stack_measure_arg_t* args = (stack_measure_arg_t*) arg;
	int id = args->id;
  int i;
  for(i = 0; i < MAX_PUSH_POP; ++i) {
	  node_t * n;
		stack_pop(stack, &n);
    //if(n == NULL)
     // printf("NULL(%i): %i\n", id, i);
    //else {

    n->next = pool[id];
    pool[id] = n;
   // }
  }

  return NULL;
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it  
  stack_measure_arg_t arg[NB_THREADS];  

      
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 

  size_t i;
  for (i = 0; i < NB_THREADS; i++)
  {
    arg[i].id = i;
    pthread_create(&thread[i], &attr, &thread_test_push_safe, (void*) &arg[i]);
  }

  for (i = 0; i < NB_THREADS; i++)
  {
    pthread_join(thread[i], NULL);
  }

  node_t * n = stack->head;
  size_t count = 0;
  while(n)
  {
    ++count;
    n = n->next;
  }

  //printf("tja %ti \n", count);


  return count == NB_THREADS * MAX_PUSH_POP;
}

int
test_pop_safe()
{
  // Same as the test above for parallel pop operation
  stack_measure_arg_t arg[NB_THREADS];  
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 

  size_t i;
  for (i = 0; i < NB_THREADS; i++)
  {
    arg[i].id = i;
    pthread_create(&thread[i], &attr, &thread_test_pop_safe, (void*) &arg[i]);
  }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  node_t * n = stack->head;
  size_t count = 0;
  while(n)
  {
    ++count;
    n = n->next;
  }

  //printf("tja %ti \n", count);


  return count == 0;
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

struct thread_test_pop_args
{
  int id;
};

typedef struct thread_test_pop_args thread_test_pop_args_t;

int thread_pop_aba(node_t **n, pthread_mutex_t * mylock, pthread_mutex_t * otherlock)
{
  node_t* old;
  node_t* next;

  int loop = 0;
  do {
    old = stack->head;
    next = old->next;
    //lock
    
    if(otherlock != NULL)
      pthread_mutex_unlock(otherlock);

    if(mylock != NULL)
      pthread_mutex_lock(mylock);
    loop = cas((size_t*)&stack->head, (size_t)old, (size_t)next) != (size_t)old;
    if(mylock != NULL)
      pthread_mutex_unlock(mylock);
    //unlock
  } while (loop);
  old->next = NULL;
  *n = old;

  return 0;
}

int thread_push_aba(node_t * n)
{
  node_t* old;
  do {
    old = stack->head;
    n->next = old;
  } while (cas((size_t*)&stack->head, (size_t)old, (size_t)n) != (size_t)old);
  return 0;
}

void*
thread_aba_0(void* args)
{
  // pop A ... Stuck on CAS
  printf("t0: POP A... AND GETTING STUCK \n");
  node_t *n;
  thread_pop_aba(&n, aba_locks[0],aba_locks[1]);
  printf("t0: FINNALY SUCCESS, POP A \n");
  printf("t0: PUSHING A \n");
  thread_push_aba(n);

  return NULL;
}

void*
thread_aba_1(void* args)
{
  // wait for t0 to TRY and pop A
  pthread_mutex_lock(aba_locks[1]);
  node_t *n = NULL;
  printf("t1: POP A... ");
  thread_pop_aba(&n, NULL, NULL);
  printf("success! \n");
  pthread_mutex_unlock(aba_locks[2]);
  pthread_mutex_lock(aba_locks[3]);

  printf("t1: PUSH A... ");
	if(n == NULL) printf("which is NULL... ");
  thread_push_aba(n);
  printf("success! \n");

  pthread_mutex_unlock(aba_locks[0]);

  // pop A, Success
  // wait for t2 to pop B
  // 

  return NULL;
}

void*
thread_aba_2(void* args)
{
  // wait for t0 to TRY and pop A
  // wait for t1 to pop A
  pthread_mutex_lock(aba_locks[2]);
  node_t *n;
  printf("t2: POP B... ");
  thread_pop_aba(&n, NULL, NULL);
  printf("success! \n");
  pthread_mutex_unlock(aba_locks[3]);
  // pop B, Success

  return NULL;
}


int
test_aba()
{
  int success, aba_detected = 0;
  // Write here a test for the ABA problem

  // push A,B,C

  printf("main: push ABC \n");
  int i;
  //for(i=0; i<3; ++i)
    //thread_push_aba(n+i);


  stack = stack_alloc();
  stack_init(stack, 3);

printf("Stack data: ");
node_t *tmp = stack->head;
while(tmp != NULL) {
	printf("%i ", tmp->data);
	tmp = tmp->next;
}
printf("\n");

  printf("main: create 4 locks \n");
  for(i=0; i<4; ++i) {
    aba_locks[i] = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(aba_locks[i], 0);
    pthread_mutex_lock(aba_locks[i]);
  }

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 

  pthread_t thread0;
  pthread_t thread1;
  pthread_t thread2;

  printf("main: spawn \n");
  pthread_create(&thread0, &attr, &thread_aba_0, NULL);
  pthread_create(&thread1, &attr, &thread_aba_1, NULL);
  pthread_create(&thread2, &attr, &thread_aba_2, NULL);

  // wait a few millisec

  // unlock 1
  //pthread_mutex_unlock(aba_locks[1]);

  printf("main: wait join \n");
  pthread_join(thread0, NULL);
  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  printf("main: join \n");

  int expected[2] = { 2, 0 };

printf("Stack data: ");
tmp = stack->head;
i = 0;
while(tmp != NULL) {
	printf("%i ", tmp->data);
	if(expected[i] != tmp->data)
	{
		printf("\nABA detected, value is %i, expected is %i \n", tmp->data, expected[i]);
		aba_detected = 1;
	}
	tmp = tmp->next;
	i++;
}
printf("\n");

  success = aba_detected;
  return success;
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
      } while (cas(args->counter, old, local) != old);
    }

  return NULL;
}

int
test_cas()
{
#if 1
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success)
    {
      printf("Got %ti, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
    }

  assert(success);

  return success;
#else
  int a, b, c, *a_p, res;
  a = 1;
  b = 2;
  c = 3;

  a_p = &a;

  printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %d\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int) res);

  res = cas((void**)&a_p, (void*)&c, (void*)&b);

  printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %X\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int)res);

  return 0;
#endif
}

// Stack performance test
#if MEASURE != 0

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;
#endif

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
  test_init();
#if MEASURE == 0

  test_run(test_cas);
  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

#else
  test_setup();
  // Run performance tests
  int i;
  stack_measure_arg_t arg[NB_THREADS];  

  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
      //(void)arg[i].id; // Makes the compiler to shut up about unused variable arg
      // Run push-based performance test based on MEASURE token
#if MEASURE == 1
      clock_gettime(CLOCK_MONOTONIC, &t_start[i]);
      // Push MAX_PUSH_POP times in parallel
			pthread_create(&thread[i], &attr, &thread_test_push_safe, (void*) &arg[i]);
      clock_gettime(CLOCK_MONOTONIC, &t_stop[i]);
#else
      // Run pop-based performance test based on MEASURE token
      clock_gettime(CLOCK_MONOTONIC, &t_start[i]);
      // Pop MAX_PUSH_POP times in parallel
			pthread_create(&thread[i], &attr, &thread_test_pop_safe, (void*) &arg[i]);
      clock_gettime(CLOCK_MONOTONIC, &t_stop[i]);
#endif
    }

  for (i = 0; i < NB_THREADS; i++)
  {
    pthread_join(thread[i], NULL);
  }

  // Wait for all threads to finish
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
      printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
          start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
          (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
          t_stop[i].tv_nsec);
    }
  test_teardown();

#endif
  test_finalize();

  return 0;
}

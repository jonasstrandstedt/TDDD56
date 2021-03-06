/*
 * stack.h
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

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

struct node
{
	struct node * next;
	int data;
};

typedef struct node node_t;

struct stack
{
  // This is a fake structure; change it to your needs
  //int change_this_member;
	node_t * head;

#if NON_BLOCKING == 0
	pthread_mutex_t stack_lock;
#endif
#if NON_BLOCKING == 1
	pthread_mutex_t stack_lock;
#endif
};

typedef struct stack stack_t;

// Pushes an element in a thread-safe manner
int stack_push(stack_t *, node_t *);
// Pops an element in a thread-safe manner
int stack_pop(stack_t *, node_t **);

stack_t * stack_alloc();
int stack_init(stack_t *stack, size_t size);
int stack_deinit(stack_t *stack);

#endif /* STACK_H */

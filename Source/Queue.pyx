# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

from libc.stdlib cimport *


cdef struct Iteration:
    Py_ssize_t begin
    Py_ssize_t end
    double previous_cost

cdef struct Queue_Node:
    Iteration* data
    Queue_Node* next
    
cdef struct Queue:
    Queue_Node* front_node
    Queue_Node* rear_node
    size_t length
    
cdef Queue* newQueue():
    cdef Queue* queue = <Queue*>malloc(sizeof(Queue))
    queue.front_node = NULL
    queue.rear_node = NULL
    queue.length = 0
    return queue
    
cdef void enqueue(Queue* queue, Iteration* data):
    cdef Queue_Node* temp = <Queue_Node*>malloc(sizeof(Queue_Node))
    temp.data = data
    temp.next = NULL
    if queue.front_node == NULL or queue.rear_node == NULL:
        queue.front_node = temp
        queue.rear_node = temp
    else:
        queue.rear_node.next = temp
        queue.rear_node = temp
    queue.length += 1
        
cdef Iteration* dequeue(Queue* queue):
    cdef Queue_Node* temp = queue.front_node
    cdef Iteration* item = temp.data
    if queue.front_node == NULL:
        print("Error in dequeue() : Queue is empty");
        exit(EXIT_FAILURE)
    if queue.length == 1:
        queue.front_node = NULL
        queue.rear_node = NULL
    else:
        queue.front_node = queue.front_node.next
    queue.length -= 1
    free(temp)
    return item

cdef bint isQueueEmpty(Queue* queue):
    return queue.length == 0

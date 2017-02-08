# -*- coding: utf-8 -*-
# distutils: language=c

cdef struct Iteration:
    Py_ssize_t begin
    Py_ssize_t end
    double previous_cost

cdef struct Queue_Node:
    void* data
    Queue_Node* next
    
cdef struct Queue:
    Queue_Node* front_node
    Queue_Node* rear_node
    size_t length
 
cdef Queue* newQueue()
cdef void enqueue(Queue* queue, void* data)
cdef void* dequeue(Queue* queue)
cdef bint isQueueEmpty(Queue* queue)

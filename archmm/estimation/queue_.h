#ifndef QUEUE__H_
#define QUEUE__H_

#include <stdio.h>
#include <stdlib.h>


struct Iteration {
    int begin;
    int end;
    double previous_cost;
};


struct QueueNode {
    void* data;
    struct QueueNode* next;
};

    
struct Queue {
    struct QueueNode* front_node;
    struct QueueNode* rear_node;
    size_t length;
};

 
struct Queue* newQueue();

void enqueue(struct Queue* queue, void* data);

void* dequeue(struct Queue* queue);

int isQueueEmpty(struct Queue* queue);


#endif // QUEUE__H_
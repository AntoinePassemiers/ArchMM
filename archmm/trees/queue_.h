#ifndef QUEUE__H_
#define QUEUE__H_

#include "utils_.h"
#include <stddef.h>

struct Queue_Node {
    void* data;
    struct Queue_Node* next;
};
    
struct Queue {
    struct Queue_Node* front_node;
    struct Queue_Node* rear_node;
    size_t length;
};

struct Queue* newQueue();

void enqueue(struct Queue* queue, void* data);

void* dequeue(struct Queue* queue);

bint isQueueEmpty(struct Queue* queue);

#endif // QUEUE__H_
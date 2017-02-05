#ifndef QUEUE_H_
#define QUEUE_H_

#include "utils_.h"


struct Queue_Node;

struct Queue;

struct Queue* newQueue();

void enqueue(struct Queue* queue, void* data);

void* dequeue(struct Queue* queue);

bint isQueueEmpty(struct Queue* queue);

#endif // QUEUE_H_
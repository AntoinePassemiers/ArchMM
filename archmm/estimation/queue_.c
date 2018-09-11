#include "queue_.h"


struct Queue* newQueue() {
    struct Queue* queue = (struct Queue*) malloc(sizeof(struct Queue));
    queue->front_node = NULL;
    queue->rear_node = NULL;
    queue->length = 0;
    return queue;
}

    
void enqueue(struct Queue* queue, void* data) {
    struct QueueNode* temp = (struct QueueNode*) malloc(sizeof(struct QueueNode));
    temp->data = data;
    temp->next = NULL;
    if ((queue->front_node == NULL) || (queue->rear_node == NULL)) {
        queue->front_node = temp;
        queue->rear_node = temp;
    } else {
        queue->rear_node->next = temp;
        queue->rear_node = temp;
    }
    queue->length++;
}

        
void* dequeue(struct Queue* queue) {
    struct QueueNode* temp = queue->front_node;
    void* item = temp->data;
    if (queue->front_node == NULL) {
        printf("Error in dequeue() : Queue is empty\n");
        exit(EXIT_FAILURE);
    }
    if (queue->length == 1) {
        queue->front_node = NULL;
        queue->rear_node = NULL;
    } else {
        queue->front_node = queue->front_node->next;
    }
    queue->length--;
    free(temp);
    return item;
}


int isQueueEmpty(struct Queue* queue) {
    return (queue->length == 0);
}

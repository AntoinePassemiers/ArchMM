#include "utils_.h"

void* ftest() {
	return (void*) PyMem_Malloc(45);
}
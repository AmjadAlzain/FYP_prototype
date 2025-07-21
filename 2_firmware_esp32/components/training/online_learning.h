// training/online_learning.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Update the prototype of a given class using the provided feature vector (256D int8)
void update_class_prototype(int class_idx, const int8_t *feature_vector);

// Reset all prototypes to initial values (from embhd_prototypes.h)
void reset_all_prototypes(void);

#ifdef __cplusplus
}
#endif

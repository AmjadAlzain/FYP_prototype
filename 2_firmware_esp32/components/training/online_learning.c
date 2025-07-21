// training/online_learning.c
#include "online_learning.h"
#include "hdc.h"            // For hdc_update_prototype(), hdc_init_prototypes()

void update_class_prototype(int class_idx, const int8_t *feature_vector) {
    hdc_update_prototype(class_idx, feature_vector);
}

void reset_all_prototypes(void)
{
    hdc_init_prototypes();
}

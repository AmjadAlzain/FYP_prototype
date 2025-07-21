# Generated Header Files

This directory will contain auto-generated C header files from trained EmbHD models.

## Files Generated After Training

When you run `train_hdc_embhd.py` or `convert_torch_hd_to_embhd.py`, the following header files will be generated:

1. `embhd_projection.h` - Contains the projection matrix to map features to HD space
2. `embhd_prototypes.h` - Contains the class prototypes for classification

## Example Content

### embhd_projection.h

```c
// EmbHD Projection Matrix (1280x10000)
#define EMBHD_IN_FEATURES 1280
#define EMBHD_OUT_FEATURES 10000

const float proj_matrix[EMBHD_IN_FEATURES][EMBHD_OUT_FEATURES] = {
    {0.123456, -0.234567, 0.345678, /* ... more values ... */},
    {0.456789, 0.567890, -0.678901, /* ... more values ... */},
    /* ... more rows ... */
};
```

### embhd_prototypes.h

```c
// EmbHD Class Prototypes (28 classes, 10000 dimensions)
#define EMBHD_NUM_CLASSES 28
#define EMBHD_VECTOR_DIM 10000

// Bipolar vectors (-1/+1)
const int8_t prototypes[EMBHD_NUM_CLASSES][EMBHD_VECTOR_DIM] = {
    {1, -1, 1, 1, -1, /* ... more values ... */},
    {-1, 1, -1, 1, 1, /* ... more values ... */},
    /* ... more class prototypes ... */
};
```

## Using These Headers

These header files are included by the ESP32 implementation of EmbHD and contain the model parameters trained on your dataset. The ESP32 code will use these parameters to initialize the model.

Do not modify these files manually, as they are auto-generated and will be overwritten when you train a new model.

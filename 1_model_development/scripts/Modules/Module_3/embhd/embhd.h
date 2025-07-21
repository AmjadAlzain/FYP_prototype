/**
 * embhd.h - Hyperdimensional Computing for ESP32-S3-EYE
 * 
 * This header file defines the core functionality of the EmbHD library,
 * a lightweight hyperdimensional computing implementation for embedded systems.
 */

#ifndef EMBHD_H
#define EMBHD_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

// Configuration
#define EMBHD_MAX_DIMENSIONS 20000
#define EMBHD_MAX_CLASSES 30
#define EMBHD_FEAT_SIZE 1280

/**
 * Vector types for different encoding schemes
 */
typedef enum {
    EMBHD_BINARY = 0,    // Binary vectors (0/1)
    EMBHD_BIPOLAR = 1,   // Bipolar vectors (-1/+1)
    EMBHD_FLOAT = 2      // Floating-point vectors
} embhd_vtype_t;

/**
 * Vector structure for HDC operations
 */
typedef struct {
    uint32_t dim;            // Actual dimensions
    embhd_vtype_t type;      // Vector type
    union {
        uint8_t *binary;     // For binary vectors (packed bits)
        int8_t *bipolar;     // For bipolar vectors
        float *float_data;   // For floating-point vectors
    } data;
} embhd_vector_t;

/**
 * Projection matrix for mapping features to HD space
 */
typedef struct {
    uint32_t in_features;    // Input dimension
    uint32_t out_features;   // Output dimension
    float *weights;          // Projection weights
} embhd_projection_t;

/**
 * Class prototype for classification
 */
typedef struct {
    embhd_vector_t *vectors;     // Prototype vectors for each class
    uint32_t num_classes;        // Number of classes
    uint32_t dim;                // Vector dimensions
    embhd_vtype_t type;          // Vector type
} embhd_prototypes_t;

/**
 * Model structure containing projection and prototypes
 */
typedef struct {
    embhd_projection_t projection;   // Projection matrix
    embhd_prototypes_t prototypes;   // Class prototypes
} embhd_model_t;

// Core functions

/**
 * Initialize a vector with specified dimensions and type
 * 
 * @param vector Pointer to vector structure to initialize
 * @param dim Dimensions of the vector
 * @param type Vector type (binary, bipolar, float)
 * @return 0 on success, negative value on error
 */
int embhd_vector_init(embhd_vector_t *vector, uint32_t dim, embhd_vtype_t type);

/**
 * Free memory allocated for a vector
 * 
 * @param vector Pointer to vector to free
 */
void embhd_vector_free(embhd_vector_t *vector);

/**
 * Initialize a projection matrix
 * 
 * @param proj Pointer to projection structure
 * @param in_features Input feature dimensions
 * @param out_features Output HD dimensions
 * @return 0 on success, negative value on error
 */
int embhd_projection_init(embhd_projection_t *proj, uint32_t in_features, uint32_t out_features);

/**
 * Free memory allocated for a projection matrix
 * 
 * @param proj Pointer to projection to free
 */
void embhd_projection_free(embhd_projection_t *proj);

/**
 * Initialize prototypes for a specified number of classes
 * 
 * @param proto Pointer to prototypes structure
 * @param num_classes Number of classes
 * @param dim Dimensions of each prototype vector
 * @param type Vector type (binary, bipolar, float)
 * @return 0 on success, negative value on error
 */
int embhd_prototypes_init(embhd_prototypes_t *proto, uint32_t num_classes, 
                          uint32_t dim, embhd_vtype_t type);

/**
 * Free memory allocated for prototypes
 * 
 * @param proto Pointer to prototypes to free
 */
void embhd_prototypes_free(embhd_prototypes_t *proto);

/**
 * Initialize a full HDC model
 * 
 * @param model Pointer to model structure
 * @param in_features Input feature dimensions
 * @param num_classes Number of classes
 * @param hd_dim HD vector dimensions
 * @param vtype Vector type
 * @return 0 on success, negative value on error
 */
int embhd_model_init(embhd_model_t *model, uint32_t in_features, 
                    uint32_t num_classes, uint32_t hd_dim, embhd_vtype_t vtype);

/**
 * Free memory allocated for a model
 * 
 * @param model Pointer to model to free
 */
void embhd_model_free(embhd_model_t *model);

// Encoding and Classification

/**
 * Encode a feature vector using projection matrix
 * 
 * @param proj Projection matrix
 * @param features Input feature vector
 * @param output Output HD vector
 */
void embhd_encode(const embhd_projection_t *proj, const float *features, 
                 embhd_vector_t *output);

/**
 * Compute similarity between a query vector and prototype
 * 
 * @param query Query vector
 * @param prototype Prototype vector
 * @return Similarity score (-1.0 to 1.0 for cosine similarity)
 */
float embhd_similarity(const embhd_vector_t *query, const embhd_vector_t *prototype);

/**
 * Classify a query vector against class prototypes
 * 
 * @param query Query vector
 * @param proto Class prototypes
 * @param scores Optional array to store similarity scores (must be pre-allocated)
 * @return Predicted class ID
 */
uint32_t embhd_classify(const embhd_vector_t *query, const embhd_prototypes_t *proto, 
                       float *scores);

/**
 * Predict class for input features using a full model
 * 
 * @param model HDC model
 * @param features Input feature vector
 * @param scores Optional array to store similarity scores (must be pre-allocated)
 * @return Predicted class ID
 */
uint32_t embhd_predict(const embhd_model_t *model, const float *features, 
                      float *scores);

// Training and Model Update

/**
 * Update a prototype vector with a new sample
 * 
 * @param proto Prototype vector to update
 * @param sample New sample vector
 * @param learning_rate Learning rate for update (0.0-1.0)
 */
void embhd_update_prototype(embhd_vector_t *proto, const embhd_vector_t *sample, 
                           float learning_rate);

/**
 * Train model on a single example
 * 
 * @param model HDC model
 * @param features Input feature vector
 * @param class_id Class ID
 * @param learning_rate Learning rate for update
 */
void embhd_train_sample(embhd_model_t *model, const float *features, 
                       uint32_t class_id, float learning_rate);

#endif /* EMBHD_H */

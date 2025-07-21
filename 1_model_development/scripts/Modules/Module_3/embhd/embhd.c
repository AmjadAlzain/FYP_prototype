#include "embhd.h"
#include <string.h>
#include <math.h>

// Helper function to allocate memory for vector data
static int allocate_vector_data(embhd_vector_t *vector) {
    switch (vector->type) {
        case EMBHD_BINARY:
            // For binary vectors, pack bits (8 values per byte)
            vector->data.binary = (uint8_t *)calloc((vector->dim + 7) / 8, sizeof(uint8_t));
            break;
        case EMBHD_BIPOLAR:
            vector->data.bipolar = (int8_t *)calloc(vector->dim, sizeof(int8_t));
            break;
        case EMBHD_FLOAT:
            vector->data.float_data = (float *)calloc(vector->dim, sizeof(float));
            break;
        default:
            return -1;
    }
    
    return (vector->data.binary != NULL) ? 0 : -2;  // Check allocation
}

int embhd_vector_init(embhd_vector_t *vector, uint32_t dim, embhd_vtype_t type) {
    if (!vector || dim == 0 || dim > EMBHD_MAX_DIMENSIONS)
        return -1;
    
    vector->dim = dim;
    vector->type = type;
    
    return allocate_vector_data(vector);
}

void embhd_vector_free(embhd_vector_t *vector) {
    if (!vector)
        return;
    
    switch (vector->type) {
        case EMBHD_BINARY:
            free(vector->data.binary);
            break;
        case EMBHD_BIPOLAR:
            free(vector->data.bipolar);
            break;
        case EMBHD_FLOAT:
            free(vector->data.float_data);
            break;
    }
    
    vector->dim = 0;
}

int embhd_projection_init(embhd_projection_t *proj, uint32_t in_features, uint32_t out_features) {
    if (!proj || in_features == 0 || out_features == 0 || out_features > EMBHD_MAX_DIMENSIONS)
        return -1;
    
    proj->in_features = in_features;
    proj->out_features = out_features;
    proj->weights = (float *)calloc(in_features * out_features, sizeof(float));
    
    return (proj->weights != NULL) ? 0 : -2;
}

void embhd_projection_free(embhd_projection_t *proj) {
    if (!proj)
        return;
    
    free(proj->weights);
    proj->in_features = 0;
    proj->out_features = 0;
    proj->weights = NULL;
}

int embhd_prototypes_init(embhd_prototypes_t *proto, uint32_t num_classes, 
                         uint32_t dim, embhd_vtype_t type) {
    if (!proto || num_classes == 0 || num_classes > EMBHD_MAX_CLASSES || 
        dim == 0 || dim > EMBHD_MAX_DIMENSIONS)
        return -1;
    
    proto->num_classes = num_classes;
    proto->dim = dim;
    proto->type = type;
    
    // Allocate array of vectors
    proto->vectors = (embhd_vector_t *)calloc(num_classes, sizeof(embhd_vector_t));
    if (!proto->vectors)
        return -2;
    
    // Initialize each vector
    for (uint32_t i = 0; i < num_classes; i++) {
        int ret = embhd_vector_init(&proto->vectors[i], dim, type);
        if (ret != 0) {
            // Clean up on failure
            for (uint32_t j = 0; j < i; j++) {
                embhd_vector_free(&proto->vectors[j]);
            }
            free(proto->vectors);
            proto->vectors = NULL;
            return ret;
        }
    }
    
    return 0;
}

void embhd_prototypes_free(embhd_prototypes_t *proto) {
    if (!proto || !proto->vectors)
        return;
    
    for (uint32_t i = 0; i < proto->num_classes; i++) {
        embhd_vector_free(&proto->vectors[i]);
    }
    
    free(proto->vectors);
    proto->vectors = NULL;
    proto->num_classes = 0;
    proto->dim = 0;
}

int embhd_model_init(embhd_model_t *model, uint32_t in_features, 
                    uint32_t num_classes, uint32_t hd_dim, embhd_vtype_t vtype) {
    if (!model || in_features == 0 || num_classes == 0 || num_classes > EMBHD_MAX_CLASSES ||
        hd_dim == 0 || hd_dim > EMBHD_MAX_DIMENSIONS)
        return -1;
    
    int ret = embhd_projection_init(&model->projection, in_features, hd_dim);
    if (ret != 0)
        return ret;
    
    ret = embhd_prototypes_init(&model->prototypes, num_classes, hd_dim, vtype);
    if (ret != 0) {
        embhd_projection_free(&model->projection);
        return ret;
    }
    
    return 0;
}

void embhd_model_free(embhd_model_t *model) {
    if (!model)
        return;
    
    embhd_projection_free(&model->projection);
    embhd_prototypes_free(&model->prototypes);
}

// Encode features to HD vector using projection matrix
void embhd_encode(const embhd_projection_t *proj, const float *features, embhd_vector_t *output) {
    if (!proj || !features || !output || !proj->weights ||
        output->dim != proj->out_features)
        return;
    
    // Temporary array for projection result
    float *tmp = (float *)calloc(output->dim, sizeof(float));
    if (!tmp)
        return;
    
    // Compute projection
    for (uint32_t i = 0; i < output->dim; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < proj->in_features; j++) {
            sum += features[j] * proj->weights[j * proj->out_features + i];
        }
        tmp[i] = sum;
    }
    
    // Quantize based on vector type
    switch (output->type) {
        case EMBHD_BINARY:
            for (uint32_t i = 0; i < output->dim; i++) {
                // Set bit if value > 0
                if (tmp[i] > 0) {
                    uint32_t byte_idx = i / 8;
                    uint8_t bit_idx = i % 8;
                    output->data.binary[byte_idx] |= (1 << bit_idx);
                }
            }
            break;
            
        case EMBHD_BIPOLAR:
            for (uint32_t i = 0; i < output->dim; i++) {
                // +1 if value > 0, -1 otherwise
                output->data.bipolar[i] = (tmp[i] > 0) ? 1 : -1;
            }
            break;
            
        case EMBHD_FLOAT:
            // Just copy the projection results
            memcpy(output->data.float_data, tmp, output->dim * sizeof(float));
            break;
    }
    
    free(tmp);
}

// Compute similarity between two vectors
float embhd_similarity(const embhd_vector_t *query, const embhd_vector_t *prototype) {
    if (!query || !prototype || query->dim != prototype->dim || 
        query->type != prototype->type)
        return -2.0f;  // Error value
    
    float similarity = 0.0f;
    
    switch (query->type) {
        case EMBHD_BINARY: {
            // Compute Hamming similarity for binary vectors
            uint32_t byte_count = (query->dim + 7) / 8;
            uint32_t matching_bits = 0;
            
            for (uint32_t i = 0; i < byte_count; i++) {
                uint8_t xor_result = query->data.binary[i] ^ prototype->data.binary[i];
                
                // Count matching bits (when XOR result is 0)
                for (uint8_t j = 0; j < 8; j++) {
                    if (!(xor_result & (1 << j)) && (i * 8 + j < query->dim)) {
                        matching_bits++;
                    }
                }
            }
            
            similarity = (float)matching_bits / query->dim;
            break;
        }
        
        case EMBHD_BIPOLAR: {
            // Compute dot product for bipolar vectors
            int32_t dot_product = 0;
            
            for (uint32_t i = 0; i < query->dim; i++) {
                dot_product += query->data.bipolar[i] * prototype->data.bipolar[i];
            }
            
            similarity = (float)dot_product / query->dim;
            break;
        }
        
        case EMBHD_FLOAT: {
            // Compute cosine similarity for float vectors
            float dot_product = 0.0f;
            float query_norm = 0.0f;
            float proto_norm = 0.0f;
            
            for (uint32_t i = 0; i < query->dim; i++) {
                dot_product += query->data.float_data[i] * prototype->data.float_data[i];
                query_norm += query->data.float_data[i] * query->data.float_data[i];
                proto_norm += prototype->data.float_data[i] * prototype->data.float_data[i];
            }
            
            query_norm = sqrtf(query_norm);
            proto_norm = sqrtf(proto_norm);
            
            if (query_norm > 0.0f && proto_norm > 0.0f) {
                similarity = dot_product / (query_norm * proto_norm);
            }
            break;
        }
    }
    
    return similarity;
}

// Classify a query vector against class prototypes
uint32_t embhd_classify(const embhd_vector_t *query, const embhd_prototypes_t *proto, float *scores) {
    if (!query || !proto || !proto->vectors || query->dim != proto->dim ||
        query->type != proto->type)
        return (uint32_t)-1;  // Error value
    
    float max_similarity = -2.0f;  // Initialize below minimum possible similarity
    uint32_t predicted_class = 0;
    
    for (uint32_t i = 0; i < proto->num_classes; i++) {
        float sim = embhd_similarity(query, &proto->vectors[i]);
        
        if (scores) {
            scores[i] = sim;
        }
        
        if (sim > max_similarity) {
            max_similarity = sim;
            predicted_class = i;
        }
    }
    
    return predicted_class;
}

// Predict class for input features using a full model
uint32_t embhd_predict(const embhd_model_t *model, const float *features, float *scores) {
    if (!model || !features)
        return (uint32_t)-1;
    
    // Allocate a temporary vector for encoding
    embhd_vector_t encoded;
    int ret = embhd_vector_init(&encoded, model->projection.out_features, model->prototypes.type);
    if (ret != 0)
        return (uint32_t)-1;
    
    // Encode the features
    embhd_encode(&model->projection, features, &encoded);
    
    // Classify
    uint32_t prediction = embhd_classify(&encoded, &model->prototypes, scores);
    
    // Clean up
    embhd_vector_free(&encoded);
    
    return prediction;
}

// Update a prototype vector with a new sample
void embhd_update_prototype(embhd_vector_t *proto, const embhd_vector_t *sample, float learning_rate) {
    if (!proto || !sample || proto->dim != sample->dim || proto->type != sample->type ||
        learning_rate < 0.0f || learning_rate > 1.0f)
        return;
    
    switch (proto->type) {
        case EMBHD_BINARY: {
            // For binary vectors, we use a probabilistic update
            uint32_t byte_count = (proto->dim + 7) / 8;
            for (uint32_t i = 0; i < byte_count; i++) {
                for (uint8_t j = 0; j < 8; j++) {
                    if (i * 8 + j >= proto->dim)
                        break;
                    
                    uint8_t sample_bit = (sample->data.binary[i] >> j) & 1;
                    uint8_t proto_bit = (proto->data.binary[i] >> j) & 1;
                    
                    // Update with probability equal to learning rate
                    if (sample_bit != proto_bit && ((float)rand() / RAND_MAX) < learning_rate) {
                        if (sample_bit) {
                            proto->data.binary[i] |= (1 << j);   // Set bit
                        } else {
                            proto->data.binary[i] &= ~(1 << j);  // Clear bit
                        }
                    }
                }
            }
            break;
        }
        
        case EMBHD_BIPOLAR: {
            // For bipolar vectors, we average with learning rate
            for (uint32_t i = 0; i < proto->dim; i++) {
                float updated = (1.0f - learning_rate) * proto->data.bipolar[i] + 
                               learning_rate * sample->data.bipolar[i];
                
                // Quantize back to -1/+1
                proto->data.bipolar[i] = (updated > 0) ? 1 : -1;
            }
            break;
        }
        
        case EMBHD_FLOAT: {
            // For float vectors, we use weighted average
            for (uint32_t i = 0; i < proto->dim; i++) {
                proto->data.float_data[i] = (1.0f - learning_rate) * proto->data.float_data[i] + 
                                           learning_rate * sample->data.float_data[i];
            }
            break;
        }
    }
}

// Train model on a single example
void embhd_train_sample(embhd_model_t *model, const float *features, uint32_t class_id, float learning_rate) {
    if (!model || !features || class_id >= model->prototypes.num_classes)
        return;
    
    // Allocate a temporary vector for encoding
    embhd_vector_t encoded;
    int ret = embhd_vector_init(&encoded, model->projection.out_features, model->prototypes.type);
    if (ret != 0)
        return;
    
    // Encode the features
    embhd_encode(&model->projection, features, &encoded);
    
    // Update the prototype for the specified class
    embhd_update_prototype(&model->prototypes.vectors[class_id], &encoded, learning_rate);
    
    // Clean up
    embhd_vector_free(&encoded);
}

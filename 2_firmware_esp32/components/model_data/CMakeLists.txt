# components/model_data/CMakeLists.txt
idf_component_register(
    SRCS "model_data.c"
    INCLUDE_DIRS "."
    EMBED_FILES detector_model.tflite feature_extractor_model.tflite
)
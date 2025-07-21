# components/inference/CMakeLists.txt
idf_component_register(
    SRCS "inference.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp-dl espressif__esp-tflite-micro model_data camera hdc
)

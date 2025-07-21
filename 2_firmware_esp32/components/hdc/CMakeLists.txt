# components/hdc/CMakeLists.txt
idf_component_register(
    SRCS "hdc.c" "embhd_weights.c"
    INCLUDE_DIRS "."
    REQUIRES camera inference
)

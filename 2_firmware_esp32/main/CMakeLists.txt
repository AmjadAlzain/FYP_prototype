# CMakeLists.txt (main)
idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES board camera hdc ui inference training
)
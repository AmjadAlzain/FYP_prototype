# components/training/CMakeLists.txt
idf_component_register(
    SRCS "online_learning.c"
    INCLUDE_DIRS "."
    REQUIRES hdc
)

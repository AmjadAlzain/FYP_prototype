# components/camera/CMakeLists.txt
idf_component_register(
    SRCS "camera.c" "camera_setup.c"
    INCLUDE_DIRS "."
    REQUIRES board espressif__esp32-camera camera_driver
)

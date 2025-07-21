# components/camera_driver/CMakeLists.txt
idf_component_register(
  SRCS         "esp_camera_driver.c"
  INCLUDE_DIRS "."
  REQUIRES     espressif__esp32-camera
)

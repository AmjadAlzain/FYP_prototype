# components/board/CMakeLists.txt
idf_component_register(
  SRCS         "esp32_s3_eye.c"
  INCLUDE_DIRS "."
  REQUIRES     espressif__esp32_s3_eye
)

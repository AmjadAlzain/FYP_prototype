# components/ui/CMakeLists.txt
idf_component_register(
  SRCS         "ui_display.c"
  INCLUDE_DIRS "."
  REQUIRES     board
  REQUIRES     camera_driver
)

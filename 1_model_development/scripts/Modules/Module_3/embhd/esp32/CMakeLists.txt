# CMakeLists.txt for EmbHD component

set(COMPONENT_SRCS 
    "embhd_esp32.c"
    "../embhd.c"
)

set(COMPONENT_ADD_INCLUDEDIRS 
    "."
    ".."
    "./headers"
)

set(COMPONENT_REQUIRES
    esp_timer
    freertos
)

register_component()

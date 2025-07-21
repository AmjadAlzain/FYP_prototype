// ui/ui_display.h
#ifndef UI_DISPLAY_H
#define UI_DISPLAY_H

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "esp_err.h"

// Register the LCD display task.
// - frame_queue: Queue from which to receive frames (camera_fb_t*) to display.
// - out_queue: (optional) Queue to forward frames after display (usually NULL since display is final).
// - return_fb: If true, return frame buffers to camera after displaying (use true when display is the last stage).
esp_err_t register_display(QueueHandle_t frame_queue, QueueHandle_t out_queue, bool return_fb);

#endif // UI_DISPLAY_H

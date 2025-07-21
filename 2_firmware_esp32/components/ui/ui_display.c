// ui/ui_display.c 
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_lcd_panel_ops.h"
#include "bsp/esp32_s3_eye.h"      // BSP (for display functions)
#include "bsp/display.h"
#include "ui_display.h"
#include "esp_camera_driver.h"

static const char *TAG = "ui_display";

// Handle to LCD panel (obtained from BSP)
static esp_lcd_panel_handle_t s_panel_handle = NULL;

typedef struct {
    QueueHandle_t in_queue;
    QueueHandle_t out_queue;
    bool          return_fb;
} display_args_t;

// Display task function
static void display_task(void *arg)
{
    // Unpack task arguments
    if (arg == NULL) {
        ESP_LOGE(TAG, "Display task received NULL arguments. Deleting task.");
        vTaskDelete(NULL);
        return;
    }
    display_args_t *task_args = (display_args_t *)arg;
    QueueHandle_t frame_queue = task_args->in_queue;
    QueueHandle_t out_queue   = task_args->out_queue;
    bool return_fb            = task_args->return_fb;

    free(task_args);

    camera_fb_t *frame = NULL;
    ESP_LOGI(TAG, "LCD display task started (return_fb=%d)", return_fb);
    while (1) {
        // Wait for a frame to display
        if (xQueueReceive(frame_queue, &frame, portMAX_DELAY) != pdTRUE) continue;
        if (!frame) {
            ESP_LOGW(TAG, "Received NULL frame pointer.");
            continue;
        }

        // Draw the frame buffer to the LCD (assuming RGB565 format matching LCD)
        // Using BSP’s panel handle and dimensions from frame
        esp_err_t ret = esp_lcd_panel_draw_bitmap(s_panel_handle, 0, 0, frame->width, frame->height, frame->buf);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "LCD draw bitmap failed: %s", esp_err_to_name(ret));
        }

        // If there's an output queue (not typical), forward the frame
        if (out_queue != NULL) {
            xQueueSend(out_queue, &frame, portMAX_DELAY);
        } else if (return_fb) {
            // No further use of frame, return it to camera buffer pool:contentReference[oaicite:30]{index=30}
            esp_camera_fb_return(frame);
        }
        // (If outQ is set and return_fb is false, the next stage should return the buffer.)
    }
}

esp_err_t register_display(QueueHandle_t frame_queue, QueueHandle_t out_queue, bool return_fb)
{
    ESP_LOGI(TAG, "Initializing LCD display...");

    // --- 1. Initialize the LCD panel using the low-level BSP functions ---    
    bsp_display_config_t display_config = {
        .max_transfer_sz = BSP_LCD_H_RES * BSP_LCD_V_RES * sizeof(uint16_t)
    };
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_err_t err = bsp_display_new(&display_config, &s_panel_handle, &io_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "bsp_display_new failed: %s", esp_err_to_name(err));
        return err;
    }

    // --- 2. Turn on the display backlight ---
    bsp_display_backlight_on();
    ESP_LOGI(TAG, "LCD panel initialized and backlight on");

    // --- 3. Create heap-allocated arguments for the task ---
    display_args_t *args = calloc(1, sizeof(display_args_t));
    if (!args) {
        ESP_LOGE(TAG, "Failed to allocate memory for display task arguments");
        return ESP_ERR_NO_MEM;
    }
    args->in_queue = frame_queue;
    args->out_queue = out_queue;
    args->return_fb = return_fb;

    // --- 4. Create the display task ---
    BaseType_t res = xTaskCreate(
        display_task, 
        "LCDDisplay",
        4 * 1024, 
        args,       // Pass pointer to heap-allocated struct
        5, 
        NULL
    );
    
    if (res != pdPASS) {
        ESP_LOGE(TAG, "Failed to create display task");
        free(args); // Free memory if task creation fails
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Display task created, ready to show frames");
    return ESP_OK;

}

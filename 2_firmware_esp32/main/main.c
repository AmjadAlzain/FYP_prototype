// main/main.c
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"      // For vTaskDelay()
#include "driver/gpio.h"
#include "esp_log.h"

// --- Custom Component Includes ---
#include "camera_setup.h"      // Your camera helpers
#include "inference.h"         // TFLite inference functions
#include "hdc.h"               // HDC model functions (replaces online_learning.h)

// Button GPIOs (adjust for your specific board)
#define BTN_CLASSIFY_PIN      0
#define BTN_UPDATE_PIN        2
#define BTN_RESET_PIN         3
#define BUTTON_PRESSED_LEVEL  0   // Assuming pull-up resistors (LOW when pressed)

static const char *TAG = "main";

// --- Global state for online learning ---
// The feature vector from the TFLite model has a dimension of HDC_INPUT_DIM.
static int8_t last_feature_vector[HDC_INPUT_DIM];
static int last_predicted_class = -1;
static bool has_last_feature = false;

void app_main(void)
{
    // --- 1. Initialize all systems ---
    ESP_LOGI(TAG, "Initializing camera...");
    camera_init(); // Assuming this function is in camera_setup.c

    ESP_LOGI(TAG, "Initializing inference pipeline...");
    inference_init(); // Initializes TFLite models

    ESP_LOGI(TAG, "Initializing HDC prototypes from flash...");
    // This is the new, correct function name.
    hdc_init_prototypes();

    // --- 2. Setup button GPIOs ---
    gpio_config_t btn_config = {
        .intr_type = GPIO_INTR_DISABLE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << BTN_CLASSIFY_PIN) | (1ULL << BTN_UPDATE_PIN) | (1ULL << BTN_RESET_PIN),
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en   = GPIO_PULLUP_ENABLE
    };
    gpio_config(&btn_config);

    ESP_LOGI(TAG, "System initialized. Awaiting button presses...");

    // --- 3. Main Application Loop ---
    while (1) {
        // --- Handle Classification Button ---
        if (gpio_get_level(BTN_CLASSIFY_PIN) == BUTTON_PRESSED_LEVEL) {
            ESP_LOGI(TAG, "[Button] Classify pressed.");
            camera_fb_t* fb = capture_frame();
            if (!fb) {
                ESP_LOGW(TAG, "Capture failed.");
                vTaskDelay(300 / portTICK_PERIOD_MS);
                continue;
            }

            bbox_t box;
            // First, detect the object in the frame
            if (inference_detect(fb, &box) == ESP_OK && box.score > 0.5f) {
                ESP_LOGI(TAG, "Detection: box [%d %d %d %d] (score %.2f)", box.x, box.y, box.w, box.h, box.score);

                // If detected, extract its features
                int8_t feature_vec[HDC_INPUT_DIM];
                if (inference_extract_features(fb, &box, feature_vec) == ESP_OK) {
                    float conf;
                    // Use the correct classification function from the HDC component
                    int pred_class = hdc_classify(feature_vec, &conf);
                    ESP_LOGI(TAG, "HDC Classified as %d (confidence %.2f)", pred_class, conf);

                    // Save the state for potential online learning
                    memcpy(last_feature_vector, feature_vec, sizeof(feature_vec));
                    last_predicted_class = pred_class;
                    has_last_feature = true;
                }
            } else {
                ESP_LOGW(TAG, "No container detected.");
                has_last_feature = false; // Invalidate last feature if nothing was detected
            }
            esp_camera_fb_return(fb);
            vTaskDelay(500 / portTICK_PERIOD_MS); // Debounce delay
        }

        // --- Handle Prototype Update Button ---
        if (gpio_get_level(BTN_UPDATE_PIN) == BUTTON_PRESSED_LEVEL) {
            ESP_LOGI(TAG, "[Button] Update prototype pressed.");
            if (has_last_feature && last_predicted_class >= 0) {
                // Call the correct update function from the HDC component
                hdc_update_prototype(last_predicted_class, last_feature_vector);
                ESP_LOGI(TAG, "Updated prototype for class %d", last_predicted_class);
            } else {
                ESP_LOGW(TAG, "No feature/class available; classify first.");
            }
            vTaskDelay(500 / portTICK_PERIOD_MS); // Debounce
        }

        // --- Handle Prototype Reset Button ---
        if (gpio_get_level(BTN_RESET_PIN) == BUTTON_PRESSED_LEVEL) {
            ESP_LOGI(TAG, "[Button] Reset prototypes pressed.");
            // Call the correct reset function
            hdc_init_prototypes();
            ESP_LOGI(TAG, "All prototypes have been reset to their initial state.");
            vTaskDelay(500 / portTICK_PERIOD_MS); // Debounce
        }

        vTaskDelay(20 / portTICK_PERIOD_MS); // Main loop delay
    }
}
#include "esp_camera.h"
#include "img_converters.h"

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// For best quality/performance over serial, capture native JPEG on the ESP32-CAM
// and stream the JPEG bytes directly (no RGB565->JPEG conversion on-device).
#define CAMERA_PIXEL_FORMAT PIXFORMAT_JPEG
#define CAMERA_FRAME_SIZE_PSRAM FRAMESIZE_VGA
#define CAMERA_FRAME_SIZE_NO_PSRAM FRAMESIZE_QVGA

// Increase if your USB‑TTL adapter supports it (must match RUYA_SERIAL_BAUD on the Pi)
#define SERIAL_BAUD_RATE 1152000


static void send_packet(uint8_t type, const uint8_t *buf, uint32_t len) {
  Serial.write(0xAA);
  Serial.write(0xBB);
  Serial.write(type);
  Serial.write((uint8_t)(len & 0xFF));
  Serial.write((uint8_t)((len >> 8) & 0xFF));
  Serial.write((uint8_t)((len >> 16) & 0xFF));
  Serial.write((uint8_t)((len >> 24) & 0xFF));
  Serial.write(buf, len);
  Serial.write(0xBB);
  Serial.write(0xAA);
}

static void send_raw_grayscale(const uint8_t *buf, size_t len) {
  send_packet(0x00, buf, (uint32_t)len);
}

static void send_jpeg_bytes(const uint8_t *buf, size_t len) {
  send_packet(0x01, buf, (uint32_t)len);
}


void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  delay(200);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = CAMERA_PIXEL_FORMAT;

  if (psramFound()) {
    config.frame_size = CAMERA_FRAME_SIZE_PSRAM;
    // Lower number = better JPEG quality (but larger frames / more bandwidth)
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = CAMERA_FRAME_SIZE_NO_PSRAM;
    config.jpeg_quality = 14;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    while (true) {
      delay(1000);
    }
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);
    s->set_gainceiling(s, (gainceiling_t)6);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_awb_gain(s, 1);
    s->set_whitebal(s, 1);
  }
}


void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    delay(10);
    return;
  }

  // If the camera is configured for JPEG, stream it directly.
  if (fb->format == PIXFORMAT_JPEG && fb->buf && fb->len > 0) {
    send_jpeg_bytes(fb->buf, fb->len);
  } else {
    // Fallback: attempt conversion to JPEG (kept for compatibility)
    uint8_t *jpg_buf = nullptr;
    size_t jpg_len = 0;
    bool ok = frame2jpg(fb, 12, &jpg_buf, &jpg_len);
    if (ok && jpg_buf && jpg_len > 0) {
      send_jpeg_bytes(jpg_buf, jpg_len);
      free(jpg_buf);
    }

    if (fb->format == PIXFORMAT_GRAYSCALE && fb->buf && fb->len > 0) {
      send_raw_grayscale(fb->buf, fb->len);
    } else if (fb->format == PIXFORMAT_RGB565 && fb->buf && fb->len > 0 && fb->width > 0 && fb->height > 0) {
      size_t gray_len = (size_t)fb->width * (size_t)fb->height;
      uint8_t *gray = (uint8_t *)malloc(gray_len);
      if (gray) {
        const uint16_t *src = (const uint16_t *)fb->buf;
        for (size_t i = 0; i < gray_len; i++) {
          uint16_t p = src[i];
          uint8_t r = (uint8_t)(((p >> 11) & 0x1F) << 3);
          uint8_t g = (uint8_t)(((p >> 5) & 0x3F) << 2);
          uint8_t b = (uint8_t)((p & 0x1F) << 3);
          gray[i] = (uint8_t)((uint16_t)r * 30 / 100 + (uint16_t)g * 59 / 100 + (uint16_t)b * 11 / 100);
        }
        send_raw_grayscale(gray, gray_len);
        free(gray);
      }
    }
  }
  esp_camera_fb_return(fb);
}

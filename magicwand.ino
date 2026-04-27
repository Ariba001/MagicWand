#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>

#include "model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ===== MODEL SETTINGS =====
#define NUM_FEATURES 6
#define WINDOW_SIZE 50
#define NUM_CLASSES 4

// ===== SCALER VALUES =====
float mean[6] = {0.07600993, 0.07404110, 0.79824937, -1.41798096, 3.86613075, -1.70864248};
float std_dev[6]  = {0.39559943, 0.37536671, 0.28198632, 35.98536594, 27.51564753, 44.55385675};

// ===== MEMORY =====
constexpr int tensor_arena_size = 30 * 1024;
uint8_t tensor_arena[tensor_arena_size];

// ===== TFLITE =====
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Static interpreter (IMPORTANT: no heap crash)
static tflite::MicroInterpreter* static_interpreter;

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  delay(4000);  // give USB time

  Serial.println("\n===== STARTING SYSTEM =====");

  if (!IMU.begin()) {
    Serial.println("IMU failed!");
    while (1);
  }

  Serial.println("IMU initialized");

  // Load model
  const tflite::Model* model = tflite::GetModel(model_quant_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  Serial.println("Model loaded");

  // Create interpreter (STATIC SAFE VERSION)
  static tflite::MicroInterpreter interpreter_instance(
    model,
    resolver,
    tensor_arena,
    tensor_arena_size,
    nullptr,
    nullptr
  );

  interpreter = &interpreter_instance;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Input type: ");
  Serial.println(input->type);

  Serial.println("System ready!");
}

// ===== LOOP =====
void loop() {

  Serial.println("\n--- NEW INFERENCE ---");

  float ax, ay, az, gx, gy, gz;

  // ===== COLLECT DATA =====
  for (int i = 0; i < WINDOW_SIZE; i++) {

    unsigned long timeout = millis();

    while (!(IMU.accelerationAvailable() && IMU.gyroscopeAvailable())) {
      if (millis() - timeout > 300) {
        Serial.println("⚠ IMU timeout");
        return;
      }
    }

    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    float raw[6] = {ax, ay, az, gx, gy, gz};

    for (int j = 0; j < NUM_FEATURES; j++) {

      float norm = (raw[j] - mean[j]) / std_dev[j];
      int index = i * NUM_FEATURES + j;

      int8_t value = (int8_t)(norm * 127);

      if (value > 127) value = 127;
      if (value < -128) value = -128;

      input->data.int8[index] = value;
    }

    delay(20);
  }

  Serial.println("Running inference...");

  // ===== INFERENCE =====
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  Serial.println("Inference done!");

  // ===== OUTPUT =====
  int max_index = 0;
  int8_t max_val = output->data.int8[0];

  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.int8[i] > max_val) {
      max_val = output->data.int8[i];
      max_index = i;
    }
  }

  Serial.print("Prediction Index: ");
  Serial.println(max_index);

  Serial.print("Prediction Label: ");

  switch (max_index) {
    case 0: Serial.println("M"); break;
    case 1: Serial.println("No Motion"); break;
    case 2: Serial.println("O"); break;
    case 3: Serial.println("Z"); break;
    default: Serial.println("Unknown"); break;
  }

  delay(1500);
}
#include <Arduino_LSM9DS1.h>
#include <math.h>
#include "w0.h"
#include "w1.h"

// ===== NORMALIZATION =====
const float MEAN[] = {0.075771995, 0.08109146, 0.79869896};
const float STD[]  = {0.39156094, 0.36468711, 0.28072602};

// ===== BIASES =====
const float b0[] = {
  0.030943288, 0.003349177, 0.041355263, 0.05637261, 0.021280725, 0.063655935,
  0.044118185, 0.06359825, -0.0026601383, 0.012998278, -0.0305035, 0.011954061,
  0.039304677, -0.013174788, 0.059308887, 0.013045808, 0.05097764, 0.016977793,
  0.07278328, 0.0016572576, 0.03899905, 0.0034804014, -0.028783524, 0.022229165,
  -0.039610147, 0.007597462, 0.07861747, 0.045866743, 0.08658187, 0.013506979,
  0.030302424, 0.07867428
};

const float b1[] = {
  0.10151993, 0.06487135, -0.01055775, 0.02359633, 0.060877483, 0.046215087,
  -0.027008025, -0.014037103, 0.013066278, 0.077154964, 0.0070495973,
  0.008500421, 0.00028560383, -0.027690493, 0.046083357, 0.024484362
};

// Output layer
const float w2[] = {
  0.24008538, -0.3319724, -0.34063783, -0.21521151,
  -0.09138585, -0.62576854, -0.39491287, -0.2488658,
  -0.23940007, -0.5976155, -0.11549375, 0.40289506,
  -0.6050334, 0.57700735, 0.48665848, 0.2587853,
  -0.15773234, 0.059161566, -0.27595282, -0.53316605,
  -0.5358645, -0.27824026, 0.49990347, -0.16514334,
  0.11484978, 0.1132206, 0.42367655, -0.123623095,
  -0.14247777, -0.0191443, 0.41988158, -0.39791945,
  -0.09458389, 0.080970734, 0.04068398, -0.56566507,
  0.43118194, 0.5598252, -0.024716968, 0.48582527,
  -0.54510206, 0.5135791, 0.27965495, -0.32480353,
  0.26764995, -0.5912483, 0.5059, -0.04567922,
  -0.45645484, -0.18337975, -0.43020228, 0.505256,
  -0.5315708, -0.33503166, -0.45463645, 0.13774118,
  -0.014482715, 0.5121923, -0.31675676, -0.46443132,
  -0.24126343, 0.4589055, -0.083082125, 0.1454498
};

const float b2[] = {0.08821, 0.018943192, -0.067672394, -0.057669416};

// ===== BUFFERS =====
float input_data[300];
float l1[32];
float l2[16];
float out[4];

const char* labels[] = {"NO", "M", "Z", "O"};

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("IMU failed!");
    while (1);
  }

  Serial.println("System Ready");
}

void loop() {

  // ===== COLLECT DATA =====
  for (int i = 0; i < 100; i++) {
    float x, y, z;

    while (!IMU.accelerationAvailable());
    IMU.readAcceleration(x, y, z);

    input_data[i*3+0] = (x - MEAN[0]) / STD[0];
    input_data[i*3+1] = (y - MEAN[1]) / STD[1];
    input_data[i*3+2] = (z - MEAN[2]) / STD[2];

    delay(20);
  }

  run_inference();

  // ===== DEBUG OUTPUT (optional) =====
  /*
  for (int i = 0; i < 4; i++) {
    Serial.print(out[i]); Serial.print(" ");
  }
  Serial.println();
  */

  // ===== PREDICTION =====
  int pred = 0;
  float max_val = out[0];

  for (int i = 1; i < 4; i++) {
    if (out[i] > max_val) {
      max_val = out[i];
      pred = i;
    }
  }

  Serial.print("Prediction: ");
  Serial.println(labels[pred]);

  delay(1000);
}

void run_inference() {

  for (int i = 0; i < 32; i++) {
    float sum = b0[i];
    for (int j = 0; j < 300; j++) {
      sum += input_data[j] * w0[j * 32 + i];
    }
    l1[i] = (sum > 0) ? sum : 0;
  }

  for (int i = 0; i < 16; i++) {
    float sum = b1[i];
    for (int j = 0; j < 32; j++) {
      sum += l1[j] * w1[j * 16 + i];
    }
    l2[i] = (sum > 0) ? sum : 0;
  }

  for (int i = 0; i < 4; i++) {
    float sum = b2[i];
    for (int j = 0; j < 16; j++) {
      sum += l2[j] * w2[j * 4 + i];
    }
    out[i] = sum;
  }

  float max_val = out[0];
  for (int i = 1; i < 4; i++) {
    if (out[i] > max_val) max_val = out[i];
  }

  float sum_exp = 0;
  for (int i = 0; i < 4; i++) {
    out[i] = exp(out[i] - max_val);
    sum_exp += out[i];
  }

  for (int i = 0; i < 4; i++) {
    out[i] /= sum_exp;
  }
}
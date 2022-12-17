# include <Arduino_LSM9DS1.h> 
# include <TensorFlowLite.h> 
# include <tensorflow/lite/micro/all_ops_resolver.h> 
# include <tensorflow/lite/micro/micro_error_reporter.h> 
# include <tensorflow/lite/micro/micro_interpreter.h> 
# include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
# include <tensorflow/lite/schema/schema_generated.h> 
# include <tensorflow/lite/version.h> 
# include "model.h" 

// const float accelerationThreshold = 1.0; // threshold of significant in G's 
const int numSamples = 1000; 
int samplesRead = numSamples; 

tflite::MicroErrorReporter tflErrorReporter; 

tflite::AllOpsResolver tflOpsResolver; 
// tflite::MicroMutableOpResolver<4> micro_op_resolver;
// micro_op_resolver.AddDepthwiseConv2D();
// micro_op_resolver.AddFullyConnected();
// micro_op_resolver.AddReshape();
// micro_op_resolver.AddSoftmax();

const tflite::Model* tflModel = nullptr; 

tflite::MicroInterpreter* tflInterpreter = nullptr; 
TfLiteTensor* tflInputTensor = nullptr; 
TfLiteTensor* tflOutputTensor = nullptr; 
// Create a static memory buffer for TFLM, the size may need to 
// be adjusted based on the model you are using 

constexpr int tensorArenaSize = 128 * 1024; 
uint8_t tensorArena[tensorArenaSize];

// array to map action index to a name 
const char* ACTIONS[] = {"walking", "walking_up", "walking_down", "sitting", "standing", "laying"};
const float MEAN[6] = {-0.86362667, 0.00377767, -0.18852233, -0.793974, 1.78418783, 0.56001583};
const float STD[6] = {0.27513757, 0.18310183, 0.41220386, 26.54400976, 21.2698579, 14.86892491};

# define NUM_ACTIONS (sizeof(ACTIONS) / sizeof(ACTIONS[0])) 

void setup() { 
  Serial.begin(9600); 
  while (!Serial); 
  // initialize the IMU 
  if (!IMU.begin()) { 
    Serial.println("Failed to initialize IMU!"); 
    while (1); 
  } 
  // print out the samples rates of the IMUs 
  Serial.print("Accelerometer sample rate = "); 
  Serial.print(IMU.accelerationSampleRate()); 
  Serial.println(" Hz"); 
  Serial.print("Gyroscope sample rate = "); 
  Serial.print(IMU.gyroscopeSampleRate()); 
  Serial.println(" Hz"); 
  Serial.println(); 

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model); 
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) { 
    Serial.println("Model schema mismatch!"); 
    while (1); 
  } 

  // Create an interpreter to run the model 
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  // Allocate memory for the model's input and output tensors 
  tflInterpreter->AllocateTensors(); 
  // Get pointers for the model's input and output tensors 
  tflInputTensor = tflInterpreter->input(0); 
  tflOutputTensor = tflInterpreter->output(0); 
} 

void loop() 
  { 
  
    float aX, aY, aZ, gX, gY, gZ; 

    if (samplesRead == numSamples) 
      {samplesRead = 0;} 

    while (samplesRead < numSamples) { 

      // check if new acceleration AND gyroscope data is available 
      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) 
        { 
          // read the acceleration and gyroscope data 
          IMU.readAcceleration(aX, aY, aZ); 
          IMU.readGyroscope(gX, gY, gZ); 

          if (samplesRead % 2 == 0)
            {
              // log sensor data
              // Serial.print("aX:"); 
              // Serial.print(aX, 3); 
              // Serial.print(","); 
              // Serial.print("aY:"); 
              // Serial.print(aY, 3); 
              // Serial.print(","); 
              // Serial.print("aZ:"); 
              // Serial.print(aZ, 3); 
              // Serial.print(","); 
              // Serial.print("gX:"); 
              // Serial.print(gX, 3); 
              // Serial.print(","); 
              // Serial.print("gY:"); 
              // Serial.print(gY, 3); 
              // Serial.print(","); 
              // Serial.print("gZ:"); 
              // Serial.print(gZ, 3); 
              // Serial.println(); 

              float a1 = (aX - MEAN[0]) / STD[0];
              float a2 = (aY - MEAN[1]) / STD[1];
              float a3 = (aZ - MEAN[2]) / STD[2];
              float a4 = (gX - MEAN[3]) / STD[3]; 
              float a5 = (gY - MEAN[4]) / STD[4]; 
              float a6 = (gZ - MEAN[5]) / STD[5]; 

              tflInputTensor->data.f[samplesRead*3 + 0] = a1 * -0.21581484 + a2 * 0.14478123 + a3 * -0.5035822 + a4 * 0.03027846 + a5 * -0.26697952 + a6 * 0.6101277 + 0.00857501;
              tflInputTensor->data.f[samplesRead*3 + 1] = a1 * 0.4697805 + a2 * -0.37991464 + a3 * -0.18120071 + a4 * 0.55594224 + a5 * -0.5761307 + a6 * -0.6913353 + 0.08086778;
              tflInputTensor->data.f[samplesRead*3 + 2] = a1 * -0.7174115 + a2 * 0.33274716 + a3 * -0.36987996 + a4 * 0.57858384 + a5 * -0.49397665 + a6 * 0.33950216 + 00.2608107;
              tflInputTensor->data.f[samplesRead*3 + 3] = a1 * 0.15214415 + a2 * 0.01606859 + a3 * -0.06454189 + a4 * -0.17820507 + a5 * 0.23984799 + a6 * 0.7703384 + 0.13189144;
              tflInputTensor->data.f[samplesRead*3 + 4] = a1 * 0.5649091 + a2 * 0.28777984 + a3 * 0.15741514 + a4 * -0.1533395 + a5 * -0.3954151 + a6 * -0.51113415 + 0.05366264;
              tflInputTensor->data.f[samplesRead*3 + 5] = a1 * 0.50473434 + a2 * -0.3698721 + a3 * -0.69464326 + a4 * 0.3219829 + a5 * 0.44500217 + a6 * 0.35960424 + -0.01997638;
            }
          samplesRead++; 

          if (samplesRead == numSamples) 
            { 
              // Run inferencing 

              // test with pre recorded data
              // for (size_t ii = 0; ii < (tflInputTensor->bytes / sizeof(float)); ++ii)
              //   {
              //     tflInputTensor->data.f[ii] = walking[ii];
              //   }

              tflOutputTensor = tflInterpreter->output(0); 
              TfLiteStatus invokeStatus = tflInterpreter->Invoke(); 

              if (invokeStatus != kTfLiteOk) 
                { 
                  Serial.println("Invoke failed!"); 
                  while (1); 
                  return; 
                } 

              // Loop through the output tensor values from the model 
              // Serial.print("walking: "); 
              // Serial.println(tflOutputTensor->data.f[0]+tflOutputTensor->data.f[1]+tflOutputTensor->data.f[2], 3); 
              // for (int i = 3; i < NUM_ACTIONS; i++) 
              //   { 
              //     Serial.print(ACTIONS[i]); 
              //     Serial.print(": "); 
              //     Serial.println(tflOutputTensor->data.f[i], 3); 
              //   }           
              // Serial.println(); 
              float scores[] = {tflOutputTensor->data.f[0]+tflOutputTensor->data.f[1]+tflOutputTensor->data.f[2], tflOutputTensor->data.f[3], tflOutputTensor->data.f[4], tflOutputTensor->data.f[5]};
              float MAX = -1;
              int ind = -1;
              for (int ii = 0; ii < 4; ii++)
                {
                  if (scores[ii] > MAX)
                    {
                      MAX = scores[ii];
                      ind = ii;
                    }
                }
              if (ind == 0) Serial.print("Walking "); 
              else if (ind == 1) Serial.print("Sitting "); 
              else if (ind == 2) Serial.print("Standing "); 
              else if (ind == 3) Serial.print("Laying "); 
              Serial.print(MAX, 3); 
              Serial.println(); 
            } 
        } 
    } 
  }
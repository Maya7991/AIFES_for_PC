/*
  www.aifes.ai
  https://github.com/Fraunhofer-IMS/AIfES_for_Arduino
  Copyright (C) 2020-2021 Fraunhofer Institute for Microelectronic Circuits and Systems. All rights reserved.

  AIfES is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.

  AIfES XOR training demo
  --------------------

    Versions:
    1.0.0   Initial version

  The sketch shows an example of how a neural network is trained from scratch in AIfES using training data.
  As in the example "0_XOR_Inference", an XOR gate is mapped here using a neural network.
  The 4 different states of an XOR gate are fed in as training data here.
  The network structure is 2-3(Sigmoid)-1(Sigmoid) and Sigmoid is used as activation function.
  In the example, the weights are initialized randomly in a range of values from -2 to +2. The Gotrot initialization was inserted as an alternative and commented out.
  For the training the ADAM Optimizer is used, the SGD Optimizer was commented out.
  The optimizer performs a batch training over 100 epochs.
  The calculation is done in float 32.

  XOR truth table / training data
  Input    Output
  0   0    0
  0   1    1
  1   0    1
  1   1    0
  */

#include <stdio.h>
#include <stdlib.h>
// #include <windows.h>
#include <time.h>
#include <unistd.h>

#include "aifes.h"

#define INPUTS  2
#define NEURONS 3
#define OUTPUTS 1

//For AIfES Express
#define DATASETS        4
#define FNN_3_LAYERS    3
#define PRINT_INTERVAL  10
uint32_t global_epoch_counter = 0;


void AIFES_inf_only()
{
    printf("AIfES Demo:\n\n");

    uint32_t i;

    float input_data[2][2] = {{0.0f, 1.0f},{1.0f, 1.0f}};                                        // Input data for the XOR ANN (0.0 / 1.0) 
    uint16_t input_shape[] = {2, 2};                                          // Definition of the input shape
    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);       // Creation of the input AIfES tensor with two dimensions and data type F32 (float32)

    // ---------------------------------- Layer definition ---------------------------------------

    // Input layer
    // uint16_t input_layer_shape[] = {1, INPUTS};          // Definition of the input layer shape (Must fit to the input tensor)
    uint16_t input_layer_shape[] = {1, INPUTS}; 

    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M( /*input dimension=*/ 2, /*input shape=*/ input_layer_shape);   // Creation of the AIfES input layer
    
    // Hidden dense layer
    float weights_data_dense_1[] = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};  // Hidden layer weights 
    float bias_data_dense_1[] = {0.0f,  0.0f, 0.0f};                                 // Hidden layer bias weights 
    ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_M( /*neurons=*/ 3, /*weights=*/ weights_data_dense_1, /*bias=*/ bias_data_dense_1); // Creation of the AIfES hidden dense layer with 3 neurons

    // ailayer_layer_norm_f32_t  layer_norm_1   = AILAYER_LAYER_NORM_F32_A( /*eps=*/ 1e-6f); // layer normalization
    float means_data[2] = {0.0f, 0.0f};
    float variances_data[2] = {1.0f, 1.0f};
    float beta[3] = {0.0f, 0.0f, 0.0f}; // to be removed
    float gamma[3] = {1.0f, 1.0f, 1.0f}; // to be removed
    ailayer_layer_norm_f32_t  layer_norm_1   = AILAYER_LAYER_NORM_F32_M( /*eps=*/ 1e-6f, /*beta=*/ beta, /*gamma=*/gamma, /*means=*/means_data, /*variances=*/variances_data );

    ailayer_l2norm_f32_t l2norm_1 = AILAYER_L2NORM_F32_A();

    // Hidden layer activation function
    ailayer_sigmoid_f32_t sigmoid_layer_1 = AILAYER_SIGMOID_F32_M();

    // Output dense layer
    float weights_data_dense_2[] = {12.0305f, -6.5858f, 11.9371f};  // Output dense layer weights
    float bias_data_dense_2[] = {-5.4247f};                         //Output dense layer bias weights
    ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_M( /*neurons=*/ 1, /*weights=*/ weights_data_dense_2, /*bias=*/ bias_data_dense_2); // Creation of the AIfES output dense layer with 1 neuron
      
    // Output layer activation function
      ailayer_sigmoid_f32_t sigmoid_layer_2 = AILAYER_SIGMOID_F32_M();

    // --------------------------- Define the structure of the model ----------------------------

    aimodel_t model;  // AIfES model
    ailayer_t *x;     // Layer object from AIfES, contains the layers

    // Passing the layers to the AIfES model
    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    // x = ailayer_layer_norm_f32_default(&layer_norm_1, x);
    x = ailayer_l2norm_f32_default(&l2norm_1, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);
    model.output_layer = x;

    aialgo_compile_model(&model); // Compile the AIfES model


    // ------------------------------------- Print the model structure ------------------------------------
    
    printf("-------------- Model structure ---------------\n");
    aialgo_print_model_structure(&model);
    printf("----------------------------------------------\n");

    // -------------------------------- Allocate and schedule the working memory for inference ---------
    
    // Allocate memory for result and temporal data
    uint32_t memory_size = aialgo_sizeof_inference_memory(&model);
    printf("Required memory for intermediate results: ");
    printf("%d",memory_size);
    printf(" bytes\n");

    void *memory_ptr = malloc(memory_size);
    
    // printf("checkpoiint 1 \n");

    // Schedule the memory over the model
    aialgo_schedule_inference_memory(&model, memory_ptr, memory_size);

    // printf("checkpoiint 2 \n");
   // ------------------------------------- Run the inference ------------------------------------

    // Create an empty output tensor for the inference result
    uint16_t output_shape[2] = {2, 1};
    float output_data[2*1];                 // Empty data array of size output_shape
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);
    
    aialgo_inference_model(&model, &input_tensor, &output_tensor); // Inference / forward pass
    
    // printf("checkpoiint 3 \n");
    // ------------------------------------- Print result ------------------------------------

    printf("\n");
    printf("Results:\n");
    printf("input 1:\tinput 2:\treal output:\tcalculated output:\n");
    printf("%f",input_data[0][0]);
    printf("\t");
    printf("%f",input_data[0][1]);
    printf("\t");
    printf("1.0");
    printf("\t\t");
    printf("%f\n", output_data[0]);

    free(memory_ptr);
}

void AIfES_training()
{
    printf("AIfES Demo:\n\n");

    uint32_t i;

    // Tensor for the training data
    // Corresponds to the XOR truth table
    float input_data[] = {0.0f, 0.0f,
                0.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, 1.0f};
    uint16_t input_shape[] = {4, INPUTS};    // Definition of the input shape
    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data); // Creation of the input AIfES tensor with two dimensions and data type F32 (float32)

    // Tensor for the target data
    // Corresponds to the XOR truth table
    float target_data[] = {0.0f,
              1.0f,
              1.0f,
              0.0f};
    uint16_t target_shape[] = {4, OUTPUTS};     // Definition of the output shape
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data); // Assign the target_data array to the tensor. It expects a pointer to the array where the data is stored

    // Tensor for the output data (result after training).
    // Same configuration as for the target tensor
    // Corresponds to the XOR truth table
    float output_data[4];
    uint16_t output_shape[] = {4, OUTPUTS};
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    // ---------------------------------- Layer definition ---------------------------------------

    // Input layer
    uint16_t input_layer_shape[] = {1, INPUTS};          // Definition of the input layer shape (Must fit to the input tensor)

    float means_data[2] = {0.0f, 0.0f}; // to be removed
    float variances_data[2] = {1.0f, 1.0f}; // to be removed
    float beta[3] = {0.0f, 0.0f, 0.0f}; // to be removed
    float gamma[3] = {1.0f, 1.0f, 1.0f}; // to be removed

    ailayer_input_f32_t   input_layer     = AILAYER_INPUT_F32_A( /*input dimension=*/ 2, /*input shape=*/ input_layer_shape);   // Creation of the AIfES input layer
    ailayer_dense_f32_t   dense_layer_1   = AILAYER_DENSE_F32_A( /*neurons=*/ 3); // Creation of the AIfES hidden dense layer with 3 neurons
    ailayer_layer_norm_f32_t  layer_norm_1   = AILAYER_LAYER_NORM_F32_M( /*eps=*/ 1e-5f, /*beta=*/ beta, /*gamma=*/gamma, /*means=*/means_data, /*variances=*/variances_data );
    // ailayer_layer_norm_f32_t  layer_norm_1   = AILAYER_LAYER_NORM_F32_A( /*eps=*/ 1e-6f); // gives segmentation fault
    ailayer_sigmoid_f32_t sigmoid_layer_1 = AILAYER_SIGMOID_F32_A(); // Hidden activation function
    ailayer_dense_f32_t   dense_layer_2   = AILAYER_DENSE_F32_A( /*neurons=*/ 1); // Creation of the AIfES output dense layer with 1 neuron
    ailayer_sigmoid_f32_t sigmoid_layer_2 = AILAYER_SIGMOID_F32_A(); // Output activation function

    ailoss_mse_t mse_loss;                          //Loss: mean squared error

    // --------------------------- Define the structure of the model ----------------------------

    aimodel_t model;  // AIfES model
    ailayer_t *x;     // Layer object from AIfES, contains the layers

    // Passing the layers to the AIfES model
    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_layer_norm_f32_default(&layer_norm_1, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);
    model.output_layer = x;

    // Add the loss to the AIfES model
    model.loss = ailoss_mse_f32_default(&mse_loss, model.output_layer);

    aialgo_compile_model(&model); // Compile the AIfES model


    // ------------------------------- Allocate memory for the parameters of the model ------------------------------
    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
    printf("Required memory for parameter (Weights, Bias, ...):");
    printf("%d",parameter_memory_size);
    printf("Byte\n");

    void *parameter_memory = malloc(parameter_memory_size);

    // Distribute the memory to the trainable parameters of the model
    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

    // ------------------------------------- Print the model structure ------------------------------------

    printf("-------------- Model structure ---------------\n");
    aialgo_print_model_structure(&model);
    printf("----------------------------------------------\n");


    // ------------------------------- Initialize the parameters ------------------------------

    // Random weights in the value range from -2 to +2
    // The value range of the weights was chosen large, so that learning success is not always given ;)
    float max = 2.0;
    float min = -2.0;
    aimath_f32_default_tensor_init_uniform(&dense_layer_1.weights,max,min);
    aimath_f32_default_tensor_init_uniform(&dense_layer_1.bias,max,min);
    aimath_f32_default_tensor_init_uniform(&dense_layer_2.weights,max,min);
    aimath_f32_default_tensor_init_uniform(&dense_layer_2.bias,max,min);

    // -------------------------------- Define the optimizer for training ---------------------

    aiopti_t *optimizer; // Object for the optimizer

    //ADAM optimizer
    aiopti_adam_f32_t adam_opti;
    adam_opti.learning_rate = 0.1f;
    adam_opti.beta1 = 0.9f;
    adam_opti.beta2 = 0.999f;
    adam_opti.eps = 1e-7;

    // Choose the optimizer
    optimizer = aiopti_adam_f32_default(&adam_opti);

    printf("checkpoiint :)\n");

    // -------------------------------- Allocate and schedule the working memory for training ---------

    uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);
    printf("Required memory for the training (Intermediate results, gradients, optimization memory): %d Byte\n", memory_size);

    void *memory_ptr = malloc(memory_size);

    // Schedule the memory over the model
    aialgo_schedule_training_memory(&model, optimizer, memory_ptr, memory_size);

    // Initialize the AIfES model
    aialgo_init_model_for_training(&model, optimizer);

    // --------------------------------- Print the result before training ----------------------------------

    uint32_t input_counter = 0;  // Counter to print the inputs/training data

    // Do the inference before training
    aialgo_inference_model(&model, &input_tensor, &output_tensor);

    printf("\n");
    printf("Before training:\n");
    printf("Results:\n");
    printf("input 1:\tinput 2:\treal output:\tcalculated output:\n");

    for (i = 0; i < 4; i++) {
    printf("%f",input_data[input_counter]);
    //Serial.print(((float* ) input_tensor.data)[i]); //Alternative print for the tensor
    input_counter++;
    printf("\t");
    printf("%f",input_data[input_counter]);
    input_counter++;
    printf("\t");
    printf("%f",target_data[i]);
    printf("\t");
    printf("%f\n",output_data[i]);
    //Serial.println(((float* ) output_tensor.data)[i]); //Alternative print for the tensor
    }

    // ------------------------------------- Print weights init ------------------------------------

    printf("float weights_data_dense_1[] = {\n");

    for (i = 0; i < INPUTS * NEURONS; i++) {

        if(i == INPUTS * NEURONS - 1)
        {
            printf("%ff\n",((float *) dense_layer_1.weights.data)[i]);
        }
        else
        {
            printf("%ff,\n",((float *) dense_layer_1.weights.data)[i]);
        }

    }
    printf("};\n\n");

     // ------------------------------------- Run the training ------------------------------------

    float loss;
    uint32_t batch_size = 4; // Configuration tip: ADAM=4   / SGD=1
    uint16_t epochs = 250;   // Configuration tip: ADAM=100 / SGD=550
    uint16_t print_interval = 10;

    printf("\n");
    printf("Start training\n");
    for(i = 0; i < epochs; i++)
    {
    // One epoch of training. Iterates through the whole data once
    aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);

    // Calculate and print loss every print_interval epochs
    if(i % print_interval == 0)
    {
      aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
      printf("Epoch: ");
      printf("%d",i);
      printf(" Loss: ");
      printf("%f\n",loss);

    }
    }
    printf("Finished training\n\n");

    // ----------------------------------------- Evaluate the trained model --------------------------

    // Do the inference after training
    aialgo_inference_model(&model, &input_tensor, &output_tensor);


    printf("After training:\n");
    printf("Results:\n");
    printf("input 1:\tinput 2:\treal output:\tcalculated output:\n");

    input_counter = 0;

    for (i = 0; i < 4; i++) {
    printf("%f",input_data[input_counter]);
    //Serial.print(((float* ) input_tensor.data)[i]); //Alternative print for the tensor
    input_counter++;
    printf("\t");
    printf("%f",input_data[input_counter]);
    input_counter++;
    printf("\t");
    printf("%f",target_data[i]);
    printf("\t");
    printf("%f\n",output_data[i]);
    //Serial.println(((float* ) output_tensor.data)[i]); //Alternative print for the tensor
    }

    //How to print the weights example
    //Serial.println(((float *) dense_layer_1.weights.data)[0]);
    //Serial.println(((float *) dense_layer_1.bias.data)[0]);

    if(loss > 0.3f)
    {
        printf("\n");
        printf("WARNING\n");
        printf("The loss is very high: %f\n", loss);
    }

    printf("\n");
    printf("A learning success is not guaranteed\n");
    printf("The weights were initialized randomly\n\n");
    printf("copy the weights in the (3_XOR_Inference_keras.ino) example:\n");
    printf("---------------------------------------------------------------------------------\n\n");

    // printf("float weights_data_dense_1[] = {\n");

    // for (i = 0; i < INPUTS * NEURONS; i++) {

    //     if(i == INPUTS * NEURONS - 1)
    //     {
    //         printf("%ff\n",((float *) dense_layer_1.weights.data)[i]);
    //     }
    //     else
    //     {
    //         printf("%ff,\n",((float *) dense_layer_1.weights.data)[i]);
    //     }

    // }
    // printf("};\n\n");

    // printf("float bias_data_dense_1[] = {\n");

    // for (i = 0; i < NEURONS; i++) {

    //     if(i == NEURONS - 1)
    //     {
    //         printf("%ff\n",((float *) dense_layer_1.bias.data)[i]);
    //     }
    //     else
    //     {
    //         printf("%ff,\n",((float *) dense_layer_1.bias.data)[i]);
    //     }

    // }
    // printf("};\n\n");

    // printf("-------------------------------\n\n");

    // printf("float weights_data_dense_2[] = {\n");

    // for (i = 0; i < NEURONS * OUTPUTS; i++) {

    //     if(i == NEURONS * OUTPUTS - 1)
    //     {
    //         printf("%ff\n",((float *) dense_layer_2.weights.data)[i]);
    //     }
    //     else
    //     {
    //         printf("%ff,\n",((float *) dense_layer_2.weights.data)[i]);
    //     }

    // }
    // printf("};\n\n");

    // printf("float bias_data_dense_2[] = {\n");

    // for (i = 0; i < OUTPUTS; i++) {

    //     if(i == OUTPUTS - 1)
    //     {
    //         printf("%ff\n",((float *) dense_layer_2.bias.data)[i]);
    //     }
    //     else
    //     {
    //         printf("%ff,\n",((float *) dense_layer_2.bias.data)[i]);
    //     }

    // }
    // printf("};\n\n");

    free(parameter_memory);
    free(memory_ptr);
}


int main(int argc, char *argv[])
{
  time_t t;
  // seed: 1737733307
  // rand test: 1742622728
  // The loss is very high: 0.666088
    // float weights_data_dense_1[] = {
    // -1.352165f,
    // -0.810810f,
    // 1.189798f,
    // -0.948687f,
    // 1.566705f,
    // 1.981252f
    // };

  // seed: 1737733430
  // rand test: 1921082633
  // nan but no error message

  // couldn't find the seed in console.
  // Error: shifted_x or d_xhat is NaN! shifted_x: -nan, d_xhat: -nan

  // seed: 1737733634
  // rand test: 1080454780
  // Epoch: 240 Loss: 0.258378

  // seed: 1737733749
  // rand test: 467823367
  // Segmentation fault

  // seed: 1737733822
  // rand test: 815132147
  // Error: d_var or d_mean is NaN! d_var: -nan, d_mean: 164958337945829376.000000
  // Epoch: 110 Loss: 0.555723
  // Error: d_var or d_mean is NaN! d_var: -nan, d_mean: 168310920697610240.000000
  // Epoch: 120 Loss: 0.507366
  // Error: d_var or d_mean is NaN! d_var: -nan, d_mean: 175792410129858560.000000
  // Error: d_var or d_mean is NaN! d_var: -nan, d_mean: 178434502211665920.000000

  // seed: 1737733968
  // rand test: 432731704
  // Start training
  // Segmentation fault

  // seed: 1737734201
  // rand test: 441264605
  // Epoch: 240 Loss: 0.238905

  // seed: 1737734421
  // rand test: 106583759
  // Epoch: 240 Loss: 0.001154

  //IMPORTANT
  //AIfES requires random weights for training
  srand((unsigned) time(&t));
  // srand(1737733307);
  printf("seed: %ld\n", t);


  printf("rand test: %d\n",rand());

  AIFES_inf_only();

	system("pause");

	return 0;
}

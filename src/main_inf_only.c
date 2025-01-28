// /*
//   www.aifes.ai
//   https://github.com/Fraunhofer-IMS/AIfES_for_Arduino
//   Copyright (C) 2020-2021 Fraunhofer Institute for Microelectronic Circuits and Systems. All rights reserved.

//   AIfES is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.

//   This program is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.

//   You should have received a copy of the GNU General Public License
//   along with this program.  If not, see <https://www.gnu.org/licenses/>.

//   AIfES XOR training demo
//   --------------------

//     Versions:
//     1.0.0   Initial version

//   The sketch shows an example of how a neural network is trained from scratch in AIfES using training data.
//   As in the example "0_XOR_Inference", an XOR gate is mapped here using a neural network.
//   The 4 different states of an XOR gate are fed in as training data here.
//   The network structure is 2-3(Sigmoid)-1(Sigmoid) and Sigmoid is used as activation function.
//   In the example, the weights are initialized randomly in a range of values from -2 to +2. The Gotrot initialization was inserted as an alternative and commented out.
//   For the training the ADAM Optimizer is used, the SGD Optimizer was commented out.
//   The optimizer performs a batch training over 100 epochs.
//   The calculation is done in float 32.

//   XOR truth table / training data
//   Input    Output
//   0   0    0
//   0   1    1
//   1   0    1
//   1   1    0
//   */

// #include <stdio.h>
// #include <stdlib.h>
// // #include <windows.h>
// #include <time.h>
// #include <unistd.h>

// #include "aifes.h"

// #define INPUTS  2
// #define NEURONS 3
// #define OUTPUTS 1

// //For AIfES Express
// #define DATASETS        4
// #define FNN_3_LAYERS    3
// #define PRINT_INTERVAL  10
// uint32_t global_epoch_counter = 0;



// void AIfES_demo()
// {
//     printf("AIfES Demo:\n\n");

//     uint32_t i;

//     float input_data[2][2] = {{0.0f, 1.0f},{1.0f, 1.0f}};                                        // Input data for the XOR ANN (0.0 / 1.0) 
//     uint16_t input_shape[] = {2, 2};                                          // Definition of the input shape
//     aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);       // Creation of the input AIfES tensor with two dimensions and data type F32 (float32)

//     // ---------------------------------- Layer definition ---------------------------------------

//     // Input layer
//     // uint16_t input_layer_shape[] = {1, INPUTS};          // Definition of the input layer shape (Must fit to the input tensor)
//     uint16_t input_layer_shape[] = {1, INPUTS}; 

//     ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M( /*input dimension=*/ 2, /*input shape=*/ input_layer_shape);   // Creation of the AIfES input layer
    
//     // Hidden dense layer
//     float weights_data_dense_1[] = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};  // Hidden layer weights 
//     float bias_data_dense_1[] = {0.0f,  0.0f, 0.0f};                                 // Hidden layer bias weights 
//     ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_M( /*neurons=*/ 3, /*weights=*/ weights_data_dense_1, /*bias=*/ bias_data_dense_1); // Creation of the AIfES hidden dense layer with 3 neurons

//     // ailayer_layer_norm_f32_t  layer_norm_1   = AILAYER_LAYER_NORM_F32_A( /*eps=*/ 1e-6f); // layer normalization
//     float means_data[2] = {0.0f, 0.0f};
//     float variances_data[2] = {1.0f, 1.0f};
//     float beta[3] = {0.0f, 0.0f, 0.0f}; // to be removed
//     float gamma[3] = {1.0f, 1.0f, 1.0f}; // to be removed
//     ailayer_layer_norm_f32_t  layer_norm_1   = AILAYER_LAYER_NORM_F32_M( /*eps=*/ 1e-6f, /*beta=*/ beta, /*gamma=*/gamma, /*means=*/means_data, /*variances=*/variances_data );

//     // Hidden layer activation function
//     ailayer_sigmoid_f32_t sigmoid_layer_1 = AILAYER_SIGMOID_F32_M();

//     // Output dense layer
//     float weights_data_dense_2[] = {12.0305f, -6.5858f, 11.9371f};  // Output dense layer weights
//     float bias_data_dense_2[] = {-5.4247f};                         //Output dense layer bias weights
//     ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_M( /*neurons=*/ 1, /*weights=*/ weights_data_dense_2, /*bias=*/ bias_data_dense_2); // Creation of the AIfES output dense layer with 1 neuron
      
//     // Output layer activation function
//       ailayer_sigmoid_f32_t sigmoid_layer_2 = AILAYER_SIGMOID_F32_M();

//     // --------------------------- Define the structure of the model ----------------------------

//     aimodel_t model;  // AIfES model
//     ailayer_t *x;     // Layer object from AIfES, contains the layers

//     // Passing the layers to the AIfES model
//     model.input_layer = ailayer_input_f32_default(&input_layer);
//     x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
//     x = ailayer_layer_norm_f32_default(&layer_norm_1, x);
//     x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
//     x = ailayer_dense_f32_default(&dense_layer_2, x);
//     x = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);
//     model.output_layer = x;

//     aialgo_compile_model(&model); // Compile the AIfES model


//     // ------------------------------------- Print the model structure ------------------------------------
    
//     printf("-------------- Model structure ---------------\n");
//     aialgo_print_model_structure(&model);
//     printf("----------------------------------------------\n");

//     // -------------------------------- Allocate and schedule the working memory for inference ---------
    
//     // Allocate memory for result and temporal data
//     uint32_t memory_size = aialgo_sizeof_inference_memory(&model);
//     printf("Required memory for intermediate results: ");
//     printf("%d",memory_size);
//     printf(" bytes\n");

//     void *memory_ptr = malloc(memory_size);
    
//     // printf("checkpoiint 1 \n");

//     // Schedule the memory over the model
//     aialgo_schedule_inference_memory(&model, memory_ptr, memory_size);

//     // printf("checkpoiint 2 \n");
//    // ------------------------------------- Run the inference ------------------------------------

//     // Create an empty output tensor for the inference result
//     uint16_t output_shape[2] = {2, 1};
//     float output_data[2*1];                 // Empty data array of size output_shape
//     aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);
    
//     aialgo_inference_model(&model, &input_tensor, &output_tensor); // Inference / forward pass
    
//     // printf("checkpoiint 3 \n");
//     // ------------------------------------- Print result ------------------------------------

//     printf("\n");
//     printf("Results:\n");
//     printf("input 1:\tinput 2:\treal output:\tcalculated output:\n");
//     printf("%f",input_data[0][0]);
//     printf("\t");
//     printf("%f",input_data[0][1]);
//     printf("\t");
//     printf("1.0");
//     printf("\t\t");
//     printf("%f\n", output_data[0]);

//     free(memory_ptr);
// }

// // change name
// int main_inf(int argc, char *argv[])
// {

//     time_t t;

//     //IMPORTANT
//     //AIfES requires random weights for training
//     srand((unsigned) time(&t));

//     printf("rand test: %d\n",rand());

//     AIfES_demo();

// 	system("pause");

// 	return 0;
// }

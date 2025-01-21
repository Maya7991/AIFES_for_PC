#include "basic/base/ailayer/ailayer_layer_normalization.h"
#include "basic/base/aimath/aimath_basic.h"

AISTRING_STORAGE_WRAPPER(aistring_layer_layer_norm, "Layer Normalization");

const aicore_layertype_t ailayer_layer_norm_type_s = {
#ifdef AIDEBUG_PRINT_MODULE_SPECS
    .name = aistring_layer_layer_norm,
	.print_specs = ailayer_layer_norm_print_specs
#else
    .name = 0,
    .print_specs = 0
#endif
};
const aicore_layertype_t *ailayer_layer_norm_type = &ailayer_layer_norm_type_s;

ailayer_t *ailayer_layer_norm(ailayer_layer_norm_t *layer, ailayer_t *input_layer)
{
    uint8_t channel_uaxis;
    if(layer->channel_axis < 0){
        channel_uaxis = (uint8_t) input_layer->result.dim + layer->channel_axis;
    } else {
        channel_uaxis = (uint8_t)layer->channel_axis;
    }   // need to be updated based on the shape of the input tensor

    layer->base.layer_type = ailayer_layer_norm_type;

    layer->base.settings = 0;
    // AILAYER_SETTINGS_SET(layer->base.settings, 0b1, AILAYER_SETTINGS_TRAINABLE, FALSE);
    // AILAYER_SETTINGS_SET(layer->base.settings, 0b1, AILAYER_SETTINGS_NO_INPUT_GRADIENT, FALSE);
    AILAYER_SETTINGS_SET(layer->base.settings, 0b1, AILAYER_SETTINGS_TRAINABLE, TRUE);
    AILAYER_SETTINGS_SET(layer->base.settings, 0b1, AILAYER_SETTINGS_NO_INPUT_GRADIENT, FALSE);

    layer->base.input_layer = input_layer;
    layer->base.output_layer = 0;
	input_layer->output_layer = &(layer->base);

    layer->base.layer_configuration = layer;
	layer->base.result.shape = input_layer->result.shape;
	layer->base.result.dim = input_layer->result.dim;  // don't know about this

    layer->base.deltas.dim = 2; // don't know about this
	layer->base.deltas.shape = layer->base.result.shape;

    layer->betas.dim = 1;
	layer->betas.shape = &layer->base.result.shape[channel_uaxis];

	layer->gammas.dim = 1;
	layer->gammas.shape = &layer->base.result.shape[channel_uaxis];

    // layer->means->dtype = layer->base.result.dtype;
    // layer->means.dim = 1;
    // layer->means->shape = &layer->base.result.shape[channel_uaxis];

    // aitensor_t means;
    // layer->means =&means;
    // layer->means->dtype = layer->base.result.dtype;
    // layer->means->dim = 1;
    // layer->means->shape = &layer->base.result.shape[channel_uaxis];

    // aitensor_t variances;
    // layer->variances =&variances;
    // layer->variances->dtype = layer->base.result.dtype;
    // layer->variances->dim = 1;
    // layer->variances->shape = &layer->base.result.shape[channel_uaxis];

    // layer->base.init_params(layer);

	layer->means.dim = 1;
	layer->means.shape = &layer->base.result.shape[channel_uaxis]; // is this correct if batches are present?

    layer->variances.dim = 1;
	layer->variances.shape = &layer->base.result.shape[channel_uaxis];

    layer->base.forward = ailayer_layer_norm_forward;
	layer->base.backward = ailayer_layer_norm_backward;

    //these are not zero as layer norm is trainable
    layer->base.calc_result_shape = ailayer_layer_norm_calc_result_shape;
	layer->base.sizeof_paramem = ailayer_layer_norm_sizeof_paramem;
	layer->base.set_paramem = ailayer_layer_norm_set_paramem;
	layer->base.sizeof_trainmem = ailayer_layer_norm_sizeof_trainmem;
	layer->base.set_trainmem = ailayer_layer_norm_set_trainmem;
	layer->base.sizeof_fwdmem = 0;
	layer->base.sizeof_bwdmem = 0;
    // lots of stuff here

    layer->base.trainable_params_count = 2;
	layer->base.trainable_params = layer->trainable_params;
	layer->base.gradients = layer->gradients;
	layer->base.optimem = layer->optimem;

	layer->trainable_params[0] = &layer->betas;
	layer->trainable_params[1] = &layer->gammas;

    ailayer_layer_norm_calc_result_shape(&layer->base);

    return &layer->base;
}

void ailayer_layer_norm_forward(ailayer_t *self)
{
    ailayer_layer_norm_t *layer = (ailayer_layer_norm_t *)(self->layer_configuration);
    aitensor_t *x_in = &(self->input_layer->result);
	aitensor_t *x_out = &(self->result);

    void *eps = layer->eps;
    aitensor_t *means = &(layer->means);
	aitensor_t *variances = &(layer->variances);

    // printf("mean dim %d \n", layer->means.dim);
    // printf("variances dim %d \n", layer->variances.dim);

    // aimath_f32_default_init_zeros(means);

    aitensor_t *betas = &(layer->betas);
	aitensor_t *gammas = &(layer->gammas);

    // printf("\n----------------------------------------------\n");
    layer->layer_mean(x_in, layer->channel_axis, means);
    // printf("\ncheckpoiint layer norm forward :mean done \n");
    // printf("\n----------------------------------------------\n");
    layer->layer_variance(x_in, layer->channel_axis, means, variances);
    // printf("\ncheckpoiint layer norm forward :variances done\n");
    // printf("\n----------------------------------------------\n");
    layer->layer_norm(x_in, layer->channel_axis, eps, means, variances, betas, gammas, x_out);
    // printf("checkpoiint layer norm forward :layer_norm done\n");
    // printf("\n----------------------------------------------\n");
    return;
}

void ailayer_layer_norm_backward(ailayer_t *self){

    
	ailayer_layer_norm_t *layer = (ailayer_layer_norm_t *)(self->layer_configuration);
    
    aitensor_t *delta_in = &(self->deltas);
	aitensor_t *delta_out = &(self->output_layer->deltas);
	aitensor_t *x_in = &(self->input_layer->result);
	aitensor_t *means = &(layer->means);
    aitensor_t *variances = &(layer->variances);
	aitensor_t *betas = &(layer->betas);
	aitensor_t *gammas = &(layer->gammas);
	aitensor_t *d_betas = layer->gradients[0];
	aitensor_t *d_gammas = layer->gradients[1];

    layer->d_layer_norm(x_in,
                        layer->channel_axis,
                        means,
                        variances,
                        betas,
                        gammas,
                        delta_out,
                        layer->eps,
                        delta_in,
                        d_betas,
                        d_gammas);

    return;
}

void ailayer_layer_norm_calc_result_shape(ailayer_t *self){

    // Unused: output shape of the layer normalization is directly inherited from the input shape.
    return;
}

// calculates the memory size required for storing trainable parameters in fwd pass
uint32_t ailayer_layer_norm_sizeof_paramem(const ailayer_t *self)
{
    uint32_t memory = 0;
    ailayer_layer_norm_t *layer = (ailayer_layer_norm_t *)(self->layer_configuration);

    // memory += sizeof(typeof(layer->beta));
    // AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);

    // memory += sizeof(typeof(layer->gamma));
    // AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);

    // Betas
	memory += layer->betas.dtype->tensor_params_size;
    AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);
	memory += layer->betas.shape[0] * aimath_sizeof_dtype(layer->betas.dtype); // data
    AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);

	// Gammas
	memory += layer->gammas.dtype->tensor_params_size;
    AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);
	memory += layer->gammas.shape[0] * aimath_sizeof_dtype(layer->gammas.dtype); // data
    AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);

    return memory;
}

void ailayer_layer_norm_set_paramem(ailayer_t *self, void *memory_ptr)
{
    
    uint32_t address_counter = 0;
    ailayer_layer_norm_t *layer = (ailayer_layer_norm_t *)(self->layer_configuration);

    // printf("layer->gamma: %f", layer->gamma);
    printf("\ncheckpoiint: set_paramem *******************************************§§\n");
    // Assign memory for beta
    // *((float *)(memory_ptr + address_counter)) = layer->beta; // Copy initialized value to memory
    // layer->beta = *((float *)(memory_ptr + address_counter)); // Assign back from memory (optional redundancy)
    // address_counter += sizeof(float);
    // AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);

    // // Assign memory for gamma
    // *((float *)(memory_ptr + address_counter)) = layer->gamma; // Copy initialized value to memory
    // layer->gamma = *((float *)(memory_ptr + address_counter)); // Assign back from memory (optional redundancy)
    // address_counter += sizeof(float);
    // AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);

    // memcpy(layer->betas.data, initial_beta_values, sizeof(float) * layer->betas.shape[0]);
    // memcpy(layer->gammas.data, initial_gamma_values, sizeof(float) * layer->gammas.shape[0]);

    // Betas
	layer->betas.tensor_params = memory_ptr + address_counter;
	address_counter += layer->betas.dtype->tensor_params_size;
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);

	layer->betas.data = memory_ptr + address_counter;
	address_counter += aimath_sizeof_tensor_data(&(layer->betas));
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);

	// Gammas
	layer->gammas.tensor_params = memory_ptr + address_counter;
	address_counter += layer->gammas.dtype->tensor_params_size;
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);

	layer->gammas.data = memory_ptr + address_counter;
	address_counter += aimath_sizeof_tensor_data(&(layer->gammas));
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);

    aimath_f32_default_init_zeros(&layer->betas);  // Set betas to 0
    aimath_f32_default_init_ones(&layer->gammas); // Set gammas to 1

    // Only betas and gammas are trainable
	layer->trainable_params[0] = &(layer->betas);
	layer->trainable_params[1] = &(layer->gammas);

	return;
}

uint32_t ailayer_layer_norm_sizeof_trainmem(const ailayer_t *self)
{
    uint32_t memory = 0;
    ailayer_layer_norm_t *layer = (ailayer_layer_norm_t *)(self->layer_configuration);

    // Betas gradients
	memory += sizeof(aitensor_t);
	memory += aimath_sizeof_tensor_data(&layer->betas);
    AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);
	memory += aimath_sizeof_tensor_params(&layer->betas);
    AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);

	// Gammas gradients
	memory += sizeof(aitensor_t);
	memory += aimath_sizeof_tensor_data(&layer->gammas);
    AIFES_ALIGN_INTEGER(memory, AIFES_MEMORY_ALIGNMENT);
	memory += aimath_sizeof_tensor_params(&layer->gammas);

    return memory;
}

void ailayer_layer_norm_set_trainmem(ailayer_t *self, void *memory_ptr)
{
    uint32_t address_counter = 0;
    ailayer_layer_norm_t *layer = (ailayer_layer_norm_t *)(self->layer_configuration);

    uint8_t channel_axis;
    if(layer->channel_axis < 0){
        channel_axis = (uint8_t) self->result.dim + layer->channel_axis;
    } else {
        channel_axis = (uint8_t) layer->channel_axis;
    }

    // Betas gradients in gradients[0]
	self->gradients[0] = memory_ptr + address_counter;
	address_counter += sizeof(aitensor_t);
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);
	self->gradients[0]->data = memory_ptr + address_counter;
	self->gradients[0]->dtype = layer->betas.dtype;
	self->gradients[0]->dim = 1;
	self->gradients[0]->shape = &self->result.shape[channel_axis];
	address_counter += aimath_sizeof_tensor_data(layer->gradients[0]);
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);
	self->gradients[0]->tensor_params = memory_ptr + address_counter;
	address_counter += aimath_sizeof_tensor_params(layer->gradients[0]);
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);

    // Gammas gradients in gradients[1]
	self->gradients[1] = memory_ptr + address_counter;
	address_counter += sizeof(aitensor_t);
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);
	self->gradients[1]->data = memory_ptr + address_counter;
	self->gradients[1]->dtype = layer->gammas.dtype;
	self->gradients[1]->dim = 1;
	self->gradients[1]->shape = &self->result.shape[channel_axis];
	address_counter += aimath_sizeof_tensor_data(layer->gradients[1]);
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);
	self->gradients[1]->tensor_params = memory_ptr + address_counter;
	address_counter += aimath_sizeof_tensor_params(layer->gradients[1]);
    AIFES_ALIGN_INTEGER(address_counter, AIFES_MEMORY_ALIGNMENT);

    aimath_f32_default_init_zeros(layer->gradients[0]);  // Set betas to 0
    aimath_f32_default_init_zeros(layer->gradients[1]); // Set gammas to 1

    // Means and variances tbd

    return;
}

#ifdef AIDEBUG_PRINT_MODULE_SPECS
AISTRING_STORAGE_WRAPPER(aistring_print_layer_specs_layer_norm_1, ", eps: ");
AISTRING_STORAGE_WRAPPER(aistring_print_layer_specs_layer_norm_2, ", channel_axis: ");
AISTRING_STORAGE_WRAPPER(aistring_print_layer_specs_layer_norm_3, ", beta: ");
AISTRING_STORAGE_WRAPPER(aistring_print_layer_specs_layer_norm_4, ", gamma: ");

void ailayer_layer_norm_print_specs(const ailayer_t *self)
{
    ailayer_layer_norm_t *layer = (ailayer_layer_norm_t *)(self->layer_configuration);

    AIPRINT(aistring_print_layer_specs_layer_norm_1);
    print_aiscalar(layer->eps, layer->base.result.dtype);
    AIPRINT(aistring_print_layer_specs_layer_norm_2);
    AIPRINT_LONG_INT("%ld", (long int) layer->channel_axis);

    // AIPRINT(aistring_print_layer_specs_layer_norm_3);
    // AIPRINT_FLOAT("%f", layer->beta);

    // AIPRINT(aistring_print_layer_specs_layer_norm_4);
    // AIPRINT_FLOAT("%f", layer->gamma);
}
#endif
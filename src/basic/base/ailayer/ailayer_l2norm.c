#include "basic/base/ailayer/ailayer_l2norm.h"
#include "basic/base/aimath/aimath_basic.h"

AISTRING_STORAGE_WRAPPER(aistring_l2norm, "L2 Normalization");

const aicore_layertype_t ailayer_l2norm_type_s = {
#ifdef AIDEBUG_PRINT_MODULE_SPECS
    .name = aistring_l2norm,
	.print_specs = ailayer_l2norm_print_specs
#else
    .name = 0,
    .print_specs = 0
#endif
};
const aicore_layertype_t  *ailayer_l2norm_type = &ailayer_l2norm_type_s;

ailayer_t *ailayer_l2norm(ailayer_l2norm_t *layer, ailayer_t *input_layer)
{
    layer->base.layer_type = ailayer_l2norm_type;

    layer->base.settings = 0;
    AILAYER_SETTINGS_SET(layer->base.settings, 0b1, AILAYER_SETTINGS_TRAINABLE, FALSE);
    AILAYER_SETTINGS_SET(layer->base.settings, 0b1, AILAYER_SETTINGS_NO_INPUT_GRADIENT, FALSE);

    layer->base.input_layer = input_layer;
    layer->base.output_layer = 0;
    input_layer->output_layer = &(layer->base);

    layer->base.layer_configuration = layer;
    layer->base.result.shape = input_layer->result.shape;
    layer->base.result.dim = input_layer->result.dim;

    layer->base.deltas.dim = 2;
	layer->base.deltas.shape = layer->base.result.shape;

    layer->base.forward = ailayer_l2norm_forward;
	layer->base.backward = ailayer_l2norm_backward;

	layer->base.calc_result_shape = ailayer_l2norm_calc_result_shape;
	layer->base.sizeof_paramem = 0;
	layer->base.set_paramem = 0;
	layer->base.sizeof_trainmem = 0;
	layer->base.set_trainmem = 0;
	layer->base.sizeof_fwdmem = 0;
	layer->base.sizeof_bwdmem = 0;

	layer->base.trainable_params_count = 0;

    ailayer_l2norm_calc_result_shape(&layer->base);

    return &(layer->base);

}

void ailayer_l2norm_forward(ailayer_t *self)
{
    ailayer_l2norm_t *layer = (ailayer_l2norm_t *)(self->layer_configuration);
	aitensor_t *x_in = &(self->input_layer->result);
	aitensor_t *x_out = &(self->result);

	layer->l2norm(x_in, x_out);
	return;
}

void ailayer_l2norm_backward(ailayer_t *self)
{
    ailayer_l2norm_t *layer = (ailayer_l2norm_t *)(self->layer_configuration);
	aitensor_t *delta_in = &(self->deltas);
	aitensor_t *delta_out = &(self->output_layer->deltas);
	aitensor_t *x_in = &(self->input_layer->result);

	// delta_in = delta_out .* l2norm'(x_in)
	layer->d_l2norm(x_in, delta_in);
	layer->multiply(delta_in, delta_out, delta_in);
	return;
}

void ailayer_l2norm_calc_result_shape(ailayer_t *self)
{
	// Unused: Shape is already defined (Pointer)
	return;
}

#ifdef AIDEBUG_PRINT_MODULE_SPECS
// AISTRING_STORAGE_WRAPPER(aistring_print_layer_specs_l2norm, "eps: ");

void ailayer_l2norm_print_specs(const ailayer_t *self)
{
    ailayer_l2norm_t *layer = (ailayer_l2norm_t *) self;

    // AIPRINT(aistring_print_layer_specs_l2norm);
    // layer->alpha_dtype->print_aiscalar(layer->alpha);
    return;
}
#endif


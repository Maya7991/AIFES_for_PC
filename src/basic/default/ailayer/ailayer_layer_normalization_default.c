#include "basic/default/ailayer/ailayer_layer_normalization_default.h"

ailayer_t *ailayer_layer_norm_f32_default(ailayer_layer_norm_f32_t *layer, ailayer_t *input_layer)
{
    layer->base.base.result.dtype = aif32;
	layer->base.base.deltas.dtype = aif32;
    layer->base.betas.dtype = aif32;
	layer->base.gammas.dtype = aif32;
    layer->base.means.dtype = aif32;
	layer->base.variances.dtype = aif32;

    layer->base.eps = &layer->eps;

    layer->base.base.calc_result_tensor_params = 0; // dont know
    layer->base.base.init_params = ailayer_layer_norm_init_params_f32_default;

    //forward
    layer->base.layer_mean = aimath_f32_default_mean_channelwise;
    layer->base.layer_variance = aimath_f32_default_variance_channelwise;
	layer->base.layer_norm = aimath_f32_default_layer_norm;

	// backward
	layer->base.d_layer_norm = aimath_f32_default_d_layer_norm;
	// layer->multiply = aimath_f32_default_multiply;

	return ailayer_layer_norm(&layer->base, input_layer);

}

void ailayer_layer_norm_init_params_f32_default(ailayer_t *self)
{
    ailayer_layer_norm_t *layer = (ailayer_layer_norm_t *) (self->layer_configuration);

    printf("what about me?");
    aimath_f32_default_init_zeros(&layer->betas);
    aimath_f32_default_init_ones(&layer->gammas);

    aimath_f32_default_init_zeros(&layer->means);
    aimath_f32_default_init_ones(&layer->variances);
    
	return;
}
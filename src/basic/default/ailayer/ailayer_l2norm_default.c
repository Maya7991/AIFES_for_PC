
#include "basic/default/ailayer/ailayer_l2norm_default.h"

ailayer_t *ailayer_l2norm_f32_default(ailayer_l2norm_f32_t *layer, ailayer_t *input_layer)
{
    layer->base.result.dtype = aif32;
	layer->base.deltas.dtype = aif32;

	layer->base.calc_result_tensor_params = 0;
	layer->base.init_params = 0;

	//forward
	layer->l2norm = aimath_f32_default_l2norm;

	// // backward
	layer->d_l2norm = aimath_f32_default_d_l2norm;
	// layer->multiply = aimath_f32_default_multiply;

	return ailayer_l2norm(layer, input_layer);
}


#ifndef AILAYER_LAYER_NORM_DEFAULT
#define AILAYER_LAYER_NORM_DEFAULT

#include "basic/base/ailayer/ailayer_layer_normalization.h"

#include "basic/default/aimath/aimath_f32_default.h"

#define AILAYER_LAYER_NORM_F32_M(eps, beta, gamma, means, variances)   {{{0,},0, 0, {0,0,0,0,(float *) beta}, {0,0,0,0,(float *) gamma}, {0,0,0,0,(float *) means}, {0,0,0,0,(float *) variances} }, eps}

#define AILAYER_LAYER_NORM_F32_A(eps)   {{{0,},0, 0, 0.0f, 1.0f }, eps}

typedef struct ailayer_layer_norm_f32  ailayer_layer_norm_f32_t;

struct ailayer_layer_norm_f32 {
    ailayer_layer_norm_t base; /**< Inherited field members from general layer struct. */

	aiscalar_f32_t eps; /**< Storage for ailayer_batch_norm.eps scalar in F32 */
};

ailayer_t *ailayer_layer_norm_f32_default(ailayer_layer_norm_f32_t *layer, ailayer_t *input_layer);

/** @brief \link aimath_f32.h F32 \endlink default implementation of the ailayer.init_params function for the Batch Normalization layer
 *
 * *Implementation of ailayer.init_params.*
 *
 * The function will initialize the tensors for betas with zeros and gammas with ones.
 *
 * @param *self  The layer structure
 */
void ailayer_layer_norm_init_params_f32_default(ailayer_t *self);

#endif // AILAYER_LAYER_NORM_DEFAULT
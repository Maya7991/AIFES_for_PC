#ifndef AILAYER_L2NORM_DEFAULT
#define AILAYER_L2NORM_DEFAULT

#include "basic/base/ailayer/ailayer_l2norm.h"
#include "basic/default/aimath/aimath_f32_default.h"

#define AILAYER_L2NORM_F32_M() {{0,}}
#define AILAYER_L2NORM_F32_A() {{0,}}

typedef struct ailayer_l2norm ailayer_l2norm_f32_t;
typedef struct ailayer_l2norm ailayer_l2norm_q7_t;

ailayer_t *ailayer_l2norm_f32_default(ailayer_l2norm_f32_t *layer, ailayer_t *input_layer);

ailayer_t *ailayer_l2norm_q7_default(ailayer_l2norm_q7_t *layer, ailayer_t *input_layer);

void ailayer_l2norm_calc_result_tensor_params_q7_default(ailayer_t *self);

#endif // AILAYER_L2NORM_DEFAULT
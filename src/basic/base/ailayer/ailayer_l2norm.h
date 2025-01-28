#ifndef AILAYER_L2NORM
#define AILAYER_L2NORM

#include "core/aifes_core.h"

typedef struct ailayer_l2norm ailayer_l2norm_t;

struct ailayer_l2norm{
    ailayer_t base;

    // Required configuration parameters for the layer
    void *eps;
    int8_t channel_axis;

    /** @name Math functions
	 * @brief Required data type specific math functions
	 */
	///@{
    void (*l2norm)(const aitensor_t *x, aitensor_t *result);

    void (*d_l2norm)(const aitensor_t *x, aitensor_t *result);

    void (*multiply)(const aitensor_t *a, const aitensor_t *b, aitensor_t *result);
    ///@}
};

extern const aicore_layertype_t *ailayer_l2norm_type;

ailayer_t *ailayer_l2norm(ailayer_l2norm_t *layer, ailayer_t *input_layer);

void ailayer_l2norm_forward(ailayer_t *self);

void ailayer_l2norm_backward(ailayer_t *self);

void ailayer_l2norm_calc_result_shape(ailayer_t *self);


#ifdef AIDEBUG_PRINT_MODULE_SPECS
/** @brief Print the layer specification
 *
 * @param *self The layer to print the specification for
 */
void ailayer_l2norm_print_specs(const ailayer_t *self);
#endif // AIDEBUG_PRINT_MODULE_SPECS

#endif // AILAYER_L2NORM
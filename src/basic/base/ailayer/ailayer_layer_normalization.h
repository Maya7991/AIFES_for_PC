#ifndef AILAYER_LAYER_NORM
#define AILAYER_LAYER_NORM

#include "core/aifes_core.h"

typedef struct ailayer_layer_norm  ailayer_layer_norm_t; 

struct ailayer_layer_norm{
    ailayer_t base; /**< Inherited field members from general ailayer struct. */

    // Required configuration parameters for the layer
    void *eps;
    int8_t channel_axis;

    // Trainable parameters
    // should these be tensors. They are just scalar values
    aitensor_t betas;    // refer ailayer_elu.h for datatype
    aitensor_t gammas;

    /* start: these 2 are not trainable, move down later*/
    aitensor_t means; // Vector with means of every sample (2D?)
	aitensor_t variances; // Vector with variances of every sample  
    /* end: these 2 are not trainable, move down later*/

    aitensor_t *trainable_params[2]; /**< Pointer to \f$ \beta \f$ and \f$ \gamma \f$ (which are the trainable parameters). */
	aitensor_t *gradients[2]; /**< Gradients of \f$ \beta \f$ and \f$ \gamma \f$ for the back propagation algorithm. */
	void *optimem[2]; /**< Memory field used by the trainings optimizer. */

    // // Variables for internal computation
    uint16_t parameter_shape[1];

    // aitensor_t means; // Vector with means of every sample (2D?)
	// aitensor_t variances; // Vector with variances of every sample  

    // Define math functions if any
    // mean, variance, normalize, scaling and shifting

    /** @brief Required math function: Channel-wise empirical mean calculation
	 *
	 * Requires a math function that calculates the empirical mean for each channel of the given axis:\n
     * @f[
     *  means_i = \frac{1}{m} \sum_{j=1}^{m} x_{i,j}
     * @f]
     *
     * @param x             Input tensor
     * @param channel_axis  Axis of the input tensor that stores the channel dimension.
     * @param means         Resulting mean vector (1D)
     */
    void (*layer_mean)(const aitensor_t *x, int8_t channel_axis, aitensor_t *result);

    void (*layer_variance)(const aitensor_t *x, int8_t channel_axis, const aitensor_t *means, aitensor_t *result);

    /** @brief Required math function: Channel-wise empirical mean calculation
	 *
     * @param layer_mean    Vector with means of every sample
     * @param layer_variance     Vector with the variances of every sample.
     */
    void (*layer_norm)(const aitensor_t *x, 
                        int8_t channel_axis,
                        const void *eps, 
                        const aitensor_t *means, 
                        const aitensor_t *variances, 
                        const aitensor_t *offsets,
                        const aitensor_t *scales, 
                        aitensor_t *result);

    void (*d_layer_norm)(const aitensor_t *x,
                        int8_t axis,
                        const void *eps,
                        const aitensor_t *means,
                        const aitensor_t *variances,
                        const aitensor_t *offsets,
                        const aitensor_t *scales,
                        const aitensor_t *delta_out,
                        aitensor_t *delta_in,
                        aitensor_t *d_betas,
                        aitensor_t *d_gammas);
};

extern const aicore_layertype_t *ailayer_layer_norm_type;

ailayer_t *ailayer_layer_norm(ailayer_layer_norm_t *layer, ailayer_t *input_layer);

void ailayer_layer_norm_forward(ailayer_t *self);

void ailayer_layer_norm_backward(ailayer_t *self);

void ailayer_layer_norm_calc_result_shape(ailayer_t *self);

uint32_t ailayer_layer_norm_sizeof_paramem(const ailayer_t *self);

void ailayer_layer_norm_set_paramem(ailayer_t *self, void *memory_ptr);

uint32_t ailayer_layer_norm_sizeof_trainmem(const ailayer_t *self);

void ailayer_layer_norm_set_trainmem(ailayer_t *self, void *memory_ptr);

#ifdef AIDEBUG_PRINT_MODULE_SPECS
/** @brief Print the layer specification
 *
 * @param *self     The layer to print the specification for
 */
void ailayer_layer_norm_print_specs(const ailayer_t *self);
#endif // AIDEBUG_PRINT_MODULE_SPECS

#endif  // AILAYER_LAYER_NORM
#define RADIUS         $RADIUS
#define Z_RADIUS       $Z_RADIUS
#define PATCH_SIZE     (RADIUS*2+1)*(RADIUS*2+1)*(Z_RADIUS*2+1)
#define IMAGE_WIDTH    $IMAGE_WIDTH
#define IMAGE_HEIGHT   $IMAGE_HEIGHT
#define IMAGE_DEPTH    $IMAGE_DEPTH
#define IMAGE_SIZE     IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH
#define QUANTIZE_MODE  $QUANTIZE_MODE
#define GRAY_LEVELS    $GRAY_LEVELS
#define FIXED_BINWIDTH $FIXED_BINWIDTH
#define FIXED_START    $FIXED_START
#define NBINS          $NBINS
#define GLCM_SIZE      NBINS*NBINS
#define MAXRUNLENGTH   $MAXRUNLENGTH
#define GLRLM_SIZE     NBINS*MAXRUNLENGTH
#define NDEV           $NDEV
#define DX             $DX
#define DY             $DY
#define DZ             $DZ

#include "math.h"
#include <stdio.h>

/*#include "local_feature_stats/glcm.h"*/
/*#include "local_feature_stats/glrm.h"*/

// arbitrarily small number used to alleviate log(0) errors as log(0+eps)
__device__ float eps = 1e-16;

/* FORWARD DECLARATIONS */
__device__ float mean_plugin_gpu(float *array);
__device__ float stddev_plugin_gpu(float *array, int dof=1);

/* UTILITIES */
namespace rn {
    __device__ float mean(float *array, unsigned int size) {
        // computes the mean of the array
        int total = 0;
        for (int i=0; i<size; i++) {
            total += array[i];
        }
        return ((float)total / size);
    }
    __device__ float variance(float* array, unsigned int size, unsigned int dof=1) {
        // get the mean
        float mu = rn::mean(array, size);

        // compute the std. deviation
        float stat = 0.0f;
        for (int i=0; i<size; i++) {
            stat += powf(array[i] - mu, 2);
        }
        return (stat / (size-dof));
    }
    __device__ float stddev(float* array, unsigned int size, unsigned int dof=1) {
        return sqrtf(rn::variance(array, size, dof));
    }
}

__device__ void _quantize_fixed_gpu(float *array) {
    /* bin definitions are as follows (11 bins total)
     * x < -100
     * -100 <= x < 350 ( in 25HU increments -> 9 bins )
     * 350 <= x
     */

    // rebin values
    for (int i=0; i<PATCH_SIZE; i++) {
        array[i] = fminf(NBINS-1, fmaxf(0, floorf((array[i]-FIXED_START)/(FIXED_BINWIDTH))+1));
    }
}
__device__ void _quantize_stat_gpu(float *array) {
    // quantize the array into nbins, placing lower 2.5% and upper 2.5% residuals of gaussian
    // distribution into first and last bins respectively

    // gaussian stats
    float f_mean = mean_plugin_gpu(array);
    float f_std = stddev_plugin_gpu(array, 1);
    float f_binwidth = 2*NDEV*f_std / (NBINS-2);
    //printf("mean:%f std:%f width:%f\\n", f_mean, f_std, f_binwidth);

    // rebin values
    for (int i=0; i<PATCH_SIZE; i++) {
        array[i] = fminf(NBINS-1, fmaxf(0, floorf(((array[i] - f_mean + NDEV*f_std)/(f_binwidth+1e-9)) + 1)));
    }
}
__device__ void quantize_gpu(float *array) {
    /* quantize mode selector
    */
    if (QUANTIZE_MODE == 0) {
        // NBINS defines -binwidth in HU
        _quantize_fixed_gpu(array);
    } else if (QUANTIZE_MODE == 1){
        // NBINS defines total number of bins
        _quantize_stat_gpu(array);
    }
}

/* FIRST ORDER STATISTICS */
__device__ float entropy_plugin_gpu(float *patch_array) {
    // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
    int size = PATCH_SIZE;

    quantize_gpu(patch_array);

    // count intensity occurences
    int hist[NBINS];
    for (int i=0; i<NBINS; i++) {
        hist[i] = 0;
    }
    for (int i=0; i<size; i++) {
        hist[(int)floorf(patch_array[i])]++;
    }

    // calculate local entropy
    float stat = 0.0f;
    for (int i=0; i<NBINS; i++) {
        stat -= ((float)hist[i]/size) * (logf(((float)hist[i]/size) + eps));
    }
    return stat;
}
__device__ float uniformity_plugin_gpu(float *patch_array) {
    // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
    int size = PATCH_SIZE;

    quantize_gpu(patch_array);

    // count intensity occurences
    int hist[NBINS];
    for (int i=0; i<NBINS; i++) {
        hist[i] = 0;
    }
    for (int i=0; i<size; i++) {
        hist[(int)floorf(patch_array[i])] += 1;
    }

    // calculate local uniformity
    float stat = 0.0;
    for (int i=0; i<NBINS; i++) {
        stat += powf((float)hist[i]/size, 2);
    }

    return sqrtf(stat);
}
__device__ float energy_plugin_gpu(float *patch_array) {
    // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
    int size = PATCH_SIZE;

    // calculate local energy
    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += powf((float)patch_array[i]/size, 2);
    }
    return stat;
}
__device__ float mean_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += patch_array[i];
    }
    return (stat/size);
}
__device__ float min_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    float stat = patch_array[0];
    for (int i=1; i<size; i++) {
        stat = fminf(stat, patch_array[i]) ;
    }
    return stat;
}
__device__ float max_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    float stat = patch_array[0];
    for (int i=1; i<size; i++) {
        stat = fmaxf(stat, patch_array[i]) ;
    }
    return stat;
}
__device__ float kurtosis_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    // get the mean
    float mu = mean_plugin_gpu(patch_array);
    float stddev = stddev_plugin_gpu(patch_array, 0);

    // compute the kurtosis
    float num = 0.0f;
    for (int i=0; i<size; i++) {
        num += powf(patch_array[i] - mu, 4);
    }
    return ((1.0f/size)*num) / (powf(stddev, 2));
}
__device__ float skewness_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    // get the mean
    float mu = mean_plugin_gpu(patch_array);
    float stddev = stddev_plugin_gpu(patch_array, 0);

    // compute the kurtosis
    float num = 0.0f;
    for (int i=0; i<size; i++) {
        num += powf( patch_array[i] - mu, 3);
    }
    return ((1.0f/size)*num) / (powf(stddev, 3));
}
__device__ float rms_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += powf(patch_array[i], 2);
    }
    return sqrtf(stat/size);
}
__device__ float variance_plugin_gpu(float *array, int dof=1) {
    // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
    int size = PATCH_SIZE;

    // get the mean
    float mu = mean_plugin_gpu(array);

    // compute the std. deviation
    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += powf( array[i] - mu, 2);
    }
    return (stat / (size-dof));
}
__device__ float stddev_plugin_gpu(float *array, int dof) {
    return sqrtf(variance_plugin_gpu(array, dof));
}
__device__ float range_plugin_gpu(float *array) {
    return (max_plugin_gpu(array) - min_plugin_gpu(array));
}
__device__ float meanabsdev_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;
    float mu = mean_plugin_gpu(patch_array);

    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += fabsf(patch_array[i] - mu);
    }
    return sqrtf(stat/size);
}

/* GLCM FEATURES */
namespace rn {
    inline __device__ unsigned int glcm_getY(unsigned int idx) {
        return (idx / NBINS);
    }
    inline __device__ unsigned int glcm_getX(unsigned int idx) {
        return (idx - glcm_getY(idx)*NBINS);
    }
    inline __device__ unsigned int glcm_getIndex(unsigned int x, unsigned int y) {
        return (NBINS * y + x);
    }
    __device__ float glcm_marginal_Px(float* glcm_mat, unsigned int y) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            accum += glcm_mat[ glcm_getIndex(i, y) ];
        }
        return accum;
    }
    __device__ float glcm_marginal_Py(float* glcm_mat, unsigned int x) {
        float accum = 0.0f;
        for (int j=0; j<NBINS; j++) {
            accum += glcm_mat[ glcm_getIndex(x, j) ];
        }
        return accum;
    }
    __device__ float glcm_meanP(float* glcm_mat) {
        return mean(glcm_mat, GLCM_SIZE);
    }
    __device__ float glcm_mean_Px(float* glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<GLCM_SIZE; i++) {
            int y = rn::glcm_getY(i);
            int x = rn::glcm_getX(i);
            accum += glcm_mat[i] * y;
        }
        return accum;
    }
    __device__ float glcm_mean_Py(float* glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<GLCM_SIZE; i++) {
            int y = rn::glcm_getY(i);
            int x = rn::glcm_getX(i);
            accum += glcm_mat[i] * x;
        }
        return accum;
    }
    __device__ float glcm_variance_Px(float* glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            for (int j=0; j<NBINS; j++) {
                accum += (powf(i - glcm_mean_Px(glcm_mat), 2) * glcm_mat[ glcm_getIndex(i, j) ]);
            }
        }
        return accum;
    }
    __device__ float glcm_variance_Py(float* glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            for (int j=0; j<NBINS; j++) {
                accum += (powf(j - glcm_mean_Py(glcm_mat), 2) * glcm_mat[ glcm_getIndex(i, j) ]);
            }
        }
        return accum;
    }
    __device__ float glcm_marginal_Pxplusy(float* glcm_mat, unsigned int k) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            for (int j=0; j<NBINS; j++) {
                if (i + j == k) {
                    accum += glcm_mat[ glcm_getIndex(i, j) ];
                }
            }
        }
        return accum;
    }
    __device__ float glcm_marginal_Pxminusy(float* glcm_mat, unsigned int k) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            for (int j=0; j<NBINS; j++) {
                if (fabsf(i - j) == k) {
                    accum += glcm_mat[ glcm_getIndex(i, j) ];
                }
            }
        }
        return accum;
    }
    __device__ float glcm_HX(float* norm_glcm_mat) {
        float accum = 0.0f;
        for (int y=0; y<NBINS; y++) {
            float val = glcm_marginal_Px(norm_glcm_mat, y);
            accum -= val * log2f(val+eps);
        }
        return accum;
    }
    __device__ float glcm_HY(float* norm_glcm_mat) {
        float accum = 0.0f;
        for (int x=0; x<NBINS; x++) {
            float val = glcm_marginal_Py(norm_glcm_mat, x);
            accum -= val * log2f(val+eps);
        }
        return accum;
    }
    __device__ float glcm_HXY(float* norm_glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<GLCM_SIZE; i++) {
            float val = norm_glcm_mat[i];
            accum -= val * log2f(val+eps);
        }
        return accum;
    }
    __device__ float glcm_HXY1(float* norm_glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<GLCM_SIZE; i++) {
            unsigned int y = rn::glcm_getY(i);
            unsigned int x = rn::glcm_getX(i);
            accum -= norm_glcm_mat[i] * log2f((rn::glcm_marginal_Px(norm_glcm_mat, y)*rn::glcm_marginal_Py(norm_glcm_mat, x))+eps);
        }
        return accum;
    }
    __device__ float glcm_HXY2(float* norm_glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<GLCM_SIZE; i++) {
            unsigned int y = rn::glcm_getY(i);
            unsigned int x = rn::glcm_getX(i);
            float val = rn::glcm_marginal_Px(norm_glcm_mat, y)*rn::glcm_marginal_Py(norm_glcm_mat, x);
            accum -= val * log2f(val+eps);
        }
        return accum;
    }
}
__device__ const bool glcm_symmetric = true;
__device__ void glcm_matrix_gpu(float *mat, float *array) {
    //OPTIONS

    // compute glcm matrix in the specified directions
    int dx = DX;
    int dy = DY;
    int dz = DZ;

    int patch_size_1d = (RADIUS*2+1);
    int patch_size_2d = powf(patch_size_1d, 2);
    for (int i=0; i<PATCH_SIZE; i++) {
        int z = i / patch_size_2d;
        int y = (i - patch_size_2d*z) / patch_size_1d;
        int x = i - patch_size_2d*z - patch_size_1d*y;
        int y_idx = array[i];
        int x_idx_query = (int)fmaxf(0,fminf(PATCH_SIZE-1, (z+dz)*patch_size_2d + (y+dy)*patch_size_1d + (x + dx))); // should consider boundary handling options
        int x_idx = (int)(array[x_idx_query]);

        // Logging
        //if (threadIdx.x == 50 && blockIdx.x == 20) {
        //    printf("i:%d - y:%d,x:%d - yq:%d,xq:%d - y_idx:%d,x_idx:%d\\n", i, y, x, i, x_idx_query, y_idx, x_idx);
        //}
        //if (threadIdx.x == 10 && blockIdx.x == 2) {
        //    printf("%d:%d\\n", i, y);
        //    printf("%d:%d,%d\\n", i, y_idx, x_idx);
        //}

        int mat_idx = NBINS * y_idx + x_idx;
        mat[mat_idx] += 1.0f;

        if (glcm_symmetric) {
            // fill lower triangular - opposite direction
            int mat_idx_inv = NBINS * x_idx + y_idx;
            mat[mat_idx_inv] += 1.0f;
        }

        // Logging
        //if (threadIdx.x == 50 && blockIdx.x == 20) {
        //    printf("y:%d x:%d idx:%d val:%d\\n", y_idx, x_idx, mat_idx, mat[mat_idx]);
        //    printf("y:%d x:%d inv_idx:%d inv_val:%d\\n", y_idx, x_idx, mat_idx_inv, mat[mat_idx_inv]);
        //}
    }

}
__device__ void glcm_normalizeInPlace(float *glcm_mat) {
    /* normalize probabilities */
    unsigned int ncounts = 0;
    if (!glcm_symmetric)
        ncounts = PATCH_SIZE;
    else
        ncounts = 2 * PATCH_SIZE;
    for (int i=0; i<GLCM_SIZE; i++) {
        glcm_mat[i] /= ncounts;
    }
}
__device__ float glcm_stat_contrast_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] * powf(y-x, 2);
        //if (threadIdx.x == 10 && blockIdx.x == 2) {
        //    printf("y:%d x:%d accum:%f\\n", y, x, accum);
        //}
    }
    return accum;
}
__device__ float glcm_stat_dissimilarity_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] * fabsf(y-x);
        //if (threadIdx.x == 10 && blockIdx.x == 2) {
        //    printf("y:%d x:%d accum:%f\\n", y, x, accum);
        //}
    }
    return accum;
}
__device__ float glcm_stat_energy_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        accum += powf(glcm_mat[i], 2);
        //if (threadIdx.x == 10 && blockIdx.x == 2) {
        //    printf("y:%d x:%d accum:%f\\n", y, x, accum);
        //}
    }
    return accum;
}
__device__ float glcm_stat_homogeneity_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] / (1 + powf(y-x, 2));
        //if (threadIdx.x == 10 && blockIdx.x == 2) {
            //printf("y:%d x:%d accum:%f\\n", y, x, accum);
        //}
    }
    return accum;
}
__device__ float glcm_stat_homogeneity1_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] / (1 + fabsf(y-x));
    }
    return accum;
}
__device__ float glcm_stat_autocorrelation_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += x*y*glcm_mat[i];
    }
    return accum;
}
__device__ float glcm_stat_clusterprominence_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += powf(x + y - rn::glcm_mean_Px(glcm_mat) - rn::glcm_mean_Py(glcm_mat), 4) * glcm_mat[i];
    }
    return accum;
}
__device__ float glcm_stat_clustershade_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += powf(x + y - rn::glcm_mean_Px(glcm_mat) - rn::glcm_mean_Py(glcm_mat), 3) * glcm_mat[i];
    }
    return accum;
}
__device__ float glcm_stat_clustertendency_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += powf(x + y - rn::glcm_mean_Px(glcm_mat) - rn::glcm_mean_Py(glcm_mat), 2) * glcm_mat[i];
    }
    return accum;
}
__device__ float glcm_stat_correlation_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += ((x * y * glcm_mat[i]) - (rn::glcm_mean_Px(glcm_mat)*rn::glcm_mean_Py(glcm_mat))) /
                  (sqrtf(rn::glcm_variance_Px(glcm_mat))*sqrtf(rn::glcm_variance_Py(glcm_mat)));
    }
    return accum;
}
__device__ float glcm_stat_sumentropy_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int k=2; k<=NBINS*2; k++) {
        float val = rn::glcm_marginal_Pxplusy(glcm_mat, k);
        accum -= val * log2f(val+eps);
    }
    return accum;
}
__device__ float glcm_stat_differenceentropy_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0;
    for (int k=0; k<NBINS; k++) {
        float val = rn::glcm_marginal_Pxminusy(glcm_mat, k);
        accum -= val * log2f(val+eps);
    }
    return accum;
}
__device__ float glcm_stat_entropy_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    return rn::glcm_HXY(glcm_mat);
}
//######### ADD ALL BELOW
__device__ float glcm_stat_avgintensity_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0.0f;
    for (int i=0; i<GLCM_SIZE; i++) {
        accum += glcm_mat[i]*rn::glcm_getY(i);
    }
   return accum;
}
__device__ float glcm_stat_differenceavg_gpu(float* glcm_mat) {
    float accum = 0.0f;
    for (int k=0; k<NBINS; k++) {
        accum += k*rn::glcm_marginal_Pxminusy(glcm_mat, k);
    }
    return accum;
}
__device__ float glcm_stat_differencevariance_gpu(float* glcm_mat) {
    float DA = glcm_stat_differenceavg_gpu(glcm_mat);
    float accum = 0;
    for (int k=0; k<NBINS; k++) {
        float val = rn::glcm_marginal_Pxminusy(glcm_mat, k);
        accum += powf(1-DA, 2) * val;
    }
    return accum;
}
__device__ float glcm_stat_imc1_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    return (rn::glcm_HXY(glcm_mat) - rn::glcm_HXY1(glcm_mat)) / fmaxf(rn::glcm_HX(glcm_mat), rn::glcm_HY(glcm_mat));
}
__device__ float glcm_stat_imc2_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    return sqrtf(1-expf(-2*(rn::glcm_HXY2(glcm_mat) - rn::glcm_HXY(glcm_mat))));
}
__device__ float glcm_stat_idm_gpu(float* glcm_mat) {
    float accum = 0.0f;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] / (1 + powf(y-x, 2));
    }
    return accum;
}
__device__ float glcm_stat_idmn_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0.0f;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] / (1 + ((powf(y-x, 2))/GLCM_SIZE));
    }
    return accum;
}
__device__ float glcm_stat_id_gpu(float* glcm_mat) {
    float accum = 0.0f;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] / (1 + fabsf(y-x));
    }
    return accum;
}
__device__ float glcm_stat_idn_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0.0f;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] / (1 + (fabsf(y-x)/GLCM_SIZE));
    }
    return accum;
}
__device__ float glcm_stat_inversevariance_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0.0f;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        if (y != x) {
            accum += glcm_mat[i] / powf(y-x, 2);
        }
    }
    return accum;
}
__device__ float glcm_stat_maxprob_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float maxval = 0.0f;
    for (int i=0; i<GLCM_SIZE; i++) {
        float val = glcm_mat[i];
        if (val > maxval) {
            maxval = val;
        }
    }
    return maxval;
}
__device__ float glcm_stat_sumavg_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0.0f;
    for (int k=2; k<=2*NBINS; k++) {
        accum += k*rn::glcm_marginal_Pxplusy(glcm_mat, k);
    }
    return accum;
}
__device__ float glcm_stat_sumvariance_gpu(float* glcm_mat) {
    float SE = glcm_stat_sumentropy_gpu(glcm_mat);
    float accum = 0;
    for (int k=2; k<=2*NBINS; k++) {
        float val = rn::glcm_marginal_Pxplusy(glcm_mat, k);
        accum += powf(k-SE, 2) * val;
    }
    return accum;
}
__device__ float glcm_stat_sumvariance2_gpu(float* glcm_mat) {
    float SA = glcm_stat_sumentropy_gpu(glcm_mat);
    float accum = 0;
    for (int k=2; k<=2*NBINS; k++) {
        float val = rn::glcm_marginal_Pxplusy(glcm_mat, k);
        accum += powf(k-SA, 2) * val;
    }
    return accum;
}
__device__ float glcm_stat_sumsquares_gpu(float* glcm_mat) {
    glcm_normalizeInPlace(glcm_mat);
    float accum = 0.0f;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += powf(y - rn::glcm_mean_Px(glcm_mat), 2) * glcm_mat[i];
    }
    return accum;
}



/* GLRLM FEATURES */
namespace rn {
    inline __device__ unsigned int glrlm_getY(unsigned int idx) {
        return (idx / MAXRUNLENGTH);
    }
    inline __device__ unsigned int glrlm_getX(unsigned int idx) {
        return (idx - glrlm_getY(idx)*MAXRUNLENGTH);
    }
    inline __device__ unsigned int glrlm_getIndex(unsigned int x, unsigned int y) {
        return (MAXRUNLENGTH * y + x);
    }
    __device__ float glrlm_sumP(float *glrlm_mat) {
        float accum = 0.0f;
        for (int i=0; i<GLRLM_SIZE; i++) {
            accum += glrlm_mat[i];
        }
        return accum;
    }
    __device__ float glrlm_sumSqP(float *glrlm_mat) {
        float accum = 0.0f;
        for (int i=0; i<GLRLM_SIZE; i++) {
            accum += powf(glrlm_mat[i], 2);
        }
        return accum;
    }
}
__device__ void glrlm_matrix_gpu(float *mat, float *array) {
    /*
      run length
       1  2  3  4  5 ... N
       --------------------
    1 |0  3  4  ...
    2 |1  3  2  ...
    3 |...
    ..|
    Q |

    y-axis: quantization level for run

    for N = patch side-length (max possible run)
        Q = number of quantization levels
    */

    // compute glrlm matrix in the specified directions
    int dx = DX;
    int dy = DY;
    int dz = DZ;

    int patch_size_1d = (RADIUS*2+1);
    int patch_size_2d = powf(patch_size_1d, 2);
    for (int i=0; i<PATCH_SIZE; i++) {
        int z = (int)floorf(i / patch_size_2d);
        int y = (int)floorf((i - patch_size_2d*z) / patch_size_1d);
        int x = (int)floorf(i - patch_size_2d*z - patch_size_1d*y);
        float intensity = array[i];
        if (intensity < 0){
            // We are trying to start from a voxel that was previously included in a run, skip this start
            continue;
        }

        // measure run length from this start point:
        unsigned int rl = 1;
        for (int j=0; j<MAXRUNLENGTH-2; j++) {
            // update query position
            z += dz;
            y += dy;
            x += dx;
            if (z>=Z_RADIUS*2+1 || z < 0 ||
                y>=patch_size_1d || y < 0 ||
                x>=patch_size_1d || x < 0) { break; }
            unsigned int q = patch_size_2d*z + patch_size_1d*y + x;
            if (array[q] == intensity) {
                rl++;
                array[q] = -1; // don't use as start point, any member of a previously counted run
            } else {break;} // run is broken
        }
        unsigned int x_idx = rl;

        int mat_idx = MAXRUNLENGTH * (int)intensity + x_idx;
        mat[mat_idx] += 1.0f;
    }
}
__device__ void glrlm_normalize_matrix(float *norm_mat, float *mat) {
    /* normalize probabilities */
    unsigned int ncounts = 0;
    for (int i=0; i<GLRLM_SIZE; i++) {
        ncounts += mat[i];
    }
    for (int i=0; i<GLRLM_SIZE; i++) {
        norm_mat[i] = mat[i]/ncounts;
    }
}
__device__ float glrlm_stat_sre_gpu(float* glrlm_mat) {
    // calculate statistic on glrlm matrix of size NBINS*MAXRUNLENGTH
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum += glrlm_mat[i]/powf(rn::glrlm_getX(i)+1, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_lre_gpu(float* glrlm_mat) {
    // calculate statistic on glrlm matrix of size NBINS*MAXRUNLENGTH
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum += glrlm_mat[i]*powf(rn::glrlm_getX(i)+1, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_gln_gpu(float* glrlm_mat) {
    // calculate statistic on glrlm matrix of size NBINS*MAXRUNLENGTH
    float accum = 0.0f;
    for (int y=0; y<NBINS; y++) {
        float subaccum = 0.0f;
        for (int x=0; x<MAXRUNLENGTH; x++) {
            subaccum += glrlm_mat[rn::glrlm_getIndex(x, y)];
        }
        accum += powf(subaccum, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_glnn_gpu(float* glrlm_mat) {
    // calculate statistic on glrlm matrix of size NBINS*MAXRUNLENGTH
    float accum = 0.0f;
    for (int y=0; y<NBINS; y++) {
        float subaccum = 0;
        for (int x=0; x<MAXRUNLENGTH; x++) {
            subaccum += glrlm_mat[rn::glrlm_getIndex(x, y)];
        }
        accum += powf(subaccum, 2);
    }
    return accum/rn::glrlm_sumSqP(glrlm_mat);
}
__device__ float glrlm_stat_rln_gpu(float* glrlm_mat) {
    // calculate statistic on glrlm matrix of size NBINS*MAXRUNLENGTH
    float accum = 0.0f;
    for (int x=0; x<MAXRUNLENGTH; x++) {
        float subaccum = 0.0f;
        for (int y=0; y<NBINS; y++) {
            subaccum += glrlm_mat[rn::glrlm_getIndex(x, y)];
        }
        accum += powf(subaccum, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_rlnn_gpu(float* glrlm_mat) {
    // calculate statistic on glrlm matrix of size NBINS*MAXRUNLENGTH
    float accum = 0.0f;
    for (int x=0; x<MAXRUNLENGTH; x++) {
        float subaccum = 0;
        for (int y=0; y<NBINS; y++) {
            subaccum += glrlm_mat[rn::glrlm_getIndex(x, y)];
        }
        accum += powf(subaccum, 2);
    }
    return accum/rn::glrlm_sumSqP(glrlm_mat);
}
__device__ float glrlm_stat_rp_gpu(float* glrlm_mat) {
    // calculate statistic on glrlm matrix of size NBINS*MAXRUNLENGTH
    float accum = rn::glrlm_sumP(glrlm_mat);
    return accum/PATCH_SIZE;
}
__device__ float glrlm_stat_glv_gpu(float* glrlm_mat) {
    float norm_mat[GLRLM_SIZE];
    glrlm_normalize_matrix(norm_mat, glrlm_mat);
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        float subaccum = 0.0f;
        for (int j=0; j<GLRLM_SIZE; j++) {
            subaccum += norm_mat[i]*(rn::glrlm_getY(j)+1);
        }
        accum += norm_mat[i]*powf(rn::glrlm_getY(i)+1-subaccum, 2);
    }
    return accum;
}
__device__ float glrlm_stat_rv_gpu(float* glrlm_mat) {
    float norm_mat[GLRLM_SIZE];
    glrlm_normalize_matrix(norm_mat, glrlm_mat);
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        float subaccum = 0.0f;
        for (int j=0; j<GLRLM_SIZE; j++) {
            subaccum += norm_mat[i]*(rn::glrlm_getX(j)+1);
        }
        accum += norm_mat[i]*powf(rn::glrlm_getX(i)+1-subaccum, 2);
    }
    return accum;
}
__device__ float glrlm_stat_re_gpu(float* glrlm_mat) {
    float norm_mat[GLRLM_SIZE];
    glrlm_normalize_matrix(norm_mat, glrlm_mat);
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum -= norm_mat[i]*log2f(norm_mat[i]+eps);
    }
    return accum;
}
__device__ float glrlm_stat_lglre_gpu(float* glrlm_mat) {
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum += glrlm_mat[i]/powf(rn::glrlm_getY(i)+1, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_hglre_gpu(float* glrlm_mat) {
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum += glrlm_mat[i]*powf(rn::glrlm_getY(i)+1, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_srlgle_gpu(float* glrlm_mat) {
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum += glrlm_mat[i]/powf(rn::glrlm_getY(i)+1 + rn::glrlm_getX(i)+1, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_srhgle_gpu(float* glrlm_mat) {
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum += glrlm_mat[i]*powf(rn::glrlm_getY(i)+1, 2)/powf(rn::glrlm_getX(i)+1, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_lrlglre_gpu(float* glrlm_mat) {
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum += glrlm_mat[i]*powf(rn::glrlm_getX(i)+1, 2)/powf(rn::glrlm_getY(i)+1, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}
__device__ float glrlm_stat_lrhglre_gpu(float* glrlm_mat) {
    float accum = 0.0f;
    for (int i=0; i<GLRLM_SIZE; i++) {
        accum += glrlm_mat[i]*powf(rn::glrlm_getY(i)+1 + rn::glrlm_getX(i)+1, 2);
    }
    return accum/rn::glrlm_sumP(glrlm_mat);
}


#define CADD_X $CADD_X
#define CADD_Y $CADD_Y
#define CADD_Z $CADD_Z
#define SADD   $SADD
#define CSUB_X $CSUB_X
#define CSUB_Y $CSUB_Y
#define CSUB_Z $CSUB_Z
#define SSUB   $SSUB

/* HAAR WAVELET FEATURES */
__device__ float haar_plugin_oneblock_gpu(float *patch_array) {
    // sum intensities in cube centered at c in patch relative to patch center with "radius" s
    unsigned int n = 0;
    float accum = 0.0f;
    for (int x=0; x<SADD; x++) {
        for (int y=0; y<SADD; y++) {
            for (int z=0; z<SADD; z++) {
                int idx = powf(RADIUS, 2)*(z-SADD-CADD_Z+Z_RADIUS) + (RADIUS*(y-SADD-CADD_Y+RADIUS)) + (x-SADD-CADD_X+RADIUS);
                if (idx >= 0 && idx < PATCH_SIZE) {
                    accum += patch_array[idx];
                    n++;
                }
            }
        }
    }
    return (float)(accum/n);
}
__device__ float haar_plugin_twoblock_gpu(float *patch_array) {
    // sum intensities in cube centered at cadd in patch relative to patch center with "radius" sadd
    // and subtract intensities in cube (csub, ssub)
    // sum intensities in cube centered at c in patch relative to patch center with "radius" s
    unsigned int n_add = 0;
    float accum_add = 0.0f;
    for (int x=0; x<SADD; x++) {
        for (int y=0; y<SADD; y++) {
            for (int z=0; z<SADD; z++) {
                int idx = powf(RADIUS, 2)*(z-SADD-CADD_Z+Z_RADIUS) + (RADIUS*(y-SADD-CADD_Y+RADIUS)) + (x-SADD-CADD_X+RADIUS);
                if (idx >= 0 && idx < PATCH_SIZE) {
                    accum_add += patch_array[idx];
                    n_add++;
                }
            }
        }
    }
    float val_add = (float)(accum_add/n_add);

    unsigned int n_sub = 0;
    float accum_sub = 0.0f;
    for (int x=0; x<SSUB; x++) {
        for (int y=0; y<SSUB; y++) {
            for (int z=0; z<SSUB; z++) {
                int idx = powf(RADIUS, 2)*(z-SSUB-CSUB_Z+Z_RADIUS) + (RADIUS*(y-SSUB-CSUB_Y+RADIUS)) + (x-SSUB-CSUB_X+RADIUS);
                if (idx >= 0 && idx < PATCH_SIZE) {
                    accum_sub += patch_array[idx];
                    n_sub++;
                }
            }
        }
    }
    float val_sub = (float)(accum_sub/n_sub);
    return val_add - val_sub;
}


__device__ float glcm_plugin_gpu(float *patch_array) {
    // quantize patch
    quantize_gpu(patch_array);

    // compute glcm matrix
    float mat[GLCM_SIZE];
    for (int i=0; i<GLCM_SIZE; i++) {
        mat[i] = 0.0f;
    }
    glcm_matrix_gpu(mat, patch_array);

    // compute matrix statistic
    float stat = ${STAT}(mat);

    return stat;
}
__device__ float glrlm_plugin_gpu(float *patch_array) {
    // quantize patch
    quantize_gpu(patch_array);

    // compute glcm matrix
    float mat[GLRLM_SIZE];
    for (int i=0; i<GLRLM_SIZE; i++) {
        mat[i] = 0.0f;
    }
    glrlm_matrix_gpu(mat, patch_array);

    // compute matrix statistic
    float stat = ${STAT}(mat);

    return stat;
}

__global__ void image_iterator_gpu(float *image_vect, float *result_vect) {
    // array index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < IMAGE_SIZE) {
        float patch_array[PATCH_SIZE];
        int i = 0;
        for (int k_z = -Z_RADIUS; k_z <= Z_RADIUS; k_z++) {
            for (int k_y = -RADIUS; k_y <= RADIUS; k_y++) {
                for (int k_x = -RADIUS; k_x <= RADIUS; k_x++) {
                    int k_idx = (int)(fminf(IMAGE_SIZE-1, fmaxf(0, (IMAGE_HEIGHT*IMAGE_WIDTH*k_z) + (IMAGE_WIDTH*k_y) + (idx + k_x))));

                    // Count unique pixel intensities
                    patch_array[i] = image_vect[k_idx];
                    //if (idx == 500) {
                    //    printf("i:%d - k_idx:%d - patchval:%f - imageval:%f\\n", i, k_idx, patch_array[i], image_vect[idx]);
                    //}
                    i += 1;
                }
            }
        }
        result_vect[idx] = ${KERNEL}(patch_array);
    }
}

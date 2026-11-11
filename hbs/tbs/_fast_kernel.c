/* Fused LSE objective and gradient for the 62k finite-P1 TBS problem.
 *
 * One call per L-BFGS evaluation: gathers the three vertex values per face,
 * forms the Wirtinger derivatives, accumulates the log-sum-exp value with a
 * shifted softmax, and scatters the gradient of
 *
 *     F_beta(x) = (1/beta) log sum_f exp(beta * r_f)
 *
 * into a dense 2n vector.  All arrays are float64; face indices index the
 * full vertex array (interior first, then the fixed boundary).
 *
 * This file is research code: it does not touch hbs_python.  Build with
 *
 *     gcc -O3 -march=native -shared -fPIC -o _fast_kernel.so _fast_kernel.c
 *
 * The arithmetic follows `_lse_evaluate_dense` in solver.py: the exact ratio
 * |fbar/fz| drives the softmax, the eps-regularized ratio drives the
 * derivatives, and the constraint gradient is returned with the solver's
 * sign convention (gradient of the objective, i.e. sum_f w_f d r_f / dx).
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Return codes */
#define TBS_OK 0
#define TBS_NONFINITE 1

static inline double sq(double v) { return v * v; }

/* Compute the objective value and gradient.
 *
 * x        : 2n vector, [Re values[:n], Im values[:n]]
 * values_re/values_im : full vertex array (interior + boundary), used as the
 *            template for the boundary entries
 * faces    : 3 per face, indices into the vertex array
 * gx, gy   : 3 per face
 * n_faces, n_interior, n_vertices
 * beta     : positive
 * grad     : output 2n vector (caller zeroes it; this function adds)
 * ratios   : output n_faces
 * min_jac  : output scalar
 * scratch  : optional work space of 5*n_faces doubles: [fzr|fzi|fbr|fbi|weights];
 *            when NULL it is allocated internally.  The first pass stores the
 *            Wirtinger derivatives so the gradient pass does not recompute
 *            them (the gather is the most expensive part of the kernel).
 */
int tbs_lse_objective(const double *x,
                      const double *values_re, const double *values_im,
                      const int64_t *faces,
                      const double *gx, const double *gy,
                      int64_t n_faces, int64_t n_interior, int64_t n_vertices,
                      double beta,
                      double *grad, double *ratios, double *min_jac,
                      double *value_out, double *scratch)
{
    (void)n_vertices;
    const int64_t n = n_interior;
    double *own = NULL;
    if (scratch == NULL) {
        own = (double *)malloc((size_t)(5*n_faces)*sizeof(double));
        if (own == NULL) return TBS_NONFINITE;
        scratch = own;
    }
    double *fzr_s = scratch;
    double *fzi_s = scratch + n_faces;
    double *fbr_s = scratch + 2*n_faces;
    double *fbi_s = scratch + 3*n_faces;
    double *weights = scratch + 4*n_faces;
    double peak = -INFINITY;
    double min_j = INFINITY;
    for (int64_t f = 0; f < n_faces; ++f) {
        const int64_t i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
        const double v0r = (i0 < n) ? x[i0] : values_re[i0];
        const double v0i = (i0 < n) ? x[n + i0] : values_im[i0];
        const double v1r = (i1 < n) ? x[i1] : values_re[i1];
        const double v1i = (i1 < n) ? x[n + i1] : values_im[i1];
        const double v2r = (i2 < n) ? x[i2] : values_re[i2];
        const double v2i = (i2 < n) ? x[n + i2] : values_im[i2];
        const double gx0 = gx[3*f], gx1 = gx[3*f+1], gx2 = gx[3*f+2];
        const double gy0 = gy[3*f], gy1 = gy[3*f+1], gy2 = gy[3*f+2];
        const double fxr = v0r*gx0 + v1r*gx1 + v2r*gx2;
        const double fxi = v0i*gx0 + v1i*gx1 + v2i*gx2;
        const double fyr = v0r*gy0 + v1r*gy1 + v2r*gy2;
        const double fyi = v0i*gy0 + v1i*gy1 + v2i*gy2;
        /* fz = (fx - i fy)/2, fbar = (fx + i fy)/2 */
        const double fzr = 0.5*(fxr + fyi), fzi = 0.5*(fxi - fyr);
        const double fbr = 0.5*(fxr - fyi), fbi = 0.5*(fxi + fyr);
        fzr_s[f] = fzr; fzi_s[f] = fzi;
        fbr_s[f] = fbr; fbi_s[f] = fbi;
        const double jac = fxr*fyi - fxi*fyr;
        if (jac < min_j) min_j = jac;
        const double fz2 = fzr*fzr + fzi*fzi;
        const double fb2 = fbr*fbr + fbi*fbi;
        double scales = fz2 + fb2;
        if (!(scales >= 1e-12)) scales = 1e-12;
        const double eps2 = 1e-20*scales;
        const double r = sqrt(fb2)/sqrt(fz2);
        ratios[f] = r;
        if (r > peak) peak = r;
        weights[f] = r; /* keep the ratio; converted to weights below */
    }
    if (!(peak > -INFINITY) || !isfinite(peak)) {
        if (own) free(own);
        return TBS_NONFINITE;
    }
    double total = 0.0;
    for (int64_t f = 0; f < n_faces; ++f) {
        const double w = exp(beta*(weights[f] - peak));
        weights[f] = w;
        total += w;
    }
    const double value = peak + log(total)/beta;
    for (int64_t f = 0; f < n_faces; ++f) weights[f] /= total;

    for (int64_t f = 0; f < n_faces; ++f) {
        const double fzr = fzr_s[f], fzi = fzi_s[f];
        const double fbr = fbr_s[f], fbi = fbi_s[f];
        const double fz2 = fzr*fzr + fzi*fzi;
        const double fb2 = fbr*fbr + fbi*fbi;
        double scales = fz2 + fb2;
        if (!(scales >= 1e-12)) scales = 1e-12;
        const double eps2 = 1e-20*scales;
        const double ratio_eps = sqrt(fb2 + eps2)/sqrt(fz2 + eps2);
        const double inv_fb2 = 1.0/(fb2 + eps2);
        const double inv_fz2 = 1.0/(fz2 + eps2);
        const double w = weights[f];
        /* cb = (gx + i gy)/2, cz = (gx - i gy)/2; the derivative of r w.r.t.
         * the local [x,y] dofs uses conj(fbar)*cb and conj(fz)*cz.  With
         * conj(fbar)*cb = 0.5[(fbr*gx + fbi*gy) + i(fbr*gy - fbi*gx)] and
         * conj(fz)*cz = 0.5[(fzr*gx - fzi*gy) - i(fzr*gy + fzi*gx)]. */
        for (int local = 0; local < 3; ++local) {
            const int64_t v = faces[3*f + local];
            if (v >= n) continue;
            const double gxl = gx[3*f + local], gyl = gy[3*f + local];
            const double b_re = 0.5*(fbr*gxl + fbi*gyl);
            const double b_im = 0.5*(fbr*gyl - fbi*gxl);
            const double z_re = 0.5*(fzr*gxl - fzi*gyl);
            const double z_im = -0.5*(fzr*gyl + fzi*gxl);
            const double dr_x = ratio_eps*(b_re*inv_fb2 - z_re*inv_fz2);
            const double dr_y = ratio_eps*(z_im*inv_fz2 - b_im*inv_fb2);
            grad[v] += w*dr_x;
            grad[n + v] += w*dr_y;
        }
    }
    if (own) free(own);
    *min_jac = min_j;
    *value_out = value;
    return TBS_OK;
}

/* Objective value only (used by the line search when the gradient is not
 * needed).  Returns the value through *out; returns TBS_NONFINITE on break. */
int tbs_lse_value(const double *x,
                  const double *values_re, const double *values_im,
                  const int64_t *faces,
                  const double *gx, const double *gy,
                  int64_t n_faces, int64_t n_interior,
                  double beta, double *out, double *min_jac)
{
    const int64_t n = n_interior;
    double peak = -INFINITY, min_j = INFINITY;
    double total = 0.0;
    /* two passes: peak, then shifted sum; ratios are not stored */
    for (int64_t f = 0; f < n_faces; ++f) {
        const int64_t i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
        const double v0r = (i0 < n) ? x[i0] : values_re[i0];
        const double v0i = (i0 < n) ? x[n + i0] : values_im[i0];
        const double v1r = (i1 < n) ? x[i1] : values_re[i1];
        const double v1i = (i1 < n) ? x[n + i1] : values_im[i1];
        const double v2r = (i2 < n) ? x[i2] : values_re[i2];
        const double v2i = (i2 < n) ? x[n + i2] : values_im[i2];
        const double gx0 = gx[3*f], gx1 = gx[3*f+1], gx2 = gx[3*f+2];
        const double gy0 = gy[3*f], gy1 = gy[3*f+1], gy2 = gy[3*f+2];
        const double fxr = v0r*gx0 + v1r*gx1 + v2r*gx2;
        const double fxi = v0i*gx0 + v1i*gx1 + v2i*gx2;
        const double fyr = v0r*gy0 + v1r*gy1 + v2r*gy2;
        const double fyi = v0i*gy0 + v1i*gy1 + v2i*gy2;
        const double fzr = 0.5*(fxr + fyi), fzi = 0.5*(fxi - fyr);
        const double fbr = 0.5*(fxr - fyi), fbi = 0.5*(fxi + fyr);
        const double jac = fxr*fyi - fxi*fyr;
        if (jac < min_j) min_j = jac;
        const double r = sqrt(fbr*fbr + fbi*fbi)/sqrt(fzr*fzr + fzi*fzi);
        if (!isfinite(r)) { *out = NAN; *min_jac = min_j; return TBS_NONFINITE; }
        if (r > peak) peak = r;
    }
    for (int64_t f = 0; f < n_faces; ++f) {
        /* recompute the ratio; one extra pass keeps the value-only path
         * allocation free */
        const int64_t i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
        const double v0r = (i0 < n) ? x[i0] : values_re[i0];
        const double v0i = (i0 < n) ? x[n + i0] : values_im[i0];
        const double v1r = (i1 < n) ? x[i1] : values_re[i1];
        const double v1i = (i1 < n) ? x[n + i1] : values_im[i1];
        const double v2r = (i2 < n) ? x[i2] : values_re[i2];
        const double v2i = (i2 < n) ? x[n + i2] : values_im[i2];
        const double gx0 = gx[3*f], gx1 = gx[3*f+1], gx2 = gx[3*f+2];
        const double gy0 = gy[3*f], gy1 = gy[3*f+1], gy2 = gy[3*f+2];
        const double fxr = v0r*gx0 + v1r*gx1 + v2r*gx2;
        const double fxi = v0i*gx0 + v1i*gx1 + v2i*gx2;
        const double fyr = v0r*gy0 + v1r*gy1 + v2r*gy2;
        const double fyi = v0i*gy0 + v1i*gy1 + v2i*gy2;
        const double fzr = 0.5*(fxr + fyi), fzi = 0.5*(fxi - fyr);
        const double fbr = 0.5*(fxr - fyi), fbi = 0.5*(fxi + fyr);
        const double r = sqrt(fbr*fbr + fbi*fbi)/sqrt(fzr*fzr + fzi*fzi);
        total += exp(beta*(r - peak));
    }
    *out = peak + log(total)/beta;
    *min_jac = min_j;
    return TBS_OK;
}
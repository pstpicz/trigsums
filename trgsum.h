/** @file trgsum.h
 *  @brief Function prototypes for the package "Trigonemetric sums".
 *
 *
 * It contains sequential and vectorized implementations of Goertzel and Reinsch algorithms
 * for finding trigonometric sums:
 *
 *       C(x) = b_0 cos(0) + b_1 cos(x) + b_2 cos(2x) +...+ b_n cos(nx)
 *       S(x) =              b_1 sin(x) + b_2 sin(2x) +...+ b_n sin(nx)
 *
 * @author Przemyslaw Stpiczynski
 * @bug No known bugs.
 */

#ifndef TRGSUM_H_INCLUDED
#define TRGSUM_H_INCLUDED


/** @brief Implementation of basic sequential Goertzel algorithm.
 *
 *  @param x Argument for C(x) and S(x)
 *  @param n Number of coefficients, b_0,...,b_n
 *  @param b Pointer to coefficients
 *  @param cx Pointer to computed C(x)
 *  @param sx Pointer to computed S(x)
 *  @return void
 */
void trg_d_s_goertzel(double x,int n, double *b, double *cx, double *sx);


/** @brief Implementation of vectorized Goertzel algorithm introduced by Stpiczynski.
 *
 *  @param x Argument for C(x) and S(x)
 *  @param n Number of coefficients, b_0,...,b_n
 *  @param b Pointer to coefficients (should be allocated using _mm_malloc())
 *  @param cx Pointer to computed C(x)
 *  @param sx Pointer to computed S(x)
 *  @return void
 */
void trg_d_v_goertzel(double x,int n, double *b, double *cx, double *sx);

/** @brief Implementation of parallel vectorized Goertzel algorithm introduced by Stpiczynski.
 *
 *  @param x Argument for C(x) and S(x)
 *  @param n Number of coefficients, b_0,...,b_n
 *  @param b Pointer to coefficients (should be allocated using _mm_malloc())
 *  @param cx Pointer to computed C(x)
 *  @param sx Pointer to computed S(x)
 *  @param p Number of processors
 *  @return void
 */
void trg_d_p_goertzel(double x,int n, double *b, double *cx, double *sx, int p);


/** @brief Implementation of basic sequential Reinschalgorithm.
 *
 *  @param x Argument for C(x) and S(x)
 *  @param n Number of coefficients, b_0,...,b_n
 *  @param b Pointer to coefficients
 *  @param cx Pointer to computed C(x)
 *  @param sx Pointer to computed S(x)
 *  @return void
 */
void trg_d_s_reinsch(double x, int n, double *b, double *cx, double *sx);


/** @brief Implementation of vectorized Reinsch algorithm introduced by Stpiczynski.
 *
 *  @param x Argument for C(x) and S(x)
 *  @param n Number of coefficients, b_0,...,b_n
 *  @param b Pointer to coefficients (should be allocated using _mm_malloc())
 *  @param cx Pointer to computed C(x)
 *  @param sx Pointer to computed S(x)
 *  @return void
 */
void trg_d_v_reinsch(double x, int n, double *b, double *cx, double *sx);

/** @brief Implementation of parallel vectorized Reinsch algorithm introduced by Stpiczynski.
 *
 *  @param x Argument for C(x) and S(x)
 *  @param n Number of coefficients, b_0,...,b_n
 *  @param b Pointer to coefficients (should be allocated using _mm_malloc())
 *  @param cx Pointer to computed C(x)
 *  @param sx Pointer to computed S(x)
 *  @param p Number of processors
 *  @return void
 */
void trg_d_p_reinsch(double x, int n, double *b, double *cx, double *sx, int p);


#endif // TRGSUM_H_INCLUDED

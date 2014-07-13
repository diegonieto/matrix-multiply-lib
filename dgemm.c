/*
 MML : Matrix Multiply LIB

Copyright (C) 2014  Diego Nieto Muñoz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "dgemm.h"
#include <xmmintrin.h>

#define BLOCKSIZE 250 // Must be pair.
#define min(a,b) (((a)<(b))?(a):(b))

void dgemm(int m, int n, int p, const double *left,
        const double *right, double *dest) {
    int i, j, k, ii, jj, kk, lim_i, lim_j, lim_k;
    double *t_left = (double*) _mm_malloc(sizeof(double)*BLOCKSIZE*BLOCKSIZE, 16);
    double *t_right = (double*) _mm_malloc(sizeof(double)*BLOCKSIZE*BLOCKSIZE, 16);
    double *t_dest = (double*) _mm_malloc(sizeof(double)*BLOCKSIZE*BLOCKSIZE, 16);
    __m128d m_left, m_right, m_acum, m_mul, m_aux;
    double value1 __attribute__ ((aligned(16)));
    double value2 __attribute__ ((aligned(16)));
    char pair;
    for (ii = 0; ii < m; ii += BLOCKSIZE)
        for (jj = 0; jj < n; jj += BLOCKSIZE) {
            kk = 0;
            // Temporal storage
            // Storage left matrix
            for (i = ii; i < min(ii + BLOCKSIZE, m); i++) {
                    for (k = kk; k < min(p, kk + BLOCKSIZE); k++) {
                        t_left[(i-ii)*BLOCKSIZE+(k-kk)] = left[p * i + k];
                    }
            }
            // Storage right matrix (transposed)
            for (k = kk; k < min(p, kk + BLOCKSIZE); k++) {
                    for (j = jj; j < min(jj + BLOCKSIZE, n); j++) {
                        t_right[(j-jj)*BLOCKSIZE+(k-kk)] = right[n * k + j];
                    }
            }
            lim_i = min(ii + BLOCKSIZE, m);
            for (i = ii; i < lim_i; i++) {
                lim_j = min(jj + BLOCKSIZE, n);
                for (j = jj; j < lim_j; j++) {
                    lim_k = min(p, kk + BLOCKSIZE);
                    pair = lim_k % 2;
                    lim_k -= pair;
                    m_acum = _mm_setzero_pd();
                    // Compute iterations using SIMD
                    for (k = kk; k < lim_k; k+=2) {
                        m_left  = _mm_load_pd(&t_left[(i-ii)*BLOCKSIZE+(k-kk)]);
                        m_right = _mm_load_pd(&t_right[(j-jj)*BLOCKSIZE+(k-kk)]);
                        m_mul   = _mm_mul_pd(m_left, m_right);
                        m_acum  = _mm_add_pd(m_acum, m_mul);
                    }
                    _mm_storel_pd(&value1, m_acum);
                    _mm_storeh_pd(&value2, m_acum);
                    // Compute non pair iteration
                    t_dest[BLOCKSIZE * (i-ii) + j-jj] = value1 + value2;
                    if(pair) {
                        t_dest[BLOCKSIZE * (i-ii) + j-jj] += t_left[(i-ii)*BLOCKSIZE+(lim_k-kk)] * t_right[(j-jj)*BLOCKSIZE+(lim_k-kk)];
                    }
                }
            }
            kk = BLOCKSIZE;
            for (kk = BLOCKSIZE; kk < p; kk += BLOCKSIZE) {
                // Temporal storage

                // Storage left matrix
                for (i = ii; i < min(ii + BLOCKSIZE, m); i++) {
                        for (k = kk; k < min(p, kk + BLOCKSIZE); k++) {
                            t_left[(i-ii)*BLOCKSIZE+(k-kk)] = left[p * i + k];
                        }
                }


                // Storage right matrix (traspose)
                for (k = kk; k < min(p, kk + BLOCKSIZE); k++) {
                        for (j = jj; j < min(jj + BLOCKSIZE, n); j++) {
                            t_right[(j-jj)*BLOCKSIZE+(k-kk)] = right[n * k + j];
                        }
                }

                for (i = ii; i < min(ii + BLOCKSIZE, m); i++) {
                    for (j = jj; j < min(jj + BLOCKSIZE, n); j++) {
                        lim_k = min(p, kk + BLOCKSIZE);
                        pair = lim_k % 2;
                        lim_k -= pair;
                        m_acum = _mm_setzero_pd();
                        // Compute iterations using SIMD
                        for (k = kk; k < lim_k; k+=2) {
                            m_left  = _mm_load_pd(&t_left[(i-ii)*BLOCKSIZE+(k-kk)]);
                            m_right = _mm_load_pd(&t_right[(j-jj)*BLOCKSIZE+(k-kk)]);
                            m_mul   = _mm_mul_pd(m_left, m_right);
                            m_acum  = _mm_add_pd(m_acum, m_mul);
                        }
                        _mm_storel_pd(&value1, m_acum);
                        _mm_storeh_pd(&value2, m_acum);
                        // Compute non pair iteration
                        if(pair) {
                            t_dest[BLOCKSIZE * (i-ii) + j-jj] += t_left[(i-ii)*BLOCKSIZE+(lim_k-kk)] * t_right[(j-jj)*BLOCKSIZE+(lim_k-kk)];
                        }
                        t_dest[BLOCKSIZE * (i-ii) + j-jj] += value1 + value2;
                    }
                }
            }
            // Temporal storage for dest
            for (i = ii; i < min(m, ii + BLOCKSIZE); i++) {
                    for (j = jj; j < min(jj + BLOCKSIZE, n); j++) {
                        dest[n * i + j] =  t_dest[BLOCKSIZE * (i-ii) + j-jj];
                    }
            }
        }
    _mm_free(t_left);
    _mm_free(t_right);
    _mm_free(t_dest);
}

void naive_dgemm(int m, int n, int p, const double *left,
        const double *right, double *dest) {
    int i, j, k;
    for(i=0; i<m; i++) {
        for(j=0; j<n; j++) {
            dest[i*n+j] = 0;
            for(k=0; k<p; k++) {
                dest[i*n+j] += left[i*p+k] * right[k*n+j];
            }
        }
    }
}

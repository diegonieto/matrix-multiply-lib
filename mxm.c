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
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

int main(int argc, char **argv) {

    int m, n, p;
    struct timeval t1, t2, t;
    int i, j;

    if(argc == 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        p = atoi(argv[3]);
    } else {
        m = 1000;
        n = 2000;
        p = 1500;
    }

    double *left = (double*) malloc(m * p * sizeof(double));
    double *right = (double*) malloc(p * n * sizeof(double));
    double *dest1 = (double*) malloc(m * n * sizeof(double));
    double *dest2 = (double*) malloc(m * n * sizeof(double));

    for (i = 0; i < m; i++)
        for (j = 0; j < p; j++)
            left[i * p + j] = i * p + j;

    for (i = 0; i < p; i++)
        for (j = 0; j < n; j++)
            right[i * n + j] = i * n + j;

    gettimeofday(&t1, NULL);
    dgemm(m, n, p, left, right, dest1);
    gettimeofday(&t2, NULL);
    timersub(&t2, &t1, &t);
    printf("Total time optimized = %f s\n", t.tv_sec + t.tv_usec / 1000000.0);

    gettimeofday(&t1, NULL);
    naive_dgemm(m, n, p, left, right, dest2);
    gettimeofday(&t2, NULL);
    timersub(&t2, &t1, &t);
    printf("Total time naive = %f s\n", t.tv_sec + t.tv_usec / 1000000.0);

    free(left);
    free(right);
    free(dest1);
    free(dest2);
}

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

void test1();
void test2();
void test3();

int main() {
    test1();
    test2();
    test3();
}

void test1() {
    int i, j;
    const int m = 4;
    const int n = 5;
    const int p = 6;
    double *left = (double*)malloc(m*p*sizeof(double));
    double *right = (double*)malloc(p*n*sizeof(double));
    double *dest1 = (double*)malloc(m*n*sizeof(double));
    double *dest2 = (double*)malloc(m*n*sizeof(double));

    for(i=0; i<m; i++)
        for(j=0; j<p; j++)
            left[i*p+j] = i*p+j;

    for(i=0; i<p; i++)
        for(j=0; j<n; j++)
            right[i*n+j] = i*n+j;

    dgemm(m,n,p,left,right, dest1);

    naive_dgemm(m,n,p,left,right, dest2);

    for(i=0; i<m; i++)
        for(j=0; j<n; j++)
            assert(dest1[i*n+j] == dest1[i*n+j]);

    free(left);
    free(right);
    free(dest1);
    free(dest2);
}

void test2() {
    int i, j;
    const int m = 2;
    const int n = 5;
    const int p = 4;
    double *left = (double*)malloc(m*p*sizeof(double));
    double *right = (double*)malloc(p*n*sizeof(double));
    double *dest1 = (double*)malloc(m*n*sizeof(double));
    double *dest2 = (double*)malloc(m*n*sizeof(double));

    for(i=0; i<m; i++)
        for(j=0; j<p; j++)
            left[i*p+j] = i*p+j;

    for(i=0; i<p; i++)
        for(j=0; j<n; j++)
            right[i*n+j] = i*n+j;

    dgemm(m,n,p,left,right, dest1);

    naive_dgemm(m,n,p,left,right, dest2);

    for(i=0; i<m; i++)
        for(j=0; j<n; j++)
            assert(dest1[i*n+j] == dest1[i*n+j]);

    free(left);
    free(right);
    free(dest1);
    free(dest2);
}

void test3() {
    int i, j;
    const int m = 3;
    const int n = 2;
    const int p = 3;
    double *left = (double*)malloc(m*p*sizeof(double));
    double *right = (double*)malloc(p*n*sizeof(double));
    double *dest1 = (double*)malloc(m*n*sizeof(double));
    double *dest2 = (double*)malloc(m*n*sizeof(double));

    for(i=0; i<m; i++)
        for(j=0; j<p; j++)
            left[i*p+j] = i*p+j;

    for(i=0; i<p; i++)
        for(j=0; j<n; j++)
            right[i*n+j] = i*n+j;

    dgemm(m,n,p,left,right, dest1);

    naive_dgemm(m,n,p,left,right, dest2);

    for(i=0; i<m; i++)
        for(j=0; j<n; j++)
            assert(dest1[i*n+j] == dest1[i*n+j]);

    free(left);
    free(right);
    free(dest1);
    free(dest2);
}

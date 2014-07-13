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

#ifndef DGEMM_H_
#define DGEMM_H_

/*
 * @brief Multiply left matrix by right matrix and store in dest
 ∗
 ∗ @param[in] m Number of rows of left
 ∗ @param[in] n Number of columns of right
 ∗ @param[in] k Number of columns of left
 ∗ @param[in] left Left matrix address
 ∗ @param[in] right Right matrix address
 ∗ @param[in] dest Result matrix address
 */
void dgemm(int m, int n, int p, const double *left,
        const double *right, double *dest);

/*
 * @brief Multiply left matrix by right matrix and store in dest
 ∗
 ∗ @param[in] m Number of rows of left
 ∗ @param[in] n Number of columns of right
 ∗ @param[in] k Number of columns of left
 ∗ @param[in] left Left matrix address
 ∗ @param[in] right Right matrix address
 ∗ @param[in] dest Result matrix address
 */
void naive_dgemm(int m, int n, int p, const double *left,
        const double *right, double *dest);

#endif /* DGEMM_H_ */

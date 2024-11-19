/**************************************************************************************************
*                                                                                                 *
* This file is part of HPIPM.                                                                     *
*                                                                                                 *
* HPIPM -- High-Performance Interior Point Method.                                                *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

// macro to string
#define STR(x) STR_AGAIN(x)
#define STR_AGAIN(x) #x

// system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// hpipm
#include "hpipm_d_ocp_qp_dim.h"
#include "hpipm_d_ocp_qp_ipm.h"
// mex
#include "mex.h"

// data
#include STR(QP_DATA_H)



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
	{

//	printf("\nin ocp_qp_dim_load\n");

	mxArray *tmp_mat;
	long long *l_ptr;
	char *c_ptr;

	/* RHS */

	// dim
	l_ptr = mxGetData( prhs[0] );
	struct d_ocp_qp_dim *dim = (struct d_ocp_qp_dim *) *l_ptr;

	/* arg */

	hpipm_size_t arg_size = sizeof(struct d_ocp_qp_ipm_arg) + d_ocp_qp_ipm_arg_memsize(dim);
	void *arg_mem = malloc(arg_size);

	c_ptr = arg_mem;

	struct d_ocp_qp_ipm_arg *arg = (struct d_ocp_qp_ipm_arg *) c_ptr;
	c_ptr += sizeof(struct d_ocp_qp_ipm_arg);

	d_ocp_qp_ipm_arg_create(dim, arg, c_ptr);
	c_ptr += d_ocp_qp_ipm_arg_memsize(dim);

	d_ocp_qp_ipm_arg_set_default(mode, arg);

	d_ocp_qp_ipm_arg_set_mu0(&mu0, arg);
	d_ocp_qp_ipm_arg_set_iter_max(&iter_max, arg);
	d_ocp_qp_ipm_arg_set_alpha_min(&alpha_min, arg);
	d_ocp_qp_ipm_arg_set_mu0(&mu0, arg);
	d_ocp_qp_ipm_arg_set_tol_stat(&tol_stat, arg);
	d_ocp_qp_ipm_arg_set_tol_eq(&tol_eq, arg);
	d_ocp_qp_ipm_arg_set_tol_ineq(&tol_ineq, arg);
	d_ocp_qp_ipm_arg_set_tol_comp(&tol_comp, arg);
	d_ocp_qp_ipm_arg_set_reg_prim(&reg_prim, arg);
	d_ocp_qp_ipm_arg_set_warm_start(&warm_start, arg);
	d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr, arg);
	d_ocp_qp_ipm_arg_set_ric_alg(&ric_alg, arg);

	/* LHS */

	tmp_mat = mxCreateNumericMatrix(1, 1, mxINT64_CLASS, mxREAL);
	l_ptr = mxGetData(tmp_mat);
	l_ptr[0] = (long long) arg_mem;
	plhs[0] = tmp_mat;

	return;

	}

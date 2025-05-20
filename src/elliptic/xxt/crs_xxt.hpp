#if !defined(_NEKRS_XXT_HPP_)
#define _NEKRS_XXT_HPP_

#include "gslib.h"

int xxt_setup(MPI_Comm comm, uint n, const ulong *id, uint nnz, const uint *Ai,
              const uint *Aj, const double *A, uint null, uint verbose);

int xxt_solve(double *h_x, double *h_b);

int xxt_free();

#endif

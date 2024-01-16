def svd_routine(lapack_driver="gesvd", **kwds):
    import scipy
    # default lapack_driver="gesdd"
    def _svd(*args):
        U,S,V = scipy.linalg.svd(*args, lapack_driver=lapack_driver, **kwds)
        return U,S,V.T.conj()
    return _svd

def lsq_solver(options):
    import numpy.linalg
    return lambda *args: numpy.linalg.lstsq(*args, rcond=None)[0]
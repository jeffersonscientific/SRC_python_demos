import random
import numpy
import numba
import multiprocessing as mpp
#

def f_pipe(f, pipe, *args, **kwargs):
    # a wrapper to run f() in an Process(), or something like it.
    x = f(*args, **kwargs)
    pipe.send(x)
    pipe.close()
    #
    return None
    
# Just do a loop-loop. This will perform unimpressively...
def calc_pi_loop_loop(N):
    M = 0
    for i in range(int(N)):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            M += 1
    return 4 * M / N

# Vectorize: Versatile, fast, -- what you will usually do.
def calc_pi_vec(N):
    #M = 0
    # spell it out, then consolidate...
    #X = numpy.random.random(N)
    #Y = numpy.random.random(N)
    #Z = X**2 + Y**2
    #
    # All of that in one line...
    # You should experiment with some of this syntax to evaluate performance.
    # this looks much faster than numpy.square()
#    Z = numpy.sum( numpy.random.random( (2,N) )**2. , axis=0 )
#    #Z = numpy.sum( numpy.square( numpy.random.random( (2,N) ) ), axis=0 )
#    #
#    # as a little trick, sum the index of (Z<1)
#    M = numpy.sum((Z<1))
    
    M = numpy.sum( (numpy.sum( numpy.random.random( (2, int(N)) )**2. , axis=0 ))<1 )
    #
    return 4.0*M/N
def calc_pi_vec_mpp(N, ncpus=1):
    #
    # if ncpus>1, run an MPP wrapper that pseudo-recursively calls back this function with ncpus=1, and so just executes
    #  the spp code (below, after the if- block.
    if ncpus > 1:
        # do some flavor of MPP.
        # For nowk sticking with the Process() example:
        pipes = [mpp.Pipe() for k in range(ncpus)]
        #
        # Split up the problem into ncpus peices...
        #
        # use a trick way to do a ceil() operatior in Python:
        # n_ceil = -(-a//b)
        N_per_cpu = int(-(-N/ncpus))
        # A list of Process() instances:
        Procs = [mpp.Process(target=f_pipe, args=[calc_pi_vec_mpp, p2, N_per_cpu,1]) for p1,p2 in pipes]
        for P in Procs:
            P.start()
        for P in Procs:
            P.join()
        #
        pi = numpy.mean([p1.recv() for p1,p2 in pipes])
        for (p1,p2),P in zip(pipes, Procs):
            p1.close()
            p2.close()
            P.close()
        #
        return pi
    #
    # See calc_pi_vec() for a break-out and explanation of this vectorized code.
    M = numpy.sum( (numpy.sum( numpy.random.random( (2, int(N)) )**2. , axis=0 ))<1 )
    #
    return 4.0*M/N
# numba:
# Can be the fastest, but only under certain, often inconvenient circumstances
## Compile with Numba:
#
@numba.jit()
def calc_pi_jit(N):
    M = 0
    for i in range(int(N)):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            M += 1
    return 4 * M / N

@numba.jit(nopython=True)
def calc_pi_jit_np(N):
    M = 0
    for i in range(int(N)):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            M += 1
    return 4 * M / N

@numba.jit(nogil=True)
def calc_pi_jit_ng(N):
    M = 0
    for i in range(int(N)):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            M += 1
    return 4 * M / N

@numba.jit(nopython=True, nogil=True)
def calc_pi_jit_ng_np(N):
    M = 0
    for i in range(int(N)):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            M += 1
    return 4 * M / N
# A hack way to write an alias for a function:
calc_pi_jig_np_ng = calc_pi_jit_ng_np
#
def calc_pi_pool(ncpu=4, N_max=1E7, f_pi=calc_pi_loop_loop):
    #
    with mpp.Pool(processes=4) as pool:
        #
        # evaluate "f(20)" asynchronously
        # I really don't know when to use apply_async() vs apply(). I used to use _async() because I could pass kwds.
        res = [pool.apply_async(f_pi, (N_max,)) for k in range(ncpu)]      # runs in *only* one process
        #res = [pool.apply(f_pi, (N_max,)) for k in range(ncpu)]
        pool.close()
        pool.join()
    #    print(res.get(timeout=1))             # prints "400"
        #print('*** ', res)
        #
        pi = numpy.mean([r.get() for r in res])
        #pi = numpy.mean(res)
      #
    return pi

#
def pool_wrapper(f, ncpus=1, f_agg=numpy.sum, *args, **kwargs):
    '''
    # An example, and generic wrapper, of how to add optional multiprocessing to a function.
    #  This version uses a simple "if-then" split to either launch a pool or just exectue the function.
    # @f: function to parallelize
    # @ncups: number of processors/processes.
    # @f_agg: aggregation function. We assume a numerical scalar or array is returned.
    # *args, **kwargs: that will be passed to f
    #
    # NOTE: It is tempting to try to write these super generic function handlers, but execution can get
    #  messy since you'll need alignment in *args and **kwargs.
    '''
    #print('** args: ', args)
    #print('** kwargs: ', kwargs)
    if ncpus > 1:
        with mpp.Pool(processes=ncpus) as pool:
            #N = int(-(-N/ncpus))
            res = [pool.apply_async(f, args=args, kwds=kwargs)  for k in range(ncpus)]
            pool.close()
            pool.join()
            #
            #print('*** res: ', [r.get() for r in res])
            return f_agg([numpy.atleast_1d(r.get()) for r in res], axis=0)
    #
    # kf ncpu==1, just execute:
    return f(*args, **kwargs)
    #
#
# One more example: Nesting a Pool() with a pseudo-recursive call. This might be our final
#  example. Benchmarking this will, somewhat disappointingly, suggest that the pickled process
#  foregoes @jit compilation./
#@numba.jit()
def calc_pi(N, ncpus=1):
    '''
    # add a Pool() wrapper to parallelize this code if ncpus>1. Otherwise, just run it.
    #  this is a good(ish?) way to write a function that can be parallelized but is still HPC safe.
    #  whatever that means.
    #
    # NOTE: But we can't numba.jit() this function -- at least not the whole thing. So it is probably better to
    #  define the working, computational code externally (ie, call calc_pi_jit()
    '''
    #
    #f_pi = calc_pi_jit
    f_pi = calc_pi_loop_loop
    # if ncpus>1, launch a Pool() and call self (pseudo-)recursively.
    if ncpus > 1:
        N_mpp = int(-(-N/ncpus))
        with mpp.Pool(processes=ncpus) as pool:
            # call this function, but with ncpus=1
            #res = [pool.apply_async(calc_pi, args=(N_mpp,1,)) for k in range(ncpus)]
            #res = [pool.apply_async(calc_pi, kwds={'N':N_mpp, ncpus:1}) for k in range(ncpus)]
            res = [pool.apply_async(calc_pi_jit, kwds={'N':N_mpp}).get() for k in range(ncpus)]
            #res = [pool.apply(calc_pi_jit, (N_mpp,) ) for k in range(ncpus)]
            pool.close()
            pool.join()
            #
            #return numpy.mean([r.get() for r in res])
            return numpy.mean([r for r in res])
    #
    
#    M = 0
#    for i in range(int(N)):
#        x = random.uniform(-1, 1)
#        y = random.uniform(-1, 1)
#        if x**2 + y**2 < 1:
#            M += 1
#    return 4 * M / N
    return f_pi(N)
    #
#
def calc_pi_n_write(f_pi=calc_pi_jit_np, Nits=int(1E7), fout_name=None):
    '''
    # comput pi and output to fout. We will use this as a simple demonstrator of emb. parallal.
    '''
    pi = f_pi(Nits)
    #
    if not fout_name is None:
        with open(fout_name, 'a') as fout:
            fout.write(f'pi:{pi}\n')
#
if __name__ == '__main__':
    '''
    # Code here will execute when themodule is executed like a program, eg:
    #  $ ./calc_py.py  # if execute is enabled (chmod +x)
    #  $ python calc_py.py
    #  This code will will *not* execute when the code is imported (into another .py module or a notebook).
    '''
    #
    # Give it something to do.
    _dummy_var = None

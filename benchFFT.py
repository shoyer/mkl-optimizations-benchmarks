# References:
#
# http://software.intel.com/en-us/intel-mkl

import time

import numpy
import numpy.fft.fftpack

import matplotlib.pyplot as plt
from matplotlib import ticker


def show_info():
    try:
        import mkl
        print "MKL MAX THREADS:", mkl.get_max_threads()
    except ImportError:
        print "MKL NOT INSTALLED"


def plot_results(datas, factor=None, algo='FFT'):
    xlabel = r'Array Size (2^x)'
    ylabel = 'Speed (GFLOPs)'
    backends = ['numpy', 'numpy+mkl']

    plt.clf()
    fig1, ax1 = plt.subplots()
    plt.figtext(0.90, 0.94, "Note: higher is better", va='top', ha='right')
    w, h = fig1.get_size_inches()
    fig1.set_size_inches(w*1.5, h)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.get_xaxis().set_minor_locator(ticker.NullLocator())
    ax1.set_xticks(datas[0][:,0])
    ax1.grid(color="lightgrey", linestyle="--", linewidth=1, alpha=0.5)
    if factor:
        ax1.set_xticklabels([str(int(x)) for x in datas[0][:,0]/factor])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlim(datas[0][0,0]*.9, datas[0][-1,0]*1.025)
    plt.suptitle("%s Performance" % ("FFT"), fontsize=28)

    for backend, data in zip(backends, datas):
        N = data[:, 0]
        plt.plot(N, data[:, 1], 'o-', linewidth=2, markersize=5, label=backend)
        plt.legend(loc='upper left', fontsize=18)

    plt.savefig(algo + '.png')
    
    
def run(repeat, size, mkl=True):
    args = tuple(1 * [size])
    a = numpy.random.randn(*args) + 1j * numpy.random.randn(*args)
    a = a.astype(numpy.complex64)
    start_time = time.time()
    for dummy in xrange(repeat):
        if mkl: 
            b = numpy.fft.fftn(a)
        else:
            b = numpy.fft.fftpack.fftn(a)
    time_taken = time.time() - start_time
    return time_taken
    
    
def main():  
         
     dataMKL = []
     dataNoMKL = []
     
     print '\n%8s %8s %16s %16s' % ('trials', '2^n', 'time(s) MKL', 'time(s) No MKL')
     print ( '----------------------------'*2)
     for n in xrange(4, 25):
         size = 2 ** n
         # to keep the experiment from taking too long
         if n < 10:
             trials = 1000
         elif n < 20:
             trials = 100
         else:
             trials = 10
             
         mflop = 5.0*size*numpy.log2(size)    
         gglop = mflop / 1000
         
         s = run(trials, size)
         avg_ms = (s/trials) * 1000000
         dataMKL.append(numpy.asarray([n, gglop/avg_ms ]))    
       
         s2 = run(trials, size, mkl=False)
         avg_ms = (s2/trials) * 1000000
         dataNoMKL.append(numpy.asarray([n, gglop/avg_ms ]))
         print '%8i %8i %12.4fs %12.4fs' % (trials, n, s, s2)
         
     datas = numpy.asarray([numpy.asarray(dataNoMKL),numpy.asarray(dataMKL)])
     
     plot_results(datas,algo='FFT')
 


if __name__ == '__main__':
    main()

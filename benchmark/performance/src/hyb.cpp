#include <iostream>
#include <string>

#include <cusp/io/matrix_market.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>

// This block enables to compile the code with and without the likwid header in place
#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cerr << "Please specify the input matrix file." << std::endl;
		exit(-1);
    }

    std::string fin = argv[1];

    cusp::hyb_matrix<int, double, cusp::host_memory> A;

    cusp::io::read_matrix_market_file(A, fin);

    cusp::random_array<double> r(A.num_rows, 0);
    cusp::array1d<double, cusp::host_memory> x(r.begin(), r.end()), y(A.num_rows);

    
    LIKWID_MARKER_INIT; 
    LIKWID_MARKER_THREADINIT;
    LIKWID_MARKER_REGISTER("Multiply");

    LIKWID_MARKER_START("Multiply");
    cusp::multiply(A, x, y);
    LIKWID_MARKER_STOP("Multiply");

    LIKWID_MARKER_CLOSE;

    return 0;
}
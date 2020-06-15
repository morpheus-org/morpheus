#include <iostream>
#include <string>
#include <chrono>

#include <morpheus/dynamic/matrix.hpp>
#include <morpheus/dynamic/print.hpp>
#include <morpheus/dynamic/io/matrix_market.hpp>
#include <morpheus/dynamic/multiply.hpp>

#include <morpheus/coo_matrix.hpp>
#include <morpheus/csr_matrix.hpp>
#include <morpheus/dense_matrix.hpp>
#include <morpheus/dense_vector.hpp>
#include <morpheus/memory.hpp>

#include <boost/mpl/vector.hpp>


int main ( )
{
	using host = morpheus::host_memory;
	using Coo_matrix = morpheus::coo_matrix<int, float, host>;
	using Csr_matrix = morpheus::csr_matrix<int, float, host>;
	using Dense_vector = morpheus::dense_vector<float, host>;
	using Dense_matrix = morpheus::dense_matrix<float, host>;


	using matrix_t = boost::mpl::vector<Coo_matrix, Csr_matrix, Dense_matrix>;
	using Matrix = morpheus::matrix<matrix_t>;

	std::string fin = "/Users/cstyl/Desktop/Projects/morpheus/matrix/fidap001.mtx";

	const int count = 20;

	// CUSP
	{
		std:: cout << "\n\nCUSP Timing..." << std::endl;
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		std::chrono::time_point<std::chrono::high_resolution_clock> mstart, mend;
		std::chrono::time_point<std::chrono::high_resolution_clock> iostart, ioend;

		std::chrono::duration<double> elapsedTime, melapsedTime, ioelapsedTime;

		double seconds = 0.0, ioseconds = 0.0, mseconds = 0.0;

		start = std::chrono::high_resolution_clock::now();

		cusp::coo_matrix<int, float, cusp::host_memory> A;

		iostart = std::chrono::high_resolution_clock::now();

		cusp::io::read_matrix_market_file(A, fin);

		ioend = std::chrono::high_resolution_clock::now();
		ioelapsedTime = ioend - iostart;
		ioseconds += ioelapsedTime.count();

		std::cout << "Rows = " << A.num_rows << " ";
		std::cout << "Cols = " << A.num_cols << " ";
		std::cout << "Nnz = " << A.num_entries << std::endl;

		cusp::random_array<float> r(A.num_rows, 0);

		cusp::array1d<float, cusp::host_memory> x(r.begin(), r.end());
		cusp::array1d<float, cusp::host_memory> y(A.num_rows);

		for(int i = 0; i < count; i++)
		{
			mstart = std::chrono::high_resolution_clock::now();

			cusp::multiply(A, x, y);
			x = y;

			mend = std::chrono::high_resolution_clock::now();
			melapsedTime = mend - mstart;
			mseconds += melapsedTime.count();
		}

		end = std::chrono::high_resolution_clock::now();
		elapsedTime = end - start;
		seconds += elapsedTime.count();

		std::cout << "Timing:" << std::endl;
		std::cout << "\tOverall = " << seconds;
		std::cout << "\tIO = " << ioseconds;
		std::cout << "\tspMv = " << mseconds;

	}

	// STATIC
	{
		std:: cout << "\n\nSTATIC Timing..." << std::endl;
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		std::chrono::time_point<std::chrono::high_resolution_clock> mstart, mend;
		std::chrono::time_point<std::chrono::high_resolution_clock> iostart, ioend;

		std::chrono::duration<double> elapsedTime, melapsedTime, ioelapsedTime;

		double seconds = 0.0, ioseconds = 0.0, mseconds = 0.0;

		start = std::chrono::high_resolution_clock::now();

		Coo_matrix A;

		iostart = std::chrono::high_resolution_clock::now();

		morpheus::io::read_matrix_market_file(A, fin);

		ioend = std::chrono::high_resolution_clock::now();
		ioelapsedTime = ioend - iostart;
		ioseconds += ioelapsedTime.count();

		std::cout << "Rows = " << A.nrows() << " ";
		std::cout << "Cols = " << A.ncols() << " ";
		std::cout << "Nnz = " << A.nnz() << std::endl;

		cusp::random_array<float> r(A.nrows(), 0);

		Dense_vector x(r.begin(), r.end());
		Dense_vector y(A.nrows());

		for(int i = 0; i < count; i++)
		{
			mstart = std::chrono::high_resolution_clock::now();

			morpheus::multiply(A, x, y);

			mend = std::chrono::high_resolution_clock::now();
			melapsedTime = mend - mstart;
			mseconds += melapsedTime.count();
		}

		end = std::chrono::high_resolution_clock::now();
		elapsedTime = end - start;
		seconds += elapsedTime.count();

		std::cout << "Timing:" << std::endl;
		std::cout << "\tOverall = " << seconds;
		std::cout << "\tIO = " << ioseconds;
		std::cout << "\tspMv = " << mseconds;

	}

	// DYNAMIC
	{
		std:: cout << "\n\nDYNAMIC Timing..." << std::endl;
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		std::chrono::time_point<std::chrono::high_resolution_clock> mstart, mend;
		std::chrono::time_point<std::chrono::high_resolution_clock> iostart, ioend;

		std::chrono::duration<double> elapsedTime, melapsedTime, ioelapsedTime;

		double seconds = 0.0, ioseconds = 0.0, mseconds = 0.0;

		start = std::chrono::high_resolution_clock::now();

		Matrix A;

		iostart = std::chrono::high_resolution_clock::now();

		morpheus::io::read_matrix_market_file(A, fin);

		ioend = std::chrono::high_resolution_clock::now();
		ioelapsedTime = ioend - iostart;
		ioseconds += ioelapsedTime.count();

		std::cout << "Rows = " << A.nrows() << " ";
		std::cout << "Cols = " << A.ncols() << " ";
		std::cout << "Nnz = " << A.nnz() << std::endl;

		cusp::random_array<float> r(A.nrows(), 0);

		Dense_vector x(r.begin(), r.end());
		Dense_vector y(A.nrows());

		for(int i = 0; i < count; i++)
		{
			mstart = std::chrono::high_resolution_clock::now();

			morpheus::multiply(A, x, y);

			mend = std::chrono::high_resolution_clock::now();
			melapsedTime = mend - mstart;
			mseconds += melapsedTime.count();
		}

		end = std::chrono::high_resolution_clock::now();
		elapsedTime = end - start;
		seconds += elapsedTime.count();

		std::cout << "Timing:" << std::endl;
		std::cout << "\tOverall = " << seconds;
		std::cout << "\tIO = " << ioseconds;
		std::cout << "\tspMv = " << mseconds;

	}


	return 0;
}

#include <iostream>
#include <morpheus/containers/dynamic_matrix.hpp>

int main()
{
    Morpheus::DynamicMatrix<int,double> A;

    std::cout << "A.name(): " << A.name() << std::endl;
    std::cout << "A.active_name(): " << A.active_name() << std::endl;
    std::cout << "A.active_index(): " << A.active_index() << std::endl;
    A.resize(5, 10, 15);
    
    try{
        A.resize(5, 10, 15, 20);
    }catch(const std::runtime_error& e)
    {
        std::cout << e.what();
    }

    A = Morpheus::CsrMatrix<int,double>();
    std::cout << "A.active_name(): " << A.active_name() << std::endl;
    std::cout << "A.active_index(): " << A.active_index() << std::endl;
    A.resize(5, 10, 15);

    A = Morpheus::DiaMatrix<int,double>();
    std::cout << "A.active_name(): " << A.active_name() << std::endl;
    std::cout << "A.active_index(): " << A.active_index() << std::endl;
    A.resize(5, 10, 15, 20);
    A.resize(5, 10, 15, 20, 35);

    return 0;
}
#include <iostream>
#include <morpheus/containers/dynamic_matrix.hpp>

template <typename... Properties>
void stats(Morpheus::DynamicMatrix<Properties...>& mat, std::string name,
           std::string fn_name) {
  std::cout << name << "." << fn_name << std::endl;
  std::cout << name << ".name(): " << mat.name() << std::endl;
  std::cout << name << ".active_name(): " << mat.active_name() << std::endl;
  std::cout << name << ".active_index(): " << mat.active_index() << std::endl;
  std::cout << std::endl;
}

int main() {
  Morpheus::DynamicMatrix<double, int> A;

  stats(A, "A", "resize(5, 10, 15)");
  A.resize(5, 10, 15);

  try {
    A.resize(5, 10, 15, 20);
  } catch (std::runtime_error& e) {
    std::cerr << "Exception Raised:: " << e.what() << std::endl;
  }
  stats(A, "A", "resize(5, 10, 15, 20)");

  A = Morpheus::CsrMatrix<double, int>();
  stats(A, "A", "resize(5, 10, 15)");
  A.resize(5, 10, 15);

  A = Morpheus::DiaMatrix<double, int>();
  stats(A, "A", "resize(5, 10, 15, 20)");
  A.resize(5, 10, 15, 20);
  A.resize(5, 10, 15, 20, 35);
  stats(A, "A", "resize(5, 10, 15, 20, 35)");

  return 0;
}
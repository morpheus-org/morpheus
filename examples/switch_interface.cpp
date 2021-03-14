#include <iostream>
#include <morpheus/containers/dynamic_matrix.hpp>

int main() {
  Morpheus::DynamicMatrix<int, double> A;
  std::cout << "Active Index: " << A.active_index() << std::endl;

  // Switch through available enums
  A.activate(Morpheus::CSR_FORMAT);
  std::cout << "Active Index: " << A.active_index() << std::endl;

  A.activate(Morpheus::DIA_FORMAT);
  std::cout << "Active Index: " << A.active_index() << std::endl;

  // Switch using integer indexing
  A.activate(0);
  std::cout << "Active Index: " << A.active_index() << std::endl;

  // In case indexing exceeds the size of available types in the underlying
  // variant, we default to the first entry
  A.activate(5);
  std::cout << "Active Index: " << A.active_index() << std::endl;

  return 0;
}
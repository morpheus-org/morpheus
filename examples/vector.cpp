/**
 * vector.cpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
 *
 * Contributing Authors:
 * Christodoulos Stylianou (c.stylianou@ed.ac.uk)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>

#include <morpheus/core/core.hpp>
#include <morpheus/containers/vector.hpp>

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);

  using vector = Morpheus::DenseVector<double>;
  std::cout << "Morpheus::DenseVector is same with Morpheus::vector:"
            << (std::is_same<vector, Morpheus::vector<double>>::value ? "true"
                                                                      : "false")
            << std::endl;
  {
    std::cout << "Default constructor:" << std::endl;
    vector x;

    std::cout << "\tx.size = " << x.size() << std::endl;
  }

  {
    std::cout << "Initializing with value:" << std::endl;
    vector x(5, 1.0);

    std::cout << "\tx.size = " << x.size() << std::endl;
    std::cout << "\tx(2) = " << x(2) << std::endl;
    std::cout << "\tx[3] = " << x[3] << std::endl;
  }

  {
    std::cout << "Checking iterators:" << std::endl;
    vector x(6, 1.0);

    x(0) = 1.2;
    x(1) = 1.3;
    x(2) = 1.4;
    x(3) = 1.5;
    x(4) = 1.6;
    x(5) = 1.7;

    for (auto it = x.begin(); it != x.end(); ++it) {
      auto i = std::distance(x.begin(), it);
      std::cout << "\tx(" << i << ") = " << *it << std::endl;
    }
  }

  //   {
  //     std::cout << "Checking reserve:" << std::endl;
  //     vector x;

  //     std::cout << "\tx.size = " << x.size() << std::endl;
  //     x.reserve(10);
  //     std::cout << "\tx.reserve(10): x.size = " << x.size() << std::endl;

  //     x(2) = 2.2;
  //     std::cout << "\tx.reserve(10): x.size = " << x.size() << std::endl;
  //     std::cout << "\tx.reserve(10): x(2) = " << x(2) << std::endl;
  //   }

  {
    std::cout << "Checking copy constructor(shallow copy):" << std::endl;
    vector x(6, 1.0);
    x(0) = 1.2;
    x(1) = 1.3;
    x(2) = 1.4;
    x(3) = 1.5;
    x(4) = 1.6;
    x(5) = 1.7;
    vector y(x);
    for (auto it = y.begin(); it != y.end(); ++it) {
      auto i = std::distance(y.begin(), it);
      std::cout << "\ty(" << i << ") = " << *it << std::endl;
    }

    y(4) = 35.13;

    for (auto it = x.begin(); it != x.end(); ++it) {
      auto i = std::distance(x.begin(), it);
      std::cout << "\tx(" << i << ") = " << *it << std::endl;
    }
  }

  {
    std::cout << "Checking copy assignment(shallow copy):" << std::endl;
    vector x(6, 1.0);
    x(0)     = 1.2;
    x(1)     = 1.3;
    x(2)     = 1.4;
    x(3)     = 1.5;
    x(4)     = 1.6;
    x(5)     = 1.7;
    vector y = x;
    for (auto it = y.begin(); it != y.end(); ++it) {
      auto i = std::distance(y.begin(), it);
      std::cout << "\ty(" << i << ") = " << *it << std::endl;
    }

    y(4) = 35.13;

    for (auto it = x.begin(); it != x.end(); ++it) {
      auto i = std::distance(x.begin(), it);
      std::cout << "\tx(" << i << ") = " << *it << std::endl;
    }
  }

  {
    std::cout << "Checking move assignment(shallow copy):" << std::endl;
    vector x(6, 1.0);
    x(0)     = 1.2;
    x(1)     = 1.3;
    x(2)     = 1.4;
    x(3)     = 1.5;
    x(4)     = 1.6;
    x(5)     = 1.7;
    vector y = std::move(x);
    for (auto it = y.begin(); it != y.end(); ++it) {
      auto i = std::distance(y.begin(), it);
      std::cout << "\ty(" << i << ") = " << *it << std::endl;
    }

    y(4) = 35.13;

    for (auto it = x.begin(); it != x.end(); ++it) {
      auto i = std::distance(x.begin(), it);
      std::cout << "\tx(" << i << ") = " << *it << std::endl;
    }
  }

  {
    std::cout << "Checking resizing:" << std::endl;
    vector x(5, 2.0);

    std::cout << "\tx.size = " << x.size() << std::endl;
    x.resize(15);
    std::cout << "\tx.resize(15):: x.size = " << x.size() << std::endl;
  }

  {
    std::cout << "Checking resizing with new value:" << std::endl;
    vector x(5, 2.0);

    std::cout << "\tx.size = " << x.size() << std::endl;
    x.resize(6, 3.0);
    std::cout << "\tx.resize(15,3.0):: x.size = " << x.size() << std::endl;
    for (auto it = x.begin(); it != x.end(); ++it) {
      auto i = std::distance(x.begin(), it);
      std::cout << "\tx(" << i << ") = " << *it << std::endl;
    }
  }

  {
    std::cout << "Checking resizing to smaller size:" << std::endl;
    vector x(5, 2.0);

    std::cout << "\tx.size = " << x.size() << std::endl;
    x.resize(2);
    std::cout << "\tx.resize(2):: x.size = " << x.size() << std::endl;
    for (auto it = x.begin(); it != x.end(); ++it) {
      auto i = std::distance(x.begin(), it);
      std::cout << "\tx(" << i << ") = " << *it << std::endl;
    }
    // std::cout << "\tx(4) = " << x(4) << std::endl;
  }

  Morpheus::finalize();

  return 0;
}
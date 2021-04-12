#ifndef MORPHEUS_CORE_CORE_HPP
#define MORPHEUS_CORE_CORE_HPP

#include <iostream>

namespace Morpheus {

struct InitArguments;

void initialize(int& argc, char* argv[]);
void initialize(InitArguments args);
void print_configuration(std::ostream& out, const bool detail);
void finalize();
}  // namespace Morpheus

#include <morpheus/core/impl/core.inl>

#endif  // MORPHEUS_CORE_CORE_HPP
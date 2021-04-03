#ifndef MORPHEUS_CORE_IMPL_CORE_INL
#define MORPHEUS_CORE_IMPL_CORE_INL

#include <Kokkos_Core.hpp>
#include <morpheus/version.hpp>

namespace Morpheus {

struct InitArguments : public Kokkos::InitArguments {
  InitArguments(int nt = -1, int nn = -1, int dv = -1, bool dw = false,
                bool ti = false)
      : Kokkos::InitArguments(nt, nn, dv, dw, ti) {}
};

void initialize(int& argc, char* argv[]) { Kokkos::initialize(argc, argv); }

void print_configuration(std::ostream& out, const bool detail = true) {
  std::ostringstream msg;

  msg << "Morpheus Version:" << std::endl;
  msg << "  " << MORPHEUS_MAJOR_VERSION << "." << MORPHEUS_MINOR_VERSION << "."
      << MORPHEUS_PATCH_VERSION << std::endl;

  Kokkos::print_configuration(msg, detail);

  if (detail == true) {
    msg << "Default Host Execution Space Configuration:" << std::endl;
    Kokkos::DefaultHostExecutionSpace::print_configuration(msg, detail);
    msg << "Default Execution Space Configuration:" << std::endl;
    Kokkos::DefaultExecutionSpace::print_configuration(msg, detail);
  }

  out << msg.str() << std::endl;
}
void initialize(InitArguments args = InitArguments()) {
  Kokkos::initialize(args);
}

void finalize() { Kokkos::finalize(); }

}  // namespace Morpheus

#endif  // MORPHEUS_CORE_IMPL_CORE_INL
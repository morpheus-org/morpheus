/**
 * Morpheus_Core.cpp
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

#include <Morpheus_Core.hpp>

#include <iostream>

namespace Morpheus {
namespace Impl {

bool is_unsigned_int(const char* str) {
  const size_t len = strlen(str);
  for (size_t i = 0; i < len; ++i) {
    if (!isdigit(str[i])) {
      return false;
    }
  }
  return true;
}

bool check_arg(char const* arg, char const* expected) {
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  if (arg_len < exp_len) return false;
  if (std::strncmp(arg, expected, exp_len) != 0) return false;
  if (arg_len == exp_len) return true;

  if (std::isalnum(arg[exp_len]) || arg[exp_len] == '-' ||
      arg[exp_len] == '_') {
    return false;
  }
  return true;
}

bool check_int_arg(char const* arg, char const* expected, int* value) {
  if (!check_arg(arg, expected)) return false;
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  bool okay           = true;
  if (arg_len == exp_len || arg[exp_len] != '=') okay = false;
  char const* number = arg + exp_len + 1;
  if (!Impl::is_unsigned_int(number) || strlen(number) == 0) okay = false;
  *value = std::stoi(number);
  if (!okay) {
    std::ostringstream ss;
    ss << "Error: expecting an '=INT' after command line argument '" << expected
       << "'";
    ss << ". Raised by Morpheus::initialize(int argc, char* argv[]).";
    throw Morpheus::RuntimeException(ss.str());
  }
  return true;
}

bool check_str_arg(char const* arg, char const* expected, std::string& value) {
  if (!check_arg(arg, expected)) return false;
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  bool okay           = true;
  if (arg_len == exp_len || arg[exp_len] != '=') okay = false;
  char const* remain = arg + exp_len + 1;
  value              = remain;
  if (!okay) {
    std::ostringstream ss;
    ss << "Error: expecting an '=STRING' after command line argument '"
       << expected << "'";
    ss << ". Raised by Morpheus::initialize(int argc, char* argv[]).";
    throw Morpheus::RuntimeException(ss.str());
  }
  return true;
}

void parse_command_line_arguments(int& argc, char* argv[],
                                  InitArguments& arguments) {
  auto& dynamic_format = arguments.dynamic_format;

  bool morpheus_format_found = false;

  int iarg = 0;

  while (iarg < argc) {
    if (Impl::check_int_arg(argv[iarg], "--morpheus-format", &dynamic_format)) {
      for (int k = iarg; k < argc - 1; k++) {
        argv[k] = argv[k + 1];
      }
      morpheus_format_found = true;
      argc--;
    } else if (!morpheus_format_found &&
               Impl::check_int_arg(argv[iarg], "--format", &dynamic_format)) {
      iarg++;
    } else if (check_arg(argv[iarg], "--morpheus-help") ||
               check_arg(argv[iarg], "--help")) {
      auto const help_message = R"(
      --------------------------------------------------------------------------------
      ------------------------Morpheus command line arguments-------------------------
      --------------------------------------------------------------------------------
      The following arguments exist also without prefix 'morpheus' (e.g. --help).
      The prefixed arguments will be removed from the list by Morpheus::initialize(),
      the non-prefixed ones are not removed. Prefixed versions take precedence over
      non prefixed ones, and the last occurrence of an argument overwrites prior
      settings.
      --morpheus-help             : print this message
      --morpheus-format=INT       : specify which format to activate when the 
                                    dynamic matrix is used.
      --------------------------------------------------------------------------------
)";
      std::cout << help_message << std::endl;

      // Remove the --morpheus-help argument from the list but leave --help
      if (check_arg(argv[iarg], "--morpheus-help")) {
        for (int k = iarg; k < argc - 1; k++) {
          argv[k] = argv[k + 1];
        }
        argc--;
      } else {
        iarg++;
      }
    } else
      iarg++;
  }
}

}  // namespace Impl

void initialize(int& argc, char* argv[]) {
  InitArguments arguments;
  Impl::parse_command_line_arguments(argc, argv, arguments);
  Kokkos::initialize(argc, argv);
}

void initialize(int& argc, char* argv[], InitArguments& arguments) {
  Impl::parse_command_line_arguments(argc, argv, arguments);
  Kokkos::initialize(argc, argv);
}

void print_configuration(std::ostream& out, const bool detail) {
  std::ostringstream msg;

  msg << "Morpheus Version:" << std::endl;
  msg << "  " << Morpheus_VERSION_MAJOR << "." << Morpheus_VERSION_MINOR << "."
      << Morpheus_VERSION_PATCH << " (" << MORPHEUS_VERSION << ")" << std::endl;

  Kokkos::print_configuration(msg, detail);

  if (detail == true) {
    msg << "Default Host Execution Space Configuration:" << std::endl;
    Kokkos::DefaultHostExecutionSpace::print_configuration(msg, detail);
    msg << "Default Execution Space Configuration:" << std::endl;
    Kokkos::DefaultExecutionSpace::print_configuration(msg, detail);
  }

  out << msg.str() << std::endl;
}
void initialize(InitArguments args) { Kokkos::initialize(args); }

void finalize() { Kokkos::finalize(); }

}  // namespace Morpheus
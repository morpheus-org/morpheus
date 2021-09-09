/**
 * Morpheus_Scan.hpp
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

#ifndef MORPHEUS_SCAN_HPP
#define MORPHEUS_SCAN_HPP

#include <Morpheus_AlgorithmTags.hpp>
#include <impl/Morpheus_Scan_Impl.hpp>

namespace Morpheus {

/*
 * Computes an inclusive prefix sum operation. The term 'inclusive'
 * means that each result includes the corresponding input operand
 * in the partial sum. When the input and output sequences are the
 * same, the scan is performed in-place.
 *
 *  \code
 *  #include <Morpheus_Core.hpp>
 *
 *  using vec    = Morpheus::DenseVector<int, int, Kokkos::HostSpace>;
 *  using serial = Kokkos::Serial;
 *
 *  typename vec::index_type N = 10;
 *  vec in(N, 1), out(N, 0);
 *
 *  Morpheus::inclusive_scan<serial>(in, out, N);
 *
 *  // out is now {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  \endcode
 *
 */
template <typename ExecSpace, typename Algorithm, typename Vector>
void inclusive_scan(const Vector& in, Vector& out,
                    typename Vector::index_type size,
                    typename Vector::index_type start) {
  Impl::inclusive_scan<ExecSpace>(in, out, size, start, typename Vector::tag{},
                                  typename Vector::tag{}, Algorithm{});
}

template <typename ExecSpace, typename Vector>
void inclusive_scan(const Vector& in, Vector& out,
                    typename Vector::index_type size,
                    typename Vector::index_type start) {
  Impl::inclusive_scan<ExecSpace>(in, out, size, start, typename Vector::tag{},
                                  typename Vector::tag{}, Alg0{});
}

/*
 * Computes an exclusive prefix sum operation. The term 'exclusive' means
 * that each result does not include the corresponding input operand in the
 * partial sum. More precisely, 0 is assigned to out[0] and the sum of
 * 0 and in[0] is assigned to out[1], and so on. When the input and output
 * sequences are the same, the scan is performed in-place.
 *
 *  \code
 *  #include <Morpheus_Core.hpp>
 *
 *  using vec    = Morpheus::DenseVector<int, int, Kokkos::HostSpace>;
 *  using serial = Kokkos::Serial;
 *
 *  typename vec::index_type N = 10;
 *  vec in(N, 1), out(N, 0);
 *
 *  Morpheus::exclusive_scan<serial>(in, out, N);
 *
 *  // out is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
 *  \endcode
 *
 */

template <typename ExecSpace, typename Algorithm, typename Vector>
void exclusive_scan(const Vector& in, Vector& out,
                    typename Vector::index_type size,
                    typename Vector::index_type start) {
  Impl::exclusive_scan<ExecSpace>(in, out, size, start, typename Vector::tag{},
                                  typename Vector::tag{}, Algorithm{});
}

template <typename ExecSpace, typename Vector>
void exclusive_scan(const Vector& in, Vector& out,
                    typename Vector::index_type size,
                    typename Vector::index_type start) {
  Impl::exclusive_scan<ExecSpace>(in, out, size, start, typename Vector::tag{},
                                  typename Vector::tag{}, Alg0{});
}

/*
 * Computes an inclusive key-value or 'segmented' prefix sum operation.
 * The term 'inclusive' means that each result includes the corresponding
 * input operand in the partial sum. The term 'segmented' means that the
 * partial sums are broken into distinct segments. In other words, within
 * each segment a separate inclusive scan operation is computed.
 *
 *  \code
 *  #include <Morpheus_Core.hpp>
 *
 *  using vec    = Morpheus::DenseVector<int, int, Kokkos::HostSpace>;
 *  using serial = Kokkos::Serial;
 *
 *  typename vec::index_type N = 10;
 *  vec in(N, 1), out(N, 0), keys(N);
 *
 *  keys[0] = 0; keys[1] = 0; keys[2] = 0;
 *  keys[3] = 1; keys[4] = 1;
 *  keys[5] = 2;
 *  keys[6] = 3; keys[7] = 3; keys[8] = 3; keys[9] = 3;
 *
 *  Morpheus::inclusive_scan_by_key<serial>(keys, in, out, N);
 *
 *  // out is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 */
template <typename ExecSpace, typename Algorithm, typename Vector1,
          typename Vector2>
void inclusive_scan_by_key(const Vector1& keys, const Vector2& in, Vector2& out,
                           typename Vector2::index_type size,
                           typename Vector2::index_type start) {
  Impl::inclusive_scan_by_key<ExecSpace>(
      in, out, size, start, typename Vector1::tag{}, typename Vector2::tag{},
      typename Vector2::tag{}, Algorithm{});
}

template <typename ExecSpace, typename Vector1, typename Vector2>
void inclusive_scan_by_key(const Vector1& keys, const Vector2& in, Vector2& out,
                           typename Vector2::index_type size,
                           typename Vector2::index_type start) {
  Impl::inclusive_scan_by_key<ExecSpace>(
      keys, in, out, size, start, typename Vector1::tag{},
      typename Vector2::tag{}, typename Vector2::tag{}, Alg0{});
}

/*
 * \p Computes an exclusive key-value or 'segmented' prefix sum operation. The
 * term 'exclusive' means that each result does not include the corresponding
 * input operand in the partial sum. The term 'segmented' means that the partial
 * sums are broken into distinct segments. In other words, within each segment a
 * separate exclusive scan operation is computed. The following code snippet
 * demonstrates how to use \p exclusive_scan_by_key.
 *
 *  \code
 *  #include <Morpheus_Core.hpp>
 *
 *  using vec    = Morpheus::DenseVector<int, int, Kokkos::HostSpace>;
 *  using serial = Kokkos::Serial;
 *
 *  typename vec::index_type N = 10;
 *  vec in(N, 1), out(N, 0), keys(N);
 *
 *  keys[0] = 0; keys[1] = 0; keys[2] = 0;
 *  keys[3] = 1; keys[4] = 1;
 *  keys[5] = 2;
 *  keys[6] = 3; keys[7] = 3; keys[8] = 3; keys[9] = 3;
 *
 *  Morpheus::exclusive_scan_by_key<serial>(keys, in, out, N);
 *
 *  // out is now {0, 1, 2, 0, 1, 0, 0, 1, 2, 3};
 *  \endcode
 *
 */
template <typename ExecSpace, typename Algorithm, typename Vector1,
          typename Vector2>
void exclusive_scan_by_key(const Vector1& keys, const Vector2& in, Vector2& out,
                           typename Vector2::index_type size,
                           typename Vector2::index_type start) {
  Impl::exclusive_scan_by_key<ExecSpace>(
      in, out, size, start, typename Vector1::tag{}, typename Vector2::tag{},
      typename Vector2::tag{}, Algorithm{});
}

template <typename ExecSpace, typename Vector1, typename Vector2>
void exclusive_scan_by_key(const Vector1& keys, const Vector2& in, Vector2& out,
                           typename Vector2::index_type size,
                           typename Vector2::index_type start) {
  Impl::exclusive_scan_by_key<ExecSpace>(
      keys, in, out, size, start, typename Vector1::tag{},
      typename Vector2::tag{}, typename Vector2::tag{}, Alg0{});
}

}  // namespace Morpheus

#endif  // MORPHEUS_SCAN_HPP
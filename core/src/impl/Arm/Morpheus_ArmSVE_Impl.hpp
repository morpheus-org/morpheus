/**
 * Morpheus_ArmSVE_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
 *
 * Contributing Authors:
 * Ricardo Jesus (rjj@ed.ac.uk)
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

#ifndef MORPHEUS_ARM_ARMSVE_IMPL_HPP
#define MORPHEUS_ARM_ARMSVE_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
#if defined(MORPHEUS_ENABLE_ARM_SVE)

#include <arm_sve.h>

using vbool_t = svbool_t;

// type for uints of sizeof(value_type)-bytes
template <size_t I>
struct uint_custom {
  static_assert((I == 8) || (I == 16) || (I == 32) || (I == 64),
                "Size must be 8, 16, 32 or 64.");
  using type = typename std::conditional_t<
      I == 64, uint64_t,
      typename std::conditional_t<
          I == 32, uint32_t,
          typename std::conditional_t<I == 16, uint16_t, uint8_t>>>;
};

// idx_t -> uint_t
template <size_t I>
using uint_t = typename uint_custom<I>::type;

// type for SVE vectors of uints of sizeof(value_type)-bytes
template <size_t I = 64>
struct svuint_custom {
  static_assert((I == 8) || (I == 16) || (I == 32) || (I == 64),
                "Size must be 8, 16, 32 or 64.");
  using type = typename std::conditional_t<
      I == 64, svuint64_t,
      typename std::conditional_t<
          I == 32, svuint32_t,
          typename std::conditional_t<I == 16, svuint16_t, svuint8_t>>>;
};

// vidx_t -> svuint_t
template <size_t I>
using svuint_t = typename svuint_custom<I>::type;

template <size_t I = 64>
struct svint_custom {
  static_assert((I == 8) || (I == 16) || (I == 32) || (I == 64),
                "Size must be 8, 16, 32 or 64.");
  using type = typename std::conditional_t<
      I == 64, svint64_t,
      typename std::conditional_t<
          I == 32, svint32_t,
          typename std::conditional_t<I == 16, svint16_t, svint8_t>>>;
};

template <size_t I>
using svint_t = typename svint_custom<I>::type;

// type for SVE vectors of `value_type` type
template <size_t I = 64>
struct svfloat_custom {
  static_assert((I == 16) || (I == 32) || (I == 64),
                "Size must be 16, 32 or 64.");
  using type = typename std::conditional_t<
      I == 64, svfloat64_t,
      typename std::conditional_t<I == 32, svfloat32_t, svfloat16_t>>;
};

template <size_t I>
using vtype_t = typename svfloat_custom<I>::type;

// SVE sz-dependent routines
template <size_t I>
uint64_t vcnt(typename std::enable_if_t<I == 64>* = nullptr) {
  return svcntd();
}

template <size_t I>
uint64_t vcnt(typename std::enable_if_t<I == 32>* = nullptr) {
  return svcntw();
}

template <size_t I>
uint64_t vcnt(typename std::enable_if_t<I == 16>* = nullptr) {
  return svcnth();
}

template <size_t I>
uint64_t vcnt(typename std::enable_if_t<I == 8>* = nullptr) {
  return svcntb();
}

template <size_t I>
vbool_t vptrue(typename std::enable_if_t<I == 64>* = nullptr) {
  return svptrue_b64();
}

template <size_t I>
vbool_t vptrue(typename std::enable_if_t<I == 32>* = nullptr) {
  return svptrue_b32();
}

template <size_t I>
vbool_t vptrue(typename std::enable_if_t<I == 16>* = nullptr) {
  return svptrue_b16();
}

template <size_t I>
vbool_t vptrue(typename std::enable_if_t<I == 8>* = nullptr) {
  return svptrue_b8();
}

template <size_t I>
vbool_t vwhilelt(int32_t op1, int32_t op2,
                 typename std::enable_if_t<I == 64>* = nullptr) {
  return svwhilelt_b64_s32(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(int64_t op1, int64_t op2,
                 typename std::enable_if_t<I == 64>* = nullptr) {
  return svwhilelt_b64_s64(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(uint32_t op1, uint32_t op2,
                 typename std::enable_if_t<I == 64>* = nullptr) {
  return svwhilelt_b64_u32(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(uint64_t op1, uint64_t op2,
                 typename std::enable_if_t<I == 64>* = nullptr) {
  return svwhilelt_b64_u64(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(int32_t op1, int32_t op2,
                 typename std::enable_if_t<I == 32>* = nullptr) {
  return svwhilelt_b32_s32(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(int64_t op1, int64_t op2,
                 typename std::enable_if_t<I == 32>* = nullptr) {
  return svwhilelt_b32_s64(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(uint32_t op1, uint32_t op2,
                 typename std::enable_if_t<I == 32>* = nullptr) {
  return svwhilelt_b32_u32(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(uint64_t op1, uint64_t op2,
                 typename std::enable_if_t<I == 32>* = nullptr) {
  return svwhilelt_b32_u64(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(int32_t op1, int32_t op2,
                 typename std::enable_if_t<I == 16>* = nullptr) {
  return svwhilelt_b16_s32(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(int64_t op1, int64_t op2,
                 typename std::enable_if_t<I == 16>* = nullptr) {
  return svwhilelt_b16_s64(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(uint32_t op1, uint32_t op2,
                 typename std::enable_if_t<I == 16>* = nullptr) {
  return svwhilelt_b16_u32(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(uint64_t op1, uint64_t op2,
                 typename std::enable_if_t<I == 16>* = nullptr) {
  return svwhilelt_b16_u64(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(int32_t op1, int32_t op2,
                 typename std::enable_if_t<I == 8>* = nullptr) {
  return svwhilelt_b8_s32(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(int64_t op1, int64_t op2,
                 typename std::enable_if_t<I == 8>* = nullptr) {
  return svwhilelt_b8_s64(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(uint32_t op1, uint32_t op2,
                 typename std::enable_if_t<I == 8>* = nullptr) {
  return svwhilelt_b8_u32(op1, op2);
}

template <size_t I>
vbool_t vwhilelt(uint64_t op1, uint64_t op2,
                 typename std::enable_if_t<I == 8>* = nullptr) {
  return svwhilelt_b8_u64(op1, op2);
}

template <size_t I>
svuint_t<I> vindex(uint_t<I> base, uint_t<I> step,
                   typename std::enable_if_t<I == 64>* = nullptr) {
  return svindex_u64(base, step);
}

template <size_t I>
svuint_t<I> vindex(uint_t<I> base, uint_t<I> step,
                   typename std::enable_if_t<I == 32>* = nullptr) {
  return svindex_u32(base, step);
}

template <size_t I>
svuint_t<I> vindex(uint_t<I> base, uint_t<I> step,
                   typename std::enable_if_t<I == 16>* = nullptr) {
  return svindex_u16(base, step);
}

template <size_t I>
svuint_t<I> vindex(uint_t<I> base, uint_t<I> step,
                   typename std::enable_if_t<I == 8>* = nullptr) {
  return svindex_u8(base, step);
}

template <size_t I>
vtype_t<I> vdup(float64_t op, typename std::enable_if_t<I == 64>* = nullptr) {
  return svdup_f64(op);
}

template <size_t I>
vtype_t<I> vdup(float32_t op, typename std::enable_if_t<I == 32>* = nullptr) {
  return svdup_f32(op);
}

template <size_t I>
uint64_t vcntp(vbool_t pg, vbool_t op,
               typename std::enable_if_t<I == 64>* = nullptr) {
  return svcntp_b64(pg, op);
}

template <size_t I>
uint64_t vcntp(vbool_t pg, vbool_t op,
               typename std::enable_if_t<I == 32>* = nullptr) {
  return svcntp_b32(pg, op);
}

template <size_t I>
uint64_t vcntp(vbool_t pg, vbool_t op,
               typename std::enable_if_t<I == 16>* = nullptr) {
  return svcntp_b16(pg, op);
}

template <size_t I>
uint64_t vcntp(vbool_t pg, vbool_t op,
               typename std::enable_if_t<I == 8>* = nullptr) {
  return svcntp_b8(pg, op);
}

// svint64_t  svld1_s64(int64_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, int64_t>>* =
        nullptr) {
  return svld1_s64(pg, base);
}
// svuint64_t svld1_u64(int64_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, int64_t>>* =
        nullptr) {
  return svld1_u64(pg, (uint64_t*)base);
}
// svint64_t  svld1_s64(uint64_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, uint64_t>>* =
        nullptr) {
  return svld1_s64(pg, (int64_t*)base);
}
// svuint64_t svld1_u64(uint64_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, uint64_t>>* =
        nullptr) {
  return svld1_u64(pg, base);
}

// svint32_t  svld1_s32(int32_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, int32_t>>* =
        nullptr) {
  return svld1_s32(pg, base);
}
// svuint32_t svld1_u32(int32_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, int32_t>>* =
        nullptr) {
  return svld1_u32(pg, (uint32_t*)base);
}
// svint32_t  svld1_s32(uint32_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, uint32_t>>* =
        nullptr) {
  return svld1_s32(pg, (int32_t*)base);
}
// svuint32_t svld1_u32(uint32_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, uint32_t>>* =
        nullptr) {
  return svld1_u32(pg, base);
}

// svint16_t  svld1_s16(int16_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 16) && std::is_same_v<T, int16_t>>* =
        nullptr) {
  return svld1_s16(pg, base);
}
// svuint16_t svld1_u16(int16_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 16) && std::is_same_v<T, int16_t>>* =
        nullptr) {
  return svld1_u16(pg, (uint16_t*)base);
}
// svint16_t  svld1_s16(uint16_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 16) && std::is_same_v<T, uint16_t>>* =
        nullptr) {
  return svld1_s16(pg, (int16_t*)base);
}
// svuint16_t svld1_u16(uint16_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 16) && std::is_same_v<T, uint16_t>>* =
        nullptr) {
  return svld1_u16(pg, base);
}

// svint8_t  svld1_s8(int8_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 8) && std::is_same_v<T, int8_t>>* =
        nullptr) {
  return svld1_s8(pg, base);
}
// svuint8_t svld1_u8(int8_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 8) && std::is_same_v<T, int8_t>>* =
        nullptr) {
  return svld1_u8(pg, (uint8_t*)base);
}
// svint8_t  svld1_s8(uint8_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 8) && std::is_same_v<T, uint8_t>>* =
        nullptr) {
  return svld1_s8(pg, (int8_t*)base);
}
// svuint8_t svld1_u8(uint8_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 8) && std::is_same_v<T, uint8_t>>* =
        nullptr) {
  return svld1_u8(pg, base);
}

// svint64_t  svld1sw_s64(int32_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, int32_t>>* =
        nullptr) {
  return svld1sw_s64(pg, base);
}
// svuint64_t svld1sw_u64(int32_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, int32_t>>* =
        nullptr) {
  return svld1sw_u64(pg, base);
}
// svint64_t  svld1uw_s64(uint32_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, uint32_t>>* =
        nullptr) {
  return svld1uw_s64(pg, base);
}
// svuint64_t svld1uw_u64(uint32_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, uint32_t>>* =
        nullptr) {
  return svld1uw_u64(pg, base);
}

// svint64_t  svld1sh_s64(int16_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, int16_t>>* =
        nullptr) {
  return svld1sh_s64(pg, base);
}
// svuint64_t svld1sh_u64(int16_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, int16_t>>* =
        nullptr) {
  return svld1sh_u64(pg, base);
}
// svint64_t  svld1uh_s64(uint16_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, uint16_t>>* =
        nullptr) {
  return svld1uh_s64(pg, base);
}
// svuint64_t svld1uh_u64(uint16_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, uint16_t>>* =
        nullptr) {
  return svld1uh_u64(pg, base);
}
// svint32_t  svld1sh_s32(int16_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, int16_t>>* =
        nullptr) {
  return svld1sh_s32(pg, base);
}
// svuint32_t svld1sh_u32(int16_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, int16_t>>* =
        nullptr) {
  return svld1sh_u32(pg, base);
}
// svint32_t  svld1uh_s32(uint16_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, uint16_t>>* =
        nullptr) {
  return svld1uh_s32(pg, base);
}
// svuint32_t svld1uh_u32(uint16_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, uint16_t>>* =
        nullptr) {
  return svld1uh_u32(pg, base);
}

// svint64_t  svld1sb_s64(int8_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, int8_t>>* =
        nullptr) {
  return svld1sb_s64(pg, base);
}
// svuint64_t svld1sb_u64(int8_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, int8_t>>* =
        nullptr) {
  return svld1sb_u64(pg, base);
}
// svint64_t  svld1ub_s64(uint8_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, uint8_t>>* =
        nullptr) {
  return svld1ub_s64(pg, base);
}
// svuint64_t svld1ub_u64(uint8_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 64) && std::is_same_v<T, uint8_t>>* =
        nullptr) {
  return svld1ub_u64(pg, base);
}
// svint32_t  svld1sb_s32(int8_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, int8_t>>* =
        nullptr) {
  return svld1sb_s32(pg, base);
}
// svuint32_t svld1sb_u32(int8_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, int8_t>>* =
        nullptr) {
  return svld1sb_u32(pg, base);
}
// svint32_t  svld1ub_s32(uint8_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, uint8_t>>* =
        nullptr) {
  return svld1ub_s32(pg, base);
}
// svuint32_t svld1ub_u32(uint8_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 32) && std::is_same_v<T, uint8_t>>* =
        nullptr) {
  return svld1ub_u32(pg, base);
}
// svint16_t  svld1sb_s16(int8_t* )  : vld1ss
template <size_t I, typename T>
svint_t<I> vld1ss(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 16) && std::is_same_v<T, int8_t>>* =
        nullptr) {
  return svld1sb_s16(pg, base);
}
// svuint16_t svld1sb_u16(int8_t* )  : vld1su
template <size_t I, typename T>
svuint_t<I> vld1su(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 16) && std::is_same_v<T, int8_t>>* =
        nullptr) {
  return svld1sb_u16(pg, base);
}
// svint16_t  svld1ub_s16(uint8_t*)  : vld1us
template <size_t I, typename T>
svint_t<I> vld1us(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 16) && std::is_same_v<T, uint8_t>>* =
        nullptr) {
  return svld1ub_s16(pg, base);
}
// svuint16_t svld1ub_u16(uint8_t*)  : vld1uu
template <size_t I, typename T>
svuint_t<I> vld1uu(
    vbool_t pg, const T* base,
    typename std::enable_if_t<(I == 16) && std::is_same_v<T, uint8_t>>* =
        nullptr) {
  return svld1ub_u16(pg, base);
}

#endif  // MORPHEUS_ENABLE_SERIAL || MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_ENABLE_ARM_SVE
#endif  // MORPHEUS_ARM_ARMSVE_IMPL_HPP
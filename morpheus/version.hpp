/*****************************************************************************
 *
 *  version.hpp
 *
 *  Edinburgh Parallel Computing Centre (EPCC)
 *
 *  (c) 2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Christodoulos Stylianou (s1887443@ed.ac.uk)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *****************************************************************************/

/*! \file version.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_VERSION_HPP
#define MORPHEUS_VERSION_HPP

//  This is the only morpheus header that is guaranteed to
//  change with every morpheus release.
//
//  MORPHEUS_VERSION % 100 is the sub-minor version
//  MORPHEUS_VERSION / 100 % 1000 is the minor version
//  MORPHEUS_VERSION / 100000 is the major version

#define MORPHEUS_VERSION 001
#define MORPHEUS_MAJOR_VERSION     (MORPHEUS_VERSION / 100000)
#define MORPHEUS_MINOR_VERSION     (MORPHEUS_VERSION / 100 % 1000)
#define MORPHEUS_SUBMINOR_VERSION  (MORPHEUS_VERSION % 100)

#endif //MORPHEUS_VERSION_HPP

/*****************************************************************************
 *
 *  timer.hpp
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

/*! \file timer.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_UTIL_TIMER_HPP
#define MORPHEUS_UTIL_TIMER_HPP

#include <chrono>
#include <string>
#include <vector>

namespace morpheus
{
	namespace detail
	{
		class TimerInstance{
		private:
			using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
			using duration_t = std::chrono::duration<double>;
			using sample = std::chrono::high_resolution_clock;

			time_point t_start;
			double t_sum;
			double t_max;
			double t_min;
			unsigned int active;
			unsigned int nsteps;

		public:
			TimerInstance();

			void start();
			void stop();
			void statistics(const std::string &name);
		};
	}   // end namespace detail

	class TimerPool
	{
	private:
		std::vector<detail::TimerInstance> instances;
		std::vector<std::string> timer_name = {"Total",
		                                       "I/O Read",
		                                       "I/O Write",
		                                       "Convert",
		                                       "SpMv"};
	public:

		enum timer_id {TOTAL = 0,
			IO_READ,
			IO_WRITE,
			CONVERT,
			SPMV,
			NTIMERS /* This must be the last entry */
		};

		TimerPool();
		void start(const int t_id);
		void stop(const int t_id);
		void statistics();

	};
}   // end namespace morpheus

#include <morpheus/util/detail/timer.inl>

#endif //MORPHEUS_UTIL_TIMER_HPP

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

#ifndef EXAMPLES_INCLUDE_TIMER_HPP
#define EXAMPLES_INCLUDE_TIMER_HPP

#include <chrono>
#include <string>
#include <iomanip>

namespace morpheus
{
	namespace examples
	{
		struct timer
		{
			using timer_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
			using duration_t = std::chrono::duration<double>;
			using sample = std::chrono::high_resolution_clock;

			std::string name;
			timer_t time_start, time_stop;
			duration_t duration;
			double elapsedTime;

			timer(std::string _name = "timer")
					: elapsedTime(0.0), name(_name)
			{}

			void start()
			{
				time_start = sample::now();
			}

			void stop()
			{
				time_stop = sample::now();
				duration = time_stop - time_start;
				elapsedTime += duration.count();
			}

			void reset()
			{
				elapsedTime = 0.0;
			}
		};

		std::ostream& operator<<(std::ostream& os, const timer& t)
		{
			os << std::setw(20) << t.name << " Timer:\t" << t.elapsedTime << " (s)" << std::endl;
			return os;
		}
	}
}


#endif //EXAMPLES_INCLUDE_TIMER_HPP

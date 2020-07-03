/*****************************************************************************
 *
 *  timer.inl
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

/*! \file timer.inl
 *  \brief Description
 */

#ifndef MORPHEUS_UTIL_DETAIL_TIMER_INL
#define MORPHEUS_UTIL_DETAIL_TIMER_INL

#include <iostream>
#include <cfloat>
#include <iomanip>

namespace morpheus
{
	namespace detail
	{
		TimerInstance::TimerInstance()
				: t_sum(0.0), t_max(FLT_MIN), t_min(FLT_MAX), active(0), nsteps(0)
		{}

		void TimerInstance::statistics(const std::string &name)
		{
			if(nsteps != 0)
			{
				std::cout << std::setw(20) << name << "\t"
				          << std::setw(10) << std::setprecision(7) << t_min << "\t"
				          << std::setw(10) << std::setprecision(7) << t_max << "\t"
				          << std::setw(10) << std::setprecision(7) << t_sum << "\t"
				          << std::setw(20) << std::setprecision(14) << t_sum/ static_cast<double>(nsteps)  << "\t"
				          << "(" << nsteps << "calls)"
				          << std::endl;
			}
		}

		void TimerInstance::start()
		{
			t_start = sample::now();
			active = 1;
			nsteps += 1;
		}

		void TimerInstance::stop()
		{
			duration_t duration;
			double t_elapse;

			if(active)
			{
				duration = sample::now() - t_start;
				t_elapse = duration.count();
				t_sum += t_elapse;
				t_max = std::max(t_max, t_elapse);
				t_min = std::min(t_min, t_elapse);
				active = 0;
			}
		}

	}   // end namespace detail

	TimerPool::TimerPool() : instances(timer_id::NTIMERS)
	{};

	void TimerPool::start(const int t_id)
	{
		instances[t_id].start();
	}

	void TimerPool::stop(const int t_id)
	{
		instances[t_id].stop();
	}

	void TimerPool::statistics()
	{
		std::cout << "\nTimer statistics:" << std::endl;
		std::cout << std::setw(20) << "Section" << "\t"
		          << std::setw(10) << "tmin" << "\t"
		          << std::setw(10) << "tmax" << "\t"
		          << std::setw(20) << "total" << std::endl;

		for(int n = 0; n < timer_id::NTIMERS; n++)
		{
			instances[n].statistics(timer_name[n]);
		}

	}

}   // end namespace morpheus

#endif //MORPHEUS_UTIL_DETAIL_TIMER_INL

#include "Timer.hpp"

void Timer::tic()
{
    m_start = std::chrono::steady_clock::now();
}

void Timer::toc()
{
    if(m_start.time_since_epoch().count() <= 0)
        return;

    auto elapsedTime = std::chrono::steady_clock::now() - m_start;
    m_start          = {};

    m_elapsed += elapsedTime;
}

std::chrono::steady_clock::duration Timer::elapsed() const
{
    return m_elapsed;
}

size_t Timer::milliseconds() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(m_elapsed).count();
}

size_t Timer::microseconds() const
{
    return std::chrono::duration_cast<std::chrono::microseconds>(m_elapsed).count();
}

size_t Timer::nanoseconds() const
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(m_elapsed).count();
}

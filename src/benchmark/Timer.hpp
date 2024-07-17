#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>

class Timer
{
public:

    Timer() : m_start{}, m_elapsed(0) {}

    void tic();
    void toc();

    std::chrono::steady_clock::duration elapsed() const;

    size_t milliseconds() const;
    size_t microseconds() const;
    size_t nanoseconds() const;

protected:
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::duration   m_elapsed;
};


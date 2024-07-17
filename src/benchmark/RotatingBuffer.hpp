#include <memory>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.hpp>

namespace RotatingBuffer
{

    struct Buffer
    {
        Buffer() = default;

        template <typename T>
        void rotate(uint n)
        {
            m_offset += n * sizeof(T);
            // wrap to zero if accessing the last element would overflow
            if((m_offset + n * sizeof(T)) > m_bytes)
                m_offset = 0;
            // throw an error if accessing the last element would overflow
            if((m_offset + n * sizeof(T)) > m_bytes)
                throw std::runtime_error("buffer too small.");
        }

        template <typename T>
        Buffer& resize(size_t n)
        {
            char* ptr;

            m_bytes = sizeof(T) * n;

            auto result = hipMalloc(&ptr, m_bytes);
            if(result != hipSuccess)
            {
                throw std::runtime_error(hipGetErrorString(result));
            }

            m_device = std::shared_ptr<char>(ptr, hipFree);
            m_offset = 0;

            return *this;
        }

        template <typename T, typename R>
        Buffer& random(R min, R max)
        {
            std::random_device                      dgen;
            hiprand_cpp::mtgp32                     engine(dgen());
            hiprand_cpp::normal_distribution<float> dist(min, max);

            size_t n   = m_bytes / sizeof(T);
            T*     ptr = reinterpret_cast<T*>(m_device.get());

            dist(engine, ptr, n);

            return *this;
        }

        template <typename T>
        T* device()
        {
            return reinterpret_cast<T*>(m_device.get() + m_offset);
        }

        template <typename T>
        T* device(uint n)
        {
            T* rv = device<T>();
            rotate<T>(n);
            return rv;
        }

        template <typename T>
        std::vector<T> host(uint n)
        {
            std::vector<T> data(n);

            T*   ptr    = reinterpret_cast<T*>(m_device.get() + m_offset);
            auto result = hipMemcpy(data.data(), ptr, n * sizeof(T), hipMemcpyDeviceToHost);
            if(result != hipSuccess)
            {
                throw std::runtime_error(hipGetErrorString(result));
            }

            return data;
        }

        static Buffer* getInstance()
        {
	    // XXX datatype...
            static Buffer* ptr = nullptr;
            if(ptr == nullptr)
            {
                ptr = new Buffer();
                ptr->resize<float>(2 * 1024 * 1024 * static_cast<size_t>(1024));
                ptr->random<float>(-1.0f, 1.0f);
            }
            return ptr;
        }

    private:
        size_t                m_bytes;
        std::shared_ptr<char> m_device;
        uint                  m_offset;
    };

}

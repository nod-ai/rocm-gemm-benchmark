#ifndef GEMM_DATA_H
#define GEMM_DATA_H

#include "gemm-bench.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define GEMM_BUFFER_MAX_CAPACITY 1e8

class GEMMDataInitializer;

class GEMMDataBuffer
{
public:
    GEMMDataBuffer(size_t maxBytes = GEMM_BUFFER_MAX_CAPACITY);
    ~GEMMDataBuffer() = default;

    void resize(size_t size);
    void rotate(size_t stride);

    const std::vector<float>& getData() const
    {
        return buffVec;
    };
    const float* getBuffer() const
    {
        return buffVec.data() + _offset;
    };
    size_t capacity() const
    {
        return _capacity;
    };
    size_t size() const
    {
        return _size;
    };

    friend GEMMDataInitializer;

private:
    size_t             _capacity;
    size_t             _offset;
    size_t             _size;
    std::vector<float> buffVec;
};

class GEMMData
{
public:
    GEMMData(const std::string&   dtype,
             size_t               capacity    = GEMM_BUFFER_MAX_CAPACITY,
             GEMMDataInitializer* initializer = nullptr);
    ~GEMMData();

    template <typename Initializer, typename... Args>
    void initialize(Args&&... args)
    {
        Initializer initializer(std::forward<Args>(args)...);
        initializer.initialize(*this);
    };
    size_t getCapacity()
    {
        return capacity;
    }

    const std::vector<float>& getA() const;
    const std::vector<float>& getB() const;
    const float*              getBufferA() const;
    const float*              getBufferB() const;

    friend GEMMDataInitializer;

private:
    size_t         capacity;
    std::string    dtype;
    GEMMDataBuffer A, B;
};

class GEMMDataInitializer
{
public:
    virtual void               initialize(GEMMData& data) = 0;
    static std::vector<float>& getDataA(GEMMData& data)
    {
        return data.A.buffVec;
    }
    static std::vector<float>& getDataB(GEMMData& data)
    {
        return data.B.buffVec;
    }
};

class GEMMNullInitializer : public GEMMDataInitializer
{
public:
    void initialize(GEMMData& data) override;
};

class GEMMRandInitializer : public GEMMDataInitializer
{
public:
    explicit GEMMRandInitializer(unsigned int seed = std::random_device{}(),
                                 float        min  = 0.0f,
                                 float        max  = 1.0f);
    void initialize(GEMMData& data) override;

private:
    std::mt19937                          gen;
    std::uniform_real_distribution<float> dist;
};

class GEMMTrigInitializer : public GEMMDataInitializer
{
public:
    explicit GEMMTrigInitializer(bool useCosine = false);
    void initialize(GEMMData& data) override;

private:
    bool useCosine;
};

#endif // GEMM_DATA_H

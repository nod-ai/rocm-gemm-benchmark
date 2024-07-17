#include "DataInitialization.hpp"

#include <algorithm>

GEMMDataBuffer::GEMMDataBuffer(size_t maxBytes)
    : _capacity(maxBytes)
    , _offset(0)
    , _size(0)
{
    buffVec.reserve(_capacity);
}

void GEMMDataBuffer::resize(size_t size)
{
    buffVec.resize(size);
}

void GEMMDataBuffer::rotate(size_t stride)
{
    _offset += stride;
    if(_offset + _size >= _capacity)
        _offset = 0;
}

GEMMData::GEMMData(const std::string& dtype, size_t capacity, GEMMDataInitializer* initializer)
    : dtype(dtype)
    , capacity(capacity)
{
    A.resize(capacity);
    B.resize(capacity);

    if(initializer)
        initializer->initialize(*this);
}

GEMMData::~GEMMData() = default;

const std::vector<float>& GEMMData::getA() const
{
    return A.getData();
}

const std::vector<float>& GEMMData::getB() const
{
    return B.getData();
}

const float* GEMMData::getBufferA() const
{
    return A.getBuffer();
}

const float* GEMMData::getBufferB() const
{
    return B.getBuffer();
}

void GEMMNullInitializer::initialize(GEMMData& data)
{
    std::fill(getDataA(data).begin(), getDataA(data).end(), 0.0f);
    std::fill(getDataB(data).begin(), getDataB(data).end(), 0.0f);
}

GEMMRandInitializer::GEMMRandInitializer(unsigned int seed, float min, float max)
    : gen(seed)
    , dist(min, max)
{
}

void GEMMRandInitializer::initialize(GEMMData& data)
{
    std::generate(getDataA(data).begin(), getDataA(data).end(), [this]() { return dist(gen); });
    std::generate(getDataB(data).begin(), getDataB(data).end(), [this]() { return dist(gen); });
}

GEMMTrigInitializer::GEMMTrigInitializer(bool useCosine)
    : useCosine(useCosine)
{
}

void GEMMTrigInitializer::initialize(GEMMData& data)
{
    auto initTrig = [this](float val) { return useCosine ? std::cos(val) : std::sin(val); };
    std::generate(getDataA(data).begin(), getDataA(data).end(), [i = 0, &initTrig]() mutable {
        return initTrig(static_cast<float>(i++));
    });
    std::generate(getDataB(data).begin(), getDataB(data).end(), [i = 0, &initTrig]() mutable {
        return initTrig(static_cast<float>(i++));
    });
}


#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/numpy.hpp>

#include "FrequencyMonitor.hpp"

namespace bp  = boost::python;
namespace bnp = boost::python::numpy;

namespace FrequencyMonitorModule
{
    template <typename T>
    bnp::ndarray array1d(std::vector<T> const& x)
    {
        auto shape  = bp::make_tuple(x.size());
        auto stride = bp::make_tuple(sizeof(T));
        auto dtype  = bnp::dtype::get_builtin<T>();

        return bnp::from_data(x.data(), dtype, shape, stride, bp::object()).copy();
    }

    void setDevice(int device)
    {
        Frequency::getFrequencyMonitor()->setDevice(device);
    }

    void start()
    {
        Frequency::getFrequencyMonitor()->start();
    }

    bp::dict stop()
    {
        auto fm = Frequency::getFrequencyMonitor();
        fm->stop();

        bp::dict sclk;
        {
            auto trace = fm->systemFrequency();
            auto stats = Frequency::statistics(trace);

            sclk["min"]    = bp::object(stats.min);
            sclk["max"]    = bp::object(stats.max);
            sclk["mean"]   = bp::object(stats.mean);
            sclk["median"] = bp::object(stats.median);
            sclk["trace"]  = array1d(trace);
        }

        bp::dict mclk;
        {
            auto trace = fm->memoryFrequency();
            auto stats = Frequency::statistics(trace);

            mclk["min"]    = bp::object(stats.min);
            mclk["max"]    = bp::object(stats.max);
            mclk["mean"]   = bp::object(stats.mean);
            mclk["median"] = bp::object(stats.median);
            mclk["trace"]  = array1d(trace);
        }

        bp::dict fclk;
        {
            auto trace = fm->dataFrequency();
            auto stats = Frequency::statistics(trace);

            fclk["min"]    = bp::object(stats.min);
            fclk["max"]    = bp::object(stats.max);
            fclk["mean"]   = bp::object(stats.mean);
            fclk["median"] = bp::object(stats.median);
            fclk["trace"]  = array1d(trace);
        }

        bp::dict temperature;
        {
            auto trace = fm->temperature();
            auto stats = Frequency::statistics(trace);

            temperature["min"]    = bp::object(stats.min);
            temperature["max"]    = bp::object(stats.max);
            temperature["mean"]   = bp::object(stats.mean);
            temperature["median"] = bp::object(stats.median);
            temperature["trace"]  = array1d(trace);
        }

        bp::dict power;
        {
            auto trace = fm->power();
            auto stats = Frequency::statistics(trace);

            power["min"]    = bp::object(stats.min);
            power["max"]    = bp::object(stats.max);
            power["mean"]   = bp::object(stats.mean);
            power["median"] = bp::object(stats.median);
            power["trace"]  = array1d(trace);
        }

        bp::dict rv;
        rv["sclk"]        = sclk;
        rv["mclk"]        = mclk;
        rv["fclk"]        = fclk;
        rv["temperature"] = temperature;
        rv["power"]       = power;
        return rv;
    }

}

BOOST_PYTHON_MODULE(frequencymonitor)
{
    bnp::initialize();

    bp::def("set_device", FrequencyMonitorModule::setDevice);
    bp::def("start", FrequencyMonitorModule::start);
    bp::def("stop", FrequencyMonitorModule::stop);
}

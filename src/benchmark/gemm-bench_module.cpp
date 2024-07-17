#define PY_SSIZE_T_CLEAN
#include "gemm-bench.hpp"
#include <Python.h>
#include <numpy/arrayobject.h>

template <typename T>
PyObject* vector_to_numpy(const std::vector<T>& vec)
{
    npy_intp  dims[1] = {static_cast<npy_intp>(vec.size())};
    PyObject* arr     = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    memcpy(PyArray_DATA((PyArrayObject*)arr), vec.data(), vec.size() * sizeof(T));
    return arr;
}

template <typename T>
PyObject* struct_to_bytes(const T& p)
{
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(&p), sizeof(T));
}

static PyObject* deserialize_result(PyObject* self, PyObject* args)
{
    PyObject* bytes;
    if(!PyArg_ParseTuple(args, "O", &bytes))
    {
        return NULL;
    }
    GEMMBench::Result* result = reinterpret_cast<GEMMBench::Result*>(PyBytes_AS_STRING(bytes));

    PyObject* dict = PyDict_New();
    PyDict_SetItemString(dict, "ok", PyBool_FromLong(result->ok));
    PyDict_SetItemString(dict, "device", PyLong_FromLong(result->device));
    PyDict_SetItemString(dict, "warm_iterations", PyLong_FromLong(result->warm_iterations));
    PyDict_SetItemString(dict, "mean_microseconds", PyFloat_FromDouble(result->mean_microseconds));
    PyDict_SetItemString(dict, "min_sclk", PyFloat_FromDouble(result->min_sclk));
    PyDict_SetItemString(dict, "mean_sclk", PyFloat_FromDouble(result->mean_sclk));
    PyDict_SetItemString(dict, "max_sclk", PyFloat_FromDouble(result->max_sclk));
    PyDict_SetItemString(dict, "min_mclk", PyFloat_FromDouble(result->min_mclk));
    PyDict_SetItemString(dict, "mean_mclk", PyFloat_FromDouble(result->mean_mclk));
    PyDict_SetItemString(dict, "max_mclk", PyFloat_FromDouble(result->max_mclk));
    PyDict_SetItemString(dict, "min_fclk", PyFloat_FromDouble(result->min_fclk));
    PyDict_SetItemString(dict, "mean_fclk", PyFloat_FromDouble(result->mean_fclk));
    PyDict_SetItemString(dict, "max_fclk", PyFloat_FromDouble(result->max_fclk));
    return dict;
}

static PyObject* deserialize_float1d(PyObject* self, PyObject* args)
{
    PyObject* bytes;
    if(!PyArg_ParseTuple(args, "O", &bytes))
    {
        return NULL;
    }
    Py_ssize_t size = PyBytes_Size(bytes) / sizeof(float);
    float*     data = reinterpret_cast<float*>(PyBytes_AS_STRING(bytes));

    npy_intp  dims[1] = {size};
    PyObject* arr     = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    memcpy(PyArray_DATA((PyArrayObject*)arr), data, size * sizeof(float));
    return arr;
}
typedef struct
{
    PyObject_HEAD GEMMBench::Problem* cpp_obj;
} ProblemObject;

static void Problem_dealloc(ProblemObject* self)
{
    delete self->cpp_obj;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Problem_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    ProblemObject* self;
    self = (ProblemObject*)type->tp_alloc(type, 0);
    if(self != NULL)
    {
        self->cpp_obj = NULL;
    }
    return (PyObject*)self;
}

static int Problem_init(ProblemObject* self, PyObject* args, PyObject* kwds)
{
    const char * tag, *A, *B, *dtype;
    unsigned int M, N, K;
    if(!PyArg_ParseTuple(args, "sIIIsss", &tag, &M, &N, &K, &A, &B, &dtype))
        return -1;

    self->cpp_obj = new GEMMBench::Problem(
        std::string(tag), M, N, K, std::string(A), std::string(B), std::string(dtype));
    return 0;
}

static PyObject* Problem_serialize(ProblemObject* self, PyObject* Py_UNUSED(ignored))
{
    return struct_to_bytes(*(self->cpp_obj));
}

static PyObject* Problem_to_dict(ProblemObject* self, PyObject* Py_UNUSED(ignored))
{
    PyObject* dict = PyDict_New();
    PyDict_SetItemString(dict, "M", PyLong_FromLong(self->cpp_obj->M));
    PyDict_SetItemString(dict, "N", PyLong_FromLong(self->cpp_obj->N));
    PyDict_SetItemString(dict, "K", PyLong_FromLong(self->cpp_obj->K));
    PyDict_SetItemString(dict, "tag", PyUnicode_FromString(self->cpp_obj->tag));
    PyDict_SetItemString(dict, "dtype", PyUnicode_FromString(self->cpp_obj->dtype));
    PyDict_SetItemString(dict, "A", PyUnicode_FromStringAndSize(&self->cpp_obj->A, 1));
    PyDict_SetItemString(dict, "B", PyUnicode_FromStringAndSize(&self->cpp_obj->B, 1));
    return dict;
}

static PyMethodDef Problem_methods[]
    = {{"serialize", (PyCFunction)Problem_serialize, METH_NOARGS, "Serialize Problem to bytes"},
       {"to_dict", (PyCFunction)Problem_to_dict, METH_NOARGS, "Convert Problem to dict"},
       {NULL}};

static PyTypeObject ProblemType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "gbm.Problem",
    .tp_basicsize                          = sizeof(ProblemObject),
    .tp_itemsize                           = 0,
    .tp_dealloc                            = (destructor)Problem_dealloc,
    .tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc                                = "Problem objects",
    .tp_methods                            = Problem_methods,
    .tp_init                               = (initproc)Problem_init,
    .tp_new                                = Problem_new,
};

// Solution class
typedef struct
{
    PyObject_HEAD GEMMBench::Solution* cpp_obj;
} SolutionObject;

static void Solution_dealloc(SolutionObject* self)
{
    delete self->cpp_obj;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Solution_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    SolutionObject* self;
    self = (SolutionObject*)type->tp_alloc(type, 0);
    if(self != NULL)
    {
        self->cpp_obj = NULL;
    }
    return (PyObject*)self;
}

static int Solution_init(SolutionObject* self, PyObject* args, PyObject* kwds)
{
    const char* name;
    if(!PyArg_ParseTuple(args, "s", &name))
        return -1;

    self->cpp_obj = new GEMMBench::Solution(std::string(name));
    return 0;
}

static PyObject* Solution_serialize(SolutionObject* self, PyObject* Py_UNUSED(ignored))
{
    return struct_to_bytes(*(self->cpp_obj));
}

static PyObject* Solution_to_dict(SolutionObject* self, PyObject* Py_UNUSED(ignored))
{
    PyObject* dict = PyDict_New();
    PyDict_SetItemString(dict, "name", PyUnicode_FromString(self->cpp_obj->name));
    return dict;
}

static PyMethodDef Solution_methods[]
    = {{"serialize", (PyCFunction)Solution_serialize, METH_NOARGS, "Serialize Solution to bytes"},
       {"to_dict", (PyCFunction)Solution_to_dict, METH_NOARGS, "Convert Solution to dict"},
       {NULL}};

static PyTypeObject SolutionType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "gbm.Solution",
    .tp_basicsize                          = sizeof(SolutionObject),
    .tp_itemsize                           = 0,
    .tp_dealloc                            = (destructor)Solution_dealloc,
    .tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc                                = "Solution objects",
    .tp_methods                            = Solution_methods,
    .tp_init                               = (initproc)Solution_init,
    .tp_new                                = Solution_new,
};

// Configuration class
typedef struct
{
    PyObject_HEAD GEMMBench::Configuration* cpp_obj;
} ConfigurationObject;

static void Configuration_dealloc(ConfigurationObject* self)
{
    delete self->cpp_obj;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Configuration_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    ConfigurationObject* self;
    self = (ConfigurationObject*)type->tp_alloc(type, 0);
    if(self != NULL)
    {
        self->cpp_obj = NULL;
    }
    return (PyObject*)self;
}

static int Configuration_init(ConfigurationObject* self, PyObject* args, PyObject* kwds)
{
    unsigned int fclk, mclk, gfxclk;
    if(!PyArg_ParseTuple(args, "III", &fclk, &mclk, &gfxclk))
        return -1;

    self->cpp_obj = new GEMMBench::Configuration(fclk, mclk, gfxclk);
    return 0;
}

static PyObject* Configuration_serialize(ConfigurationObject* self, PyObject* Py_UNUSED(ignored))
{
    return struct_to_bytes(*(self->cpp_obj));
}

static PyObject* Configuration_to_dict(ConfigurationObject* self, PyObject* Py_UNUSED(ignored))
{
    PyObject* dict = PyDict_New();
    PyDict_SetItemString(dict, "fclk", PyLong_FromLong(self->cpp_obj->fclk));
    PyDict_SetItemString(dict, "mclk", PyLong_FromLong(self->cpp_obj->mclk));
    PyDict_SetItemString(dict, "gfxclk", PyLong_FromLong(self->cpp_obj->gfxclk));
    return dict;
}

static PyMethodDef Configuration_methods[] = {
    {"serialize",
     (PyCFunction)Configuration_serialize,
     METH_NOARGS,
     "Serialize Configuration to bytes"},
    {"to_dict", (PyCFunction)Configuration_to_dict, METH_NOARGS, "Convert Configuration to dict"},
    {NULL}};

static PyTypeObject ConfigurationType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "gbm.Configuration",
    .tp_basicsize                          = sizeof(ConfigurationObject),
    .tp_itemsize                           = 0,
    .tp_dealloc                            = (destructor)Configuration_dealloc,
    .tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc                                = "Configuration objects",
    .tp_methods                            = Configuration_methods,
    .tp_init                               = (initproc)Configuration_init,
    .tp_new                                = Configuration_new,
};

// Result class
typedef struct
{
    PyObject_HEAD GEMMBench::Result* cpp_obj;
} ResultObject;

static void Result_dealloc(ResultObject* self)
{
    delete self->cpp_obj;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Result_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    ResultObject* self;
    self = (ResultObject*)type->tp_alloc(type, 0);
    if(self != NULL)
    {
        self->cpp_obj = NULL;
    }
    return (PyObject*)self;
}

static int Result_init(ResultObject* self, PyObject* args, PyObject* kwds)
{
    self->cpp_obj = new GEMMBench::Result();
    return 0;
}

static PyTypeObject ResultType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "gbm.Result",
    .tp_basicsize                          = sizeof(ResultObject),
    .tp_itemsize                           = 0,
    .tp_dealloc                            = (destructor)Result_dealloc,
    .tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc                                = "Result objects",
    .tp_init                               = (initproc)Result_init,
    .tp_new                                = Result_new,
};

static PyMethodDef GbmMethods[]
    = {{"deserialize_result", deserialize_result, METH_VARARGS, "Deserialize Result"},
       {"float1d", deserialize_float1d, METH_VARARGS, "Deserialize float array"},
       {NULL, NULL, 0, NULL}};

static struct PyModuleDef gbmmodule = {PyModuleDef_HEAD_INIT, "gbm", NULL, -1, GbmMethods};

PyMODINIT_FUNC PyInit_gbm(void)
{
    PyObject* m;

    import_array();

    m = PyModule_Create(&gbmmodule);
    if(m == NULL)
        return NULL;

    if(PyType_Ready(&ProblemType) < 0)
        return NULL;
    if(PyType_Ready(&SolutionType) < 0)
        return NULL;
    if(PyType_Ready(&ConfigurationType) < 0)
        return NULL;
    if(PyType_Ready(&ResultType) < 0)
        return NULL;

    Py_INCREF(&ProblemType);
    PyModule_AddObject(m, "Problem", (PyObject*)&ProblemType);
    Py_INCREF(&SolutionType);
    PyModule_AddObject(m, "Solution", (PyObject*)&SolutionType);
    Py_INCREF(&ConfigurationType);
    PyModule_AddObject(m, "Configuration", (PyObject*)&ConfigurationType);
    Py_INCREF(&ResultType);
    PyModule_AddObject(m, "Result", (PyObject*)&ResultType);

    return m;
}

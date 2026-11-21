#include "PythonBridgePytorchState.h"
#include "PythonCallBackThrottler.h"
#include <Python.h>
#include <exception>

using namespace std::placeholders;

PyObject* pythonBridgePytorchState_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {

    pythonBridgePytorchStateObject* self;
    self = (pythonBridgePytorchStateObject*)type->tp_alloc(type, 0);
    if (self != NULL) { // -> allocation successfull
        // assign initial values
        self->m_pytorchModel = NULL;
    }
    return (PyObject*)self;
}

int pythonBridgePytorchState_init(PyObject* self, PyObject* args, PyObject* kwds) {

    pythonBridgePytorchStateObject* m = (pythonBridgePytorchStateObject*)self;

    try {
        // init only by PyBridge
        m->m_pytorchModel.reset();
    } catch (const std::exception& ex) {
        m->m_pytorchModel = NULL;
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return -1;
    } catch (...) {
        m->m_pytorchModel = NULL;
        PyErr_SetString(PyExc_RuntimeError, "Initialization failed");
        return -1;
    }

    return 0;
}

void pythonBridgePytorchState_dealloc(pythonBridgePytorchStateObject* self) {
    PyTypeObject* tp = Py_TYPE(self);

    auto m = reinterpret_cast<pythonBridgePytorchStateObject*>(self);

    if (m->m_pytorchModel) {
        m->m_pytorchModel.reset();
    }

    tp->tp_free(self);
    Py_DECREF(tp);
}

PyObject* pythonBridgePytorchState_decodeToBinary(PyObject* self, PyObject* args) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const char* text;
    PyObject* callBack;
    unsigned int callBackIgnore;
    int useEndMarker;
    double corrVactor;
    unsigned int maxExtend;
    double huffMax;

    if (!PyArg_ParseTuple(args, "yIOpdId", &text, &callBackIgnore, &callBack, &useEndMarker, &corrVactor, &maxExtend,
                          &huffMax))
        return NULL;

    if (!PyCallable_Check(callBack)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }
    Py_INCREF(callBack);

    std::tuple<std::string, unsigned int, unsigned int> res;
    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    {
        PyThreadState* _save;
        _save = PyEval_SaveThread();

        PythonCallBackThrottler throttler(&_save, callBackIgnore, callBack);
        try {
            res = _self->m_pytorchModel->decodeToBinary(
                text, [&](int toProcess) { throttler.process(toProcess); }, static_cast<bool>(useEndMarker), corrVactor,
                maxExtend, huffMax);
        } catch (std::exception& except) {
            PyEval_RestoreThread(_save);
            Py_DECREF(callBack);
            PyErr_SetString(PyExc_RuntimeError, except.what());
            return nullptr;
        }
        PyEval_RestoreThread(_save);
    }
    Py_DECREF(callBack);
    std::string output = std::get<0>(res);
    int leftover = std::get<1>(res);
    int maxDestroy = std::get<2>(res);
    return Py_BuildValue("(sii)", output.c_str(), leftover, maxDestroy);
}

PyObject* pythonBridgePytorchState_decodeToBinaryRangeCoding(PyObject* self, PyObject* args) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const char* text;
    PyObject* callBack;
    unsigned int callBackIgnore;
    double corrVactor;
    unsigned int normalBufferSize;
    unsigned int minUsefullBitsSize;
    int useEndMarker;

    if (!PyArg_ParseTuple(args, "yIOpdII", &text, &callBackIgnore, &callBack, &useEndMarker, &corrVactor,
                          &minUsefullBitsSize, &normalBufferSize))
        return NULL;

    if (!PyCallable_Check(callBack)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }
    Py_INCREF(callBack);

    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    std::tuple<std::string, int> res;
    {
        PyThreadState* _save;
        _save = PyEval_SaveThread();

        PythonCallBackThrottler throttler(&_save, callBackIgnore, callBack);
        try {
            res = _self->m_pytorchModel->decodeToBinaryRangeCoding(
                text, [&](int toProcess) { throttler.process(toProcess); }, static_cast<bool>(useEndMarker),
                minUsefullBitsSize, normalBufferSize, corrVactor);
        } catch (std::exception& except) {
            PyEval_RestoreThread(_save);
            Py_DECREF(callBack);
            PyErr_SetString(PyExc_RuntimeError, except.what());
            return nullptr;
        }
        PyEval_RestoreThread(_save);
    }
    Py_DECREF(callBack);

    std::string output = std::get<0>(res);
    int leftover = std::get<1>(res);
    return Py_BuildValue("(si)", output.c_str(), leftover);
}

PyObject* pythonBridgePytorchState_decodeFromBinary(PyObject* self, PyObject* args) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const char* text;
    PyObject* callBack;
    unsigned int callBackIgnore;
    int useEndMarker;
    double corrVactor;
    unsigned int maxExtend;
    double huffMax;

    if (!PyArg_ParseTuple(args, "sIOpdId", &text, &callBackIgnore, &callBack, &useEndMarker, &corrVactor, &maxExtend,
                          &huffMax))
        return NULL;

    if (!PyCallable_Check(callBack)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }
    Py_INCREF(callBack);

    std::tuple<std::string, int> res;

    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    {
        PyThreadState* _save;
        _save = PyEval_SaveThread();

        PythonCallBackThrottler throttler(&_save, callBackIgnore, callBack);
        try {
            res = _self->m_pytorchModel->decodeFromBinary(
                text, [&](int toProcess) { throttler.process(toProcess); }, static_cast<bool>(useEndMarker), corrVactor,
                maxExtend, huffMax);
        } catch (std::exception& except) {
            PyEval_RestoreThread(_save);
            PyErr_SetString(PyExc_RuntimeError, except.what());
            Py_XDECREF(callBack);
            return nullptr;
        }
        PyEval_RestoreThread(_save);
    }
    Py_DECREF(callBack);

    std::string output = std::get<0>(res);
    int leftover = std::get<1>(res);
    return Py_BuildValue("(yi)", output.c_str(), leftover);
}

PyObject* pythonBridgePytorchState_initRangeCoding(PyObject* self, PyObject* args) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    unsigned int normalBufferSize;

    if (!PyArg_ParseTuple(args, "I", &normalBufferSize))
        return NULL;

    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    try {
        _self->m_pytorchModel->initRangeCoding(normalBufferSize);
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }

    Py_RETURN_NONE;
}

PyObject* pythonBridgePytorchState_decodeFromBinaryRangeCoding(PyObject* self, PyObject* args) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const char* text;
    PyObject* callBack;
    unsigned int callBackIgnore;
    int useEndMarker;
    double corrVactor;
    unsigned int minUsefullBitsSize;
    unsigned int normalBufferSize;
    long long sizeToEncode;

    if (!PyArg_ParseTuple(args, "sLIOpdII", &text, &sizeToEncode, &callBackIgnore, &callBack, &useEndMarker,
                          &corrVactor, &minUsefullBitsSize, &normalBufferSize))
        return NULL;

    if (!PyCallable_Check(callBack)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }
    Py_INCREF(callBack);

    std::tuple<std::string, int> res;

    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    {
        PyThreadState* _save;
        _save = PyEval_SaveThread();

        PythonCallBackThrottler throttler(&_save, callBackIgnore, callBack);
        try {
            res = _self->m_pytorchModel->decodeFromBinaryRangeCoding(
                text, [&](int toProcess) { throttler.process(toProcess); }, static_cast<bool>(useEndMarker),
                minUsefullBitsSize, normalBufferSize, sizeToEncode, corrVactor);
        } catch (std::exception& except) {
            PyEval_RestoreThread(_save);
            PyErr_SetString(PyExc_RuntimeError, except.what());
            Py_XDECREF(callBack);
            return nullptr;
        }
        PyEval_RestoreThread(_save);
    }
    Py_DECREF(callBack);

    std::string output = std::get<0>(res);
    int leftover = std::get<1>(res);
    return Py_BuildValue("(yi)", output.c_str(), leftover);
}

PyObject* pythonBridgePytorchState_runWithoutDecoding(PyObject* self, PyObject* args) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const char* text;
    PyObject* callBack;
    unsigned int callBackIgnore;
    bool updateBlacklist;

    if (!PyArg_ParseTuple(args, "yIOp", &text, &callBackIgnore, &callBack, &updateBlacklist))
        return NULL;
    if (!PyCallable_Check(callBack)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }
    Py_INCREF(callBack);

    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    {
        PyThreadState* _save;
        _save = PyEval_SaveThread();

        PythonCallBackThrottler throttler(&_save, callBackIgnore, callBack);
        try {
            _self->m_pytorchModel->initWithText(
                text, [&](int toProcess) { throttler.process(toProcess); }, updateBlacklist);
        } catch (std::exception& except) {
            PyEval_RestoreThread(_save);
            PyErr_SetString(PyExc_RuntimeError, except.what());
            Py_XDECREF(callBack);
            return nullptr;
        }

        PyEval_RestoreThread(_save);
    }
    Py_DECREF(callBack);
    Py_RETURN_NONE;
}

PyObject* pythonBridgePytorchState_cloneDecState(PyObject* self, [[maybe_unused]] PyObject* unused) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);

    PyTypeObject* tp = Py_TYPE(self);

    PyObject* pyResObj = pythonBridgePytorchState_new(tp, nullptr, nullptr);
    Py_INCREF(pyResObj);

    auto _pyResObj = reinterpret_cast<pythonBridgePytorchStateObject*>(pyResObj);

    try {
        _pyResObj->m_pytorchModel = std::move(_self->m_pytorchModel->cloneState());
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }

    return pyResObj;
}

PyObject* pythonBridgePytorchState_inLasterCreatedNodes(PyObject* self, PyObject* args) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    const char* toFind;
    if (!PyArg_ParseTuple(args, "y", &toFind))
        return NULL;
    bool res = false;
    try {
        res = _self->m_pytorchModel->inNodes(toFind);
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }
    if (res)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

PyObject* pythonBridgePytorchState_resetDecState(PyObject* self, [[maybe_unused]] PyObject* unused) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    try {
        _self->m_pytorchModel->resetHidden();
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* pythonBridgePytorchState_getMin(PyObject* self, [[maybe_unused]] PyObject* unused) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    BigUInt min;
    try {
        min = _self->m_pytorchModel->getMin();
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }
    auto res = min.toString();
    return PyUnicode_DecodeUTF8(res.c_str(), res.size(), nullptr);
}

PyObject* pythonBridgePytorchState_getMax(PyObject* self, [[maybe_unused]] PyObject* unused) {
    assert(self);

    auto _self = reinterpret_cast<pythonBridgePytorchStateObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_pytorchModel->getParrent()->running_mutex);
    BigUInt max;
    try {
        max = _self->m_pytorchModel->getMax();
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }
    auto res = max.toString();
    return PyUnicode_DecodeUTF8(res.c_str(), res.size(), nullptr);
}
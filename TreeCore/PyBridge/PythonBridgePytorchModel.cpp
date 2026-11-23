#include <Python.h>
#include <cassert>
#include <exception>
#include <set>
#include <string>

#include "PythonBridgePytorchModel.h"
#include "PythonBridgePytorchState.h"
#include "PytorchModelCache.h"

PyObject* pythonBridgePytorchModel_new(PyTypeObject* type, [[maybe_unused]] PyObject* args, PyObject* kwds) {

    auto self = (pythonBridgePytorchModelObject*)type->tp_alloc(type, 0);
    if (self != NULL) { // -> allocation successfull
        // assign initial values
        self->m_mainObj = NULL;
    }
    return (PyObject*)self;
}

int pythonBridgePytorchModel_init(PyObject* self, PyObject* args, PyObject* kwds) {

    auto m = reinterpret_cast<pythonBridgePytorchModelObject*>(self);
    char* path;
    if (!PyArg_ParseTuple(args, "s", &path))
        return -1;

    try {
        m->m_mainObj = getModelAt(path);

        PyObject* pyTypeObj = PyType_FromSpec(&spec_pythonBridgePytorchState);
        if (pyTypeObj == nullptr) {
            m->pyTypeObj = nullptr;
            return -1;
        }
        m->pyTypeObj = pyTypeObj;

    } catch (const std::exception& ex) {
        m->m_mainObj = NULL;
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return -1;
    } catch (...) {
        m->m_mainObj = NULL;
        PyErr_SetString(PyExc_RuntimeError, "Initialization failed");
        return -1;
    }

    return 0;
}

void pythonBridgePytorchModel_dealloc(pythonBridgePytorchModelObject* self) {
    PyTypeObject* tp = Py_TYPE(self);

    auto m = reinterpret_cast<pythonBridgePytorchModelObject*>(self);

    if (m->m_mainObj) {
        m->m_mainObj.reset();
    }
    Py_XDECREF(m->pyTypeObj);
    tp->tp_free(self);
};

PyObject* pythonBridgePytorchModel_clone(PyObject* self, [[maybe_unused]] PyObject* ignored) {
    assert(self);
    auto _self = reinterpret_cast<pythonBridgePytorchModelObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_mainObj->running_mutex);

    std::shared_ptr<PytorchModel> CppObj;

    try {
        CppObj = _self->m_mainObj->clone();
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }

    PyObject* pyObj = pythonBridgePytorchModel_new(Py_TYPE(_self), nullptr, nullptr);

    auto _pyObj = reinterpret_cast<pythonBridgePytorchModelObject*>(pyObj);

    _pyObj->m_mainObj = std::move(CppObj);
    Py_INCREF(_self->pyTypeObj);
    _pyObj->pyTypeObj = _self->pyTypeObj;

    return pyObj;
}

PyObject* pythonBridgePytorchModel_getInitState(PyObject* self, [[maybe_unused]] PyObject* ignored) {
    assert(self);
    auto _self = reinterpret_cast<pythonBridgePytorchModelObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_mainObj->running_mutex);

    std::shared_ptr<PytorchState> CppObj;

    try {
        CppObj = _self->m_mainObj->generateState(_self->m_mainObj);
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }

    PyObject* pyObj = pythonBridgePytorchState_new((PyTypeObject*)(_self->pyTypeObj), nullptr, nullptr);

    auto _pyObj = reinterpret_cast<pythonBridgePytorchStateObject*>(pyObj);

    _pyObj->m_pytorchModel = std::move(CppObj);

    return pyObj;
}
PyObject* pythonBridgePytorchModel_getCharList(PyObject* self, [[maybe_unused]] PyObject* ignored) {
    assert(self);
    auto _self = reinterpret_cast<pythonBridgePytorchModelObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_mainObj->running_mutex);

    std::set<std::string> CharSet;
    try {
        CharSet = _self->m_mainObj->listU8Chars();
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }

    std::string res = "";
    for (auto i : CharSet) {
        res.append(i);
    }
    return PyUnicode_DecodeUTF8(res.c_str(), res.size(), "strict");
}
PyObject* pythonBridgePytorchModel_loadBlacklist(PyObject* self, PyObject* args) {
    assert(self);
    auto _self = reinterpret_cast<pythonBridgePytorchModelObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_mainObj->running_mutex);
    char* path;
    int capitalize;
    int force;

    if (!PyArg_ParseTuple(args, "spp", &path, &capitalize, &force))
        return NULL;
    try {
        _self->m_mainObj->loadBlacklist(path, capitalize, force);
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* pythonBridgePytorchModel_loadBlacklistLine(PyObject* self, PyObject* args) {
    assert(self);
    auto _self = reinterpret_cast<pythonBridgePytorchModelObject*>(self);
    const std::lock_guard<std::mutex> lock(_self->m_mainObj->running_mutex);
    char* line;
    int force;

    if (!PyArg_ParseTuple(args, "sp", &line, &force))
        return NULL;
    std::string errorText;
    try {
        _self->m_mainObj->loadBlacklistLine(line, errorText, force);
    } catch (std::exception& except) {
        PyErr_SetString(PyExc_RuntimeError, except.what());
        return nullptr;
    }
    if (errorText != "") {
        std::string fullText = errorText + "Still apply the blacklist where it is possible."
                                           "Consider using force True if you use the API.";
        PyErr_SetString(PyExc_RuntimeError, fullText.c_str());
        return nullptr;
    }
    Py_RETURN_NONE;
}

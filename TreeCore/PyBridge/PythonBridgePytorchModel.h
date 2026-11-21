#pragma once

#include "PytorchModel.h"
#include <Python.h>
#include <memory>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    std::shared_ptr<PytorchModel> m_mainObj;
    PyObject* pyTypeObj;
} pythonBridgePytorchModelObject;

int pythonBridgePytorchModel_init(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pythonBridgePytorchModel_new(PyTypeObject* subtype, PyObject* args, PyObject* kwds);
void pythonBridgePytorchModel_dealloc(pythonBridgePytorchModelObject* self);

PyObject* pythonBridgePytorchModel_getInitState(PyObject* self, PyObject* ignored);
PyObject* pythonBridgePytorchModel_getCharList(PyObject* self, PyObject* ignored);
PyObject* pythonBridgePytorchModel_loadBlacklist(PyObject* self, PyObject* args);
PyObject* pythonBridgePytorchModel_loadBlacklistLine(PyObject* self, PyObject* args);
PyObject* pythonBridgePytorchModel_clone(PyObject* self, PyObject* ignored);

static PyMethodDef pythonBridgePytorchModel_methods[] = {
    {"getInitState", (PyCFunction)pythonBridgePytorchModel_getInitState, METH_NOARGS, PyDoc_STR("generates a State")},
    {"getCharList", (PyCFunction)pythonBridgePytorchModel_getCharList, METH_NOARGS, PyDoc_STR("get List of all allowed Char for this Model")},
    {"loadBlacklist", (PyCFunction)pythonBridgePytorchModel_loadBlacklist, METH_VARARGS, PyDoc_STR("loads a list of Words that should not be found in the output(reduces Preformance)")},
    {"loadBlacklistLine", (PyCFunction)pythonBridgePytorchModel_loadBlacklistLine, METH_VARARGS, PyDoc_STR("loads a string that should not be found in the output(reduces Preformance)")},
    {"clone", (PyCFunction)pythonBridgePytorchModel_clone, METH_NOARGS, PyDoc_STR("create a copy of this Model")},
    {NULL, NULL} /* Sentinel */
};

static struct PyMemberDef pythonBridgePytorchModel_members[] = {
    {NULL} /* Sentinel */
};

static PyType_Slot pythonBridgePytorchModel_slots[] = {
    {Py_tp_new, (void*)pythonBridgePytorchModel_new},
    {Py_tp_init, (void*)pythonBridgePytorchModel_init},
    {Py_tp_dealloc, (void*)pythonBridgePytorchModel_dealloc},
    {Py_tp_members, pythonBridgePytorchModel_members},
    {Py_tp_methods, pythonBridgePytorchModel_methods},
    {0, 0}
};

static PyType_Spec spec_pythonBridgePytorchModel = {
    "TorchModelCompress.pytorchModel",                             // name
    sizeof(pythonBridgePytorchModelObject) + sizeof(PytorchModel), // basicsize
    0,                                                             // itemsize
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                      // flags
    pythonBridgePytorchModel_slots                                 // slots
};

#pragma once

#include "PytorchState.h"
#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    std::shared_ptr<PytorchState> m_pytorchModel;
} pythonBridgePytorchStateObject;

int pythonBridgePytorchState_init(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pythonBridgePytorchState_new(PyTypeObject* subtype, PyObject* args, PyObject* kwds);
void pythonBridgePytorchState_dealloc(pythonBridgePytorchStateObject* self);

PyObject* pythonBridgePytorchState_decodeToBinary(PyObject* self, PyObject* args);
PyObject* pythonBridgePytorchState_decodeFromBinary(PyObject* self, PyObject* args);
PyObject* pythonBridgePytorchState_runWithoutDecoding(PyObject* self, PyObject* args);

PyObject* pythonBridgePytorchState_initRangeCoding(PyObject* self, PyObject* args);
PyObject* pythonBridgePytorchState_decodeToBinaryRangeCoding(PyObject* self, PyObject* args);
PyObject* pythonBridgePytorchState_decodeFromBinaryRangeCoding(PyObject* self, PyObject* args);
PyObject* pythonBridgePytorchState_getMin(PyObject* self, [[maybe_unused]] PyObject* unused);
PyObject* pythonBridgePytorchState_getMax(PyObject* self, [[maybe_unused]] PyObject* unused);

PyObject* pythonBridgePytorchState_cloneDecState(PyObject* self, [[maybe_unused]] PyObject* unused);
PyObject* pythonBridgePytorchState_inLastCreatedNodes(PyObject* self, PyObject* args);
PyObject* pythonBridgePytorchState_resetDecState(PyObject* self, [[maybe_unused]] PyObject* unused);


static PyMethodDef pythonBridgePytorchState_methods[] = {
    {"initRangeCoding", (PyCFunction)pythonBridgePytorchState_initRangeCoding, METH_VARARGS, PyDoc_STR("Init a Range encoding call before other Range functions")},
    {"decodeToBinaryRangeCoding", (PyCFunction)pythonBridgePytorchState_decodeToBinaryRangeCoding, METH_VARARGS, PyDoc_STR("Decode Input to Binary String of 1 and 0")},
    {"decodeFromBinaryRangeCoding", (PyCFunction)pythonBridgePytorchState_decodeFromBinaryRangeCoding, METH_VARARGS, PyDoc_STR("Decode Input of Binary String of 1 and 0 to tokens")},
    {"getMin", (PyCFunction)pythonBridgePytorchState_getMin, METH_NOARGS, PyDoc_STR("get the current Min for RangeCoding")},
    {"getMax", (PyCFunction)pythonBridgePytorchState_getMax, METH_NOARGS, PyDoc_STR("get the current Max for RangeCoding")},
    {"decodeToBinary", (PyCFunction)pythonBridgePytorchState_decodeToBinary, METH_VARARGS, PyDoc_STR("Decode Input to Binary String of 1 and 0")},
    {"decodeFromBinary", (PyCFunction)pythonBridgePytorchState_decodeFromBinary, METH_VARARGS, PyDoc_STR("Decode Input of Binary String of 1 and 0 to tokens")},
    {"runWithoutDecoding", (PyCFunction)pythonBridgePytorchState_runWithoutDecoding, METH_VARARGS, PyDoc_STR("Run the Decoder but do not generated Output")},
    {"cloneDecState", (PyCFunction)pythonBridgePytorchState_cloneDecState, METH_NOARGS, PyDoc_STR("Clone the current state of the Model")},
    {"inLastCreatedNodes", (PyCFunction)pythonBridgePytorchState_inLastCreatedNodes, METH_VARARGS, PyDoc_STR("Check if string is in the last decoded tokens")},
    {"resetDecState", (PyCFunction)pythonBridgePytorchState_resetDecState, METH_NOARGS, PyDoc_STR("Reset the state of the Model")},
    {NULL, NULL} /* Sentinel */
};

static struct PyMemberDef pythonBridgePytorchState_members[] = {
    {NULL} /* Sentinel */
};

static PyType_Slot pythonBridgePytorchState_slots[] = {
    {Py_tp_new, (void*)pythonBridgePytorchState_new},
    {Py_tp_init, (void*)pythonBridgePytorchState_init},
    {Py_tp_dealloc, (void*)pythonBridgePytorchState_dealloc},
    {Py_tp_members, pythonBridgePytorchState_members},
    {Py_tp_methods, pythonBridgePytorchState_methods},
    {0, 0}
};

static PyType_Spec spec_pythonBridgePytorchState = {
    "TorchModelCompress.pytorchModel.pytorchState",  // name
    sizeof(pythonBridgePytorchStateObject),          // basicsize
    0,                                               // itemsize
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        // flags
    pythonBridgePytorchState_slots                   // slots
};
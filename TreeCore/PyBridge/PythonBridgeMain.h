#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static int bridge_slot_mod_exec(PyObject* module);

static PyMethodDef py_methods[] = {
    {NULL, NULL} /* Sentinel */
};

static PyModuleDef_Slot py_slots[] = {{Py_mod_exec, bridge_slot_mod_exec},
#if PY_VERSION_HEX >= 0x030C0000
                                      {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030D0000
                                      {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
                                      {0, NULL}};

static struct PyModuleDef pyModule = {
    PyModuleDef_HEAD_INIT,
    "TorchModelCompress",                                         /*m_name*/
    PyDoc_STR("Compress data with a PyTorch probability model"),  /*m_doc*/
    0,                                                            /*m_size*/
    py_methods,                                                   /*m_methods*/
    py_slots,                                                     /*m_slots*/
};

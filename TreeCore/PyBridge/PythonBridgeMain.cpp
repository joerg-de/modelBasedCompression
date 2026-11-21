#include "PythonBridgeMain.h"
#include "PythonBridgePytorchModel.h"

PyMODINIT_FUNC PyInit_TorchModelCompress(void) {
    return PyModuleDef_Init(&pyModule);
}

static int bridge_slot_mod_exec(PyObject* module) {

    PyObject* tochMClass = PyType_FromSpec(&spec_pythonBridgePytorchModel);
    if (tochMClass == NULL) {
        return -1;
    }
    Py_INCREF(tochMClass);

    if (PyModule_AddObject(module, "pytorchModel", tochMClass) < 0) {
        Py_DECREF(tochMClass);
        return -1;
    }
    return 0;
}
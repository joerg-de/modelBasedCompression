#include "PythonCallBackThrottler.h"
#include <Python.h>

void PythonCallBackThrottler::process(int toPy) {
    if (runningI >= callBackIgnore) {
        runningI = 0;

        PyEval_RestoreThread(*_save);
        auto arglist = Py_BuildValue("(i)", toPy);
        auto result = PyObject_CallObject(callBack, arglist);
        Py_DECREF(arglist);
        if (result != NULL) {
            Py_DECREF(result);
        } else {
            *_save = PyEval_SaveThread();
            throw PythonCallBackThrottlerError();
        }
        *_save = PyEval_SaveThread();
    } else {
        runningI++;
    }
}
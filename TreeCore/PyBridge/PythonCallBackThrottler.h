#pragma once
#include <Python.h>
#include <exception>

class PythonCallBackThrottlerError : public std::exception {
  public:
    PythonCallBackThrottlerError() = default;
    const char* what() const noexcept override {
        return "Python callback throttler interrupted";
    }
};

class PythonCallBackThrottler {
  public:
    PythonCallBackThrottler(PyThreadState** _save, unsigned int callBackIgnore, PyObject* callBack)
        : _save(_save), callBackIgnore(callBackIgnore), callBack(callBack) {}
    void process(int toPy);

  private:
    unsigned int runningI = 0;
    PyThreadState** _save;
    unsigned int callBackIgnore;
    PyObject* callBack;
};
#pragma once

#include <string>
#include <torch/torch.h>

class Hidden {
  public:
    Hidden(c10::IValue* hidden) noexcept : hidden(hidden) {};
    Hidden() {};
    Hidden(c10::IValue* hidden, std::vector<std::string> ignor, int nextchar) noexcept
        : hidden(hidden), ignor(ignor), nextchar(nextchar) {};
    // string compare that are shorted must come first --> ab bevore abc
    std::vector<std::string> ignor;
    c10::IValue* hidden;
    int nextchar = -1;
};

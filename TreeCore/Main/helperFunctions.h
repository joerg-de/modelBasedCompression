#pragma once

unsigned int closest_power(double x) noexcept {
    double inv_log = std::log2(1 / x);
    unsigned int possible_results_low = static_cast<unsigned int>(std::floor(inv_log));
    unsigned int possible_results_up = static_cast<unsigned int>(std::ceil(inv_log));
    // 0 is not working in huffman tree
    if (possible_results_low == 0)
        return 1;
    if (((1. / (1 << possible_results_low)) - x) < (x - (1. / (1 << possible_results_up))))
        return possible_results_low;
    else
        return possible_results_up;
}

static std::string IntToString(int data, int length) noexcept {
    std::string res = "";
    if (data >= (1 << (length + 1)))
        return "";
    for (int i = 0; i < length; ++i) {
        if ((data % 2) == 0)
            res.append("0");
        else
            res.append("1");
        data = data / 2;
    }
    return res;
}
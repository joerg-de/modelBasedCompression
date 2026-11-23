#pragma once

unsigned int closest_power(double x) noexcept {
    if (x <= 0.0 || x > 0.375) return 1;
    double inv_log = -std::log2(x);
    unsigned int low = static_cast<unsigned int>(std::floor(inv_log));
    unsigned int high = low + 1;
	

    if (((1. / (1 << low)) - x) < (x - (1. / (1 << high))))
        return low;
    else
        return high;
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
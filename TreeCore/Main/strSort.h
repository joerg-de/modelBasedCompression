#pragma once

// a and b i true
// b and b is false
// aaa and a ist false
// a and aaa is true

static bool SimpleSortOperator(const std::string& a, const std::string& b) noexcept {
    auto lhs = a.cbegin();
    auto rhs = b.cbegin();

    for (; lhs != a.cend() && rhs != b.cend(); ++lhs, ++rhs) {
        unsigned char lhs_val = *lhs;
        unsigned char rhs_val = *rhs;

        if (lhs_val != rhs_val)
            return lhs_val < rhs_val;
    }

    return (rhs != b.cend());
}

static bool SimpleSortOperatorNoSize(const std::string& a, const std::string& b) noexcept {
    auto lhs = a.cbegin();
    auto rhs = b.cbegin();

    for (; lhs != a.cend() && rhs != b.cend(); ++lhs, ++rhs) {
        unsigned char lhs_val = *lhs;
        unsigned char rhs_val = *rhs;

        if (lhs_val != rhs_val)
            return lhs_val < rhs_val;
    }

    return false;
}

static bool SimpleSortOperatorNoSize(const char* a, const std::string& b) noexcept {
    auto rhs = b.cbegin();

    for (; *a && rhs != b.cend(); ++a, ++rhs) {
        unsigned char lhs_val = *a;
        unsigned char rhs_val = *rhs;

        if (lhs_val != rhs_val)
            return lhs_val < rhs_val;
    }

    return false;
}

struct WrapComparator {
    WrapComparator(std::vector<std::string>* wordlist) noexcept : wordlist(wordlist) {};
    bool operator()(unsigned int a, unsigned int b) const noexcept {
        return SimpleSortOperator((*wordlist)[a], (*wordlist)[b]);
    }
    std::vector<std::string>* wordlist;
};

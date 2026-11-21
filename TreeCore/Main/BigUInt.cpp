#include "BigUInt.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#define MANTISSA_SIZE 23

// begin() is LSB
// rbegin() is MSB

std::vector<bool> extract(BigUInt& Min, BigUInt& Max) {
    std::vector<bool> ret;
    while (Min.getSize() > 0) {
        if (Min.getMSB() != Max.getMSB())
            break;
        ret.insert(ret.begin(), Min.getMSB());
        Min.size--;
        Max.size--;
    }
    Min.clean();
    Max.clean();
    return ret;
}
bool BigUInt::getBitAt(unsigned int pos) const {
    uint64_t cur = Value[pos / sizeof(uint64_t) / 8];
    cur >>= pos % (sizeof(uint64_t) * 8);
    return cur & 1;
}
bool BigUInt::getMSB() const {
    return getBitAt(size - 1);
}
void BigUInt::clean() {
    auto NewSize = (size + (sizeof(uint64_t) * 8) - 1) / (sizeof(uint64_t) * 8);
    if (NewSize < Value.size()) {
        Value.erase(Value.begin() + NewSize, Value.rbegin().base());
    } else {
        Value.insert(Value.end(), NewSize - Value.size(), 0);
    }
    if (size % (sizeof(uint64_t) * 8) != 0) {
        *Value.rbegin() = *Value.rbegin() & ((1ll << (size % (sizeof(uint64_t) * 8))) - 1);
    }
}

bool isInStream(const BigUInt& Min, const BigUInt& Max, const BigUInt& data) {
    return data < Max && Min <= data;
}

BigUInt::BigUInt(const char* data, unsigned int size)
    : Value((size + (sizeof(uint64_t) * 8) - 1) / (sizeof(uint64_t) * 8), 0), size(size) {
    for (unsigned int i = 0; i < size; ++i) {
        auto toOr = static_cast<uint64_t>(data[size - i - 1] == '1') << (i % (sizeof(uint64_t) * 8));
        Value[i / (sizeof(uint64_t) * 8)] |= toOr;
    }
}

bool BigUInt::operator<(const BigUInt& rhs) const {
    auto thisI = Value.rbegin();
    auto rhsI = rhs.Value.rbegin();
    while (Value.rend() != thisI) {
        if (*thisI == *rhsI) {
        } else {
            return *thisI < *rhsI;
        }
        ++thisI;
        ++rhsI;
    }
    // when equal
    return false;
}

bool BigUInt::operator<(unsigned int rhs) const {
    auto thisI = Value.rbegin();
    for (int i = 0; i < static_cast<int>(Value.size()) - 1; ++i) {
        if (Value.rbegin()[i] == 0) {
        } else {
            return false;
        }
        ++thisI;
    }
    // when equal
    return *Value.begin() < rhs;
}

bool BigUInt::operator==(const BigUInt& rhs) const {
    auto thisI = Value.rbegin();
    auto rhsI = rhs.Value.rbegin();
    while (Value.rend() != thisI && rhsI != rhs.Value.rend()) {
        auto thisB = *thisI;
        auto rhsB = *rhsI;
        if (thisB != rhsB)
            return false;
        ++thisI;
        ++rhsI;
    }
    // when equal
    return true;
}
bool BigUInt::operator!=(const BigUInt& rhs) const {
    auto thisI = Value.rbegin();
    auto rhsI = rhs.Value.rbegin();
    while (Value.rend() != thisI && rhsI != rhs.Value.rend()) {
        auto thisB = *thisI;
        auto rhsB = *rhsI;
        if (thisB != rhsB)
            return true;
        ++thisI;
        ++rhsI;
    }
    // when equal
    return false;
}
bool BigUInt::operator<=(const BigUInt& rhs) const {
    auto thisI = Value.rbegin();
    auto rhsI = rhs.Value.rbegin();
    while (Value.rend() != thisI) {
        if (*thisI == *rhsI) {
        } else {
            return *thisI < *rhsI;
        }
        ++thisI;
        ++rhsI;
    }
    // when equal
    return true;
}
BigUInt BigUInt::operator+(const BigUInt& rhs) const {
    BigUInt ret;
    ret.size = std::max(getSize(), rhs.getSize());
    auto thisI = Value.begin();
    auto rhsI = rhs.Value.begin();
    bool C = false;
    while (Value.end() != thisI || rhsI != rhs.Value.end()) {
        uint64_t thisII;
        if (Value.end() != thisI)
            thisII = *thisI;
        else
            thisII = 0;
        uint64_t rhsII;
        if (rhsI != rhs.Value.end())
            rhsII = *rhsI;
        else
            rhsII = 0;
        ret.Value.push_back(thisII + rhsII + C);
        C = (*ret.Value.rbegin() < thisII);
        if (Value.end() != thisI)
            ++thisI;
        if (rhsI != rhs.Value.end())
            ++rhsI;
    }
    return ret;
}
BigUInt& BigUInt::operator++() {
    bool C = true;
    for (auto thisI = Value.begin(); Value.end() != thisI && C; ++thisI) {
        *thisI = *thisI + 1;
        C = (*thisI == 0);
    }
    return *this;
}

void BigUInt::AddWithOffsetStep32(uint64_t val, unsigned int offset) {
    unsigned int Voffset = offset / 2;
    bool C;
    if (offset & 1) {
        uint64_t val1 = static_cast<uint64_t>(val & 0xFFFFFFFF) << 32;
        uint64_t val2 = val >> 32;

        auto old = Value[Voffset];
        Value[Voffset] += val1;
        C = (Value[Voffset] < old);

        ++Voffset;
        old = Value[Voffset];
        Value[Voffset] += val2 + C;
        C = (Value[Voffset] < old);

        if (Value.end() != Value.begin() + Voffset)
            ++Voffset;
    } else {
        auto old = Value[Voffset];
        Value[Voffset] += val;
        C = (Value[Voffset] < old);

        ++Voffset;
    }
    auto thisI = Value.begin() + Voffset;
    while (Value.end() != thisI && C == true) {
        *thisI = *thisI + C;
        C = (*thisI == 0);
        ++thisI;
    }
}

BigUInt BigUInt::operator-(const BigUInt& rhs) const {
    BigUInt ret;
    ret.size = std::max(getSize(), rhs.getSize());
    auto thisI = Value.begin();
    auto rhsI = rhs.Value.begin();
    bool C = false;
    while (Value.end() != thisI || rhsI != rhs.Value.end()) {
        uint64_t thisII;
        if (Value.end() != thisI)
            thisII = *thisI;
        else
            thisII = 0;
        uint64_t rhsII;
        if (rhsI != rhs.Value.end())
            rhsII = *rhsI;
        else
            rhsII = 0;
        ret.Value.push_back(thisII - rhsII - C);
        C = (*ret.Value.rbegin() > thisII);
        if (Value.end() != thisI)
            ++thisI;
        if (rhsI != rhs.Value.end())
            ++rhsI;
    }
    return ret;
}

BigUInt& BigUInt::operator-=(const BigUInt& rhs) {
    auto rhsI = rhs.Value.begin();
    bool C = false;
    for (auto thisI = Value.begin(); Value.end() != thisI; ++thisI) {
        uint64_t rhsII;
        if (rhsI != rhs.Value.end())
            rhsII = *rhsI;
        else
            rhsII = 0;
        uint64_t tmp = *thisI - rhsII - C;
        C = (tmp > *thisI);
        *thisI = tmp;
        if (rhsI != rhs.Value.end())
            ++rhsI;
    }
    return *this;
}

BigUInt::BigUInt(unsigned int size, long long inValue)
    : Value((size + (sizeof(uint64_t) * 8) - 1) / (sizeof(uint64_t) * 8), 0), size(size) {
    Value[0] = inValue;
}
BigUInt::BigUInt(unsigned int size, bool value)
    : Value((size + (sizeof(uint64_t) * 8) - 1) / (sizeof(uint64_t) * 8), value ? 0xFFFFFFFFFFFFFFFF : 0), size(size) {
    clean();
}
BigUInt::BigUInt(const BigUInt& in, unsigned int size)
    : Value((size + (sizeof(uint64_t) * 8) - 1) / (sizeof(uint64_t) * 8), 0), size(size) {
    for (unsigned int i = 0; i < in.Value.size(); ++i)
        Value[i] = in.Value[i];
}

BigUInt BigUInt::div(float by, unsigned int ElementsLeftover, const BigUInt& Max, const BigUInt& Current) const {

    int exp;
    float baseF = std::frexp(by, &exp);
    // 23 bit mantissa max is 1 - epsilon
    baseF = baseF * (1 << (MANTISSA_SIZE + 1));
    unsigned int base = static_cast<unsigned int>(baseF);

    auto ret = *this * base;

    // this can not be negative
    int bitsToRemove = MANTISSA_SIZE - exp + 1;
    if (static_cast<int>(ret.getSize()) - bitsToRemove <= 0) // fallback
        return BigUInt(Max.getSize(), 1ll);
    bool C = ret.getBitAt(bitsToRemove - 1);
    ret.shiftRight(bitsToRemove);
    if (C)
        ++ret;

    BigUInt MaxExt(Max, Max.getSize() + 1);
    MaxExt -= ret;
    MaxExt -= Current;
    MaxExt.clean();
    if (MaxExt < ElementsLeftover || MaxExt.getMSB()) [[unlikely]] {
        BigUInt ElementLeftoverBUI(Max.getSize(), static_cast<long long>(ElementsLeftover));
        ret = Max - ElementLeftoverBUI - Current;
    }
    if (ret.isZero())
        return BigUInt(getSize(), 1ll);

    ret.ExpandTo(getSize());

    return ret;
}

void BigUInt::shiftRight(int bitsToRemove) {
    if (bitsToRemove > 0x3F) {
        shiftRight(bitsToRemove - 0x3F);
        bitsToRemove = 0x3F;
    }
    for (auto i = Value.begin(); i != Value.end(); ++i) {
        *i = *i >> bitsToRemove;
        if (i + 1 != Value.end())
            *i |= *(i + 1) << (0x40 - bitsToRemove);
    }
    size -= bitsToRemove;
    clean();
}
void BigUInt::shiftLeft(unsigned int bitsToAdd, bool val) {
    if (bitsToAdd > 0x3F) {
        shiftLeft(bitsToAdd - 0x3F, val);
        bitsToAdd = 0x3F;
    }
    size += bitsToAdd;
    clean();
    uint64_t tmp = val ? ((1ll << bitsToAdd) - 1) : 0;
    for (auto i = Value.begin(); i != Value.end(); ++i) {
        uint64_t tmpNext = *i >> (0x40 - bitsToAdd);
        *i = (*i << bitsToAdd) | tmp;
        tmp = tmpNext;
    }
    clean();
}

BigUInt BigUInt::operator*(const BigUInt& other) const {
    BigUInt result(getSize() + other.getSize(), 0ll); // 64 bit result

    for (auto i = 0; i < Value.size(); ++i) {
        unsigned long long i_1 = Value[i] & 0xFFFFFFFF;
        unsigned long long i_2 = Value[i] >> 32;
        if (i_1 != 0) {
            for (auto j = 0; j < other.Value.size(); ++j) {
                unsigned long long j_1 = other.Value[j] & 0xFFFFFFFF;
                unsigned long long j_2 = other.Value[j] >> 32;
                result.AddWithOffsetStep32(i_1 * j_1, i * 2 + j * 2);
                result.AddWithOffsetStep32(i_1 * j_2, i * 2 + j * 2 + 1);
            }
        }
        if (i_2 != 0) {
            for (auto j = 0; j < other.Value.size(); ++j) {
                unsigned long long j_1 = other.Value[j] & 0xFFFFFFFF;
                unsigned long long j_2 = other.Value[j] >> 32;
                result.AddWithOffsetStep32(i_2 * j_1, i * 2 + j * 2 + 1);
                result.AddWithOffsetStep32(i_2 * j_2, i * 2 + j * 2 + 2);
            }
        }
    }
    return result;
}

BigUInt BigUInt::operator*(unsigned int other) const {
    BigUInt result(getSize() + MANTISSA_SIZE + 1, 0ll); // 64 bit result
    for (auto i = 0; i < Value.size(); ++i) {
        unsigned long long j_1 = Value[i] & 0xFFFFFFFF;
        unsigned long long j_2 = Value[i] >> 32;
        result.AddWithOffsetStep32(other * j_1, i * 2);
        result.AddWithOffsetStep32(other * j_2, i * 2 + 1);
    }
    return result;
}

#ifdef DEBUG_PRINT
void BigUInt::print() const {
    std::cout << std::hex;
    std::cout << "(" << size << ")0x";
    if (Value.size() != (size + (sizeof(uint64_t) * 8) - 1) / (sizeof(uint64_t) * 8))
        std::cout << "Error";
    for (auto j = Value.rbegin(); j != Value.rend(); ++j)
        std::cout << *j;
}
#endif

// first bit is MSB
std::string BigUInt::toString() const {
    std::string ret;
    ret.reserve(size);
    for (auto j = size - 1; j > 0; --j)
        ret.push_back(getBitAt(j) ? '1' : '0');
    return ret;
}

void BigUInt::ExpandTo(unsigned int size) {
    this->size = size;
    clean();
}

bool BigUInt::isZero() const {
    for (auto i : Value)
        if (i != 0)
            return false;
    return true;
}

unsigned int BigUInt::getSizeNoPadding() const {
    unsigned int res = size;
    for (int i = size - 1; i > 0; --i) {
        if (getBitAt(i) == true)
            break;
        --res;
    }
    return res;
};
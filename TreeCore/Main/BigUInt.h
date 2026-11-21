#pragma once
#include <string>
#include <vector>

// #define DEBUG_PRINT

// begin() is LSB
// rbegin() is MSB
class BigUInt {
  public:
    BigUInt() : size(0) {};
    // extend only
    BigUInt(const BigUInt& in, unsigned int size);
    BigUInt(unsigned int size, long long value);
    BigUInt(unsigned int size, bool value);
    BigUInt(const char* data, unsigned int size);
    BigUInt div(float by, unsigned int minBitsLeftover, const BigUInt& Max, const BigUInt& Current) const;
    void shiftRight(int bitsToRemove);
    void shiftLeft(unsigned int bits, bool value);
    unsigned int getSize() const {
        return size;
    };
    unsigned int getSizeNoPadding() const;

    bool operator<(const BigUInt& rhs) const;
    bool operator<(unsigned int rhs) const;
    bool operator<=(const BigUInt& rhs) const;
    bool operator==(const BigUInt& rhs) const;
    bool operator!=(const BigUInt& rhs) const;
    BigUInt operator-(const BigUInt& other) const;
    BigUInt& operator-=(const BigUInt& rhs);
    BigUInt operator+(const BigUInt& other) const;
    BigUInt operator*(const BigUInt& other) const;
    BigUInt operator*(unsigned int other) const;
    BigUInt& operator++();
    bool isZero() const;
#ifdef DEBUG_PRINT
    void print() const;
#endif
    void ExpandTo(unsigned int size);
    bool getBitAt(unsigned int pos) const;
    std::string toString() const;

  private:
    void clean();
    bool getMSB() const;
    void AddWithOffsetStep32(uint64_t val, unsigned int offset);
    std::vector<uint64_t> Value;
    unsigned int size;
    friend std::vector<bool> extract(BigUInt& Min, BigUInt& Max);
};

// size of data is not checked
bool isInStream(const BigUInt& Min, const BigUInt& Max, const BigUInt& data);

std::vector<bool> extract(BigUInt& Min, BigUInt& Max);
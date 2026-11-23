#pragma once

#include "strSort.h"
#include <string>
#include <torch/torch.h>

// this + TreeNode is exactly 128 Byte big if you change that change stackAllocOpimizer.hpp

class Node {

  public:
    Node(const Node*& left, const Node*& right, double prob) noexcept : left(left), right(right), prob(prob) {};
    Node(double prob, std::string symbol, Hidden hidden) noexcept
        : left(nullptr), right(nullptr), symbol(symbol), hidden(hidden), prob(prob) {};
    double getHuffError() const noexcept {
        double inv_log = -std::log2(prob);
        unsigned int possible_results_low = static_cast<unsigned int>(std::floor(inv_log));
        unsigned int possible_results_up = possible_results_low + 1;

        // 0 is not working in huffman tree
        if (possible_results_low == 0) {
            return (prob - 0.5);
        }
        double x_low = ((1. / (1 << possible_results_low)) - prob);
        double x_up = (prob - (1. / (1 << possible_results_up)));
        if (x_low < x_up) {
            return x_low;
        } else {
            return x_up;
        }
    }

    double prob;
    const Node* left;
    const Node* right;
    std::string symbol;
    Hidden hidden;

    friend bool operator<(const Node& l, const Node& r) noexcept {
        auto pl = &l;
        auto pr = &r;
        if (l.prob == r.prob) {
            while (pl->symbol.size() == 0)
                pl = pl->right;
            while (pr->symbol.size() == 0)
                pr = pr->right;
            return SimpleSortOperator(pl->symbol, pr->symbol);
        } else {
            return l.prob <= r.prob;
        }
    }
};

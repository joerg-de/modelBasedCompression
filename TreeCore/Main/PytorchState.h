#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "Hidden.h"
#include "Node.h"
#include "PytorchModel.h"

#include "Settings.h"
#ifdef USE_CUSTOM_ALLOC
#include "stackAllocOptimizer.hpp"
#endif
#ifdef SUPPORT_RANGE_CODING
#include "BigUInt.h"
#endif

class PytorchState {
  public:
    PytorchState(std::shared_ptr<PytorchModel> parent);
    PytorchModel* getParent() const {
        return parent.get();
    }
    void resetHidden();
    std::shared_ptr<PytorchState> cloneState() const;
#ifdef SUPPORT_HUFFMAN
    bool inNodes(const char* toFind) const noexcept;
    void initWithText(const char* text, std::function<void(unsigned long long)> callBack, bool updateBlacklist = false);
    // do not use Decode/Encode after you called decodeToBinary with useEndMarker
    std::tuple<std::string, unsigned int, unsigned int>
    decodeToBinary(const char* text, std::function<void(unsigned long long)> callBack, bool useEndMarker = false,
                   double corrVactor = 1, unsigned int HuffMaxExtend = 32, double huffMin = 0.0078125);

    // do not use Decode/Encode after you called decodeFromBinary with useEndMarker
    //           res        , leftover
    std::tuple<std::string, int> decodeFromBinary(const char* data, std::function<void(unsigned long long)> callBack,
                                                  bool useEndMarker = false, double corrVactor = 1,
                                                  unsigned int HuffMaxExtend = 32, double huffMin = 0.0078125);
#endif
#ifdef SUPPORT_RANGE_CODING
    // dont need initWithText as en and decoding is nearly as fast
    // MinUsefullBitsSize must be bigger than log2(NumToken)
    void initRangeCoding(unsigned int NormalBufferSize = 64);
    std::tuple<std::string, int>
    decodeFromBinaryRangeCoding(const char* data, std::function<void(unsigned long long)> callBack,
                                bool useEndMarker = false, unsigned int MinUsefullBitsSize = 32,
                                unsigned int NormalBufferSize = 64, long long sizeToEncode = 0, double corrVactor = 1);
    // useEndMarker make sure you can continue if you dont set useEndMarker and continue the encoding can fail
    //         res        , leftover
    std::tuple<std::string, int>
    decodeToBinaryRangeCoding(const char* text, std::function<void(unsigned long long)> callBack,
                              bool useEndMarker = false, unsigned int MinUsefullBitsSize = 32,
                              unsigned int NormalBufferSize = 64, double corrVactor = 1, bool ignoreUnkToken = true);
    BigUInt getMin() {
        return Min;
    }
    BigUInt getMax() {
        return Max;
    }

  private:
    void checkSize(unsigned int BufferSize);
    void UpdateMinMax(BigUInt& Min, BigUInt& Max, unsigned int MinUsefullBitsSize, unsigned int NormalBufferSize) const;

  public:
#endif

  private:
    double correctBlacklist(unsigned int* last, float* output_list);
    void runForward(int nextChar, double corrVactor, const c10::IValue* in_hidden, c10::IValue* out_hidden,
                    at::Tensor* output_list);
#ifdef SUPPORT_HUFFMAN
    // sort for speedup so most used to least used
#ifdef USE_CUSTOM_ALLOC
    stack_allocator<Node> MainNodeAllocer;
    std::set<Node, std::less<Node>, stack_allocator<Node>>
    model_create_node_list(Hidden& hidden, double corrVactor, unsigned int HuffMaxExtend, double huffMin);
    void model_round_to_node(Hidden& hidden, std::set<Node, std::less<Node>, stack_allocator<Node>>& IO_Nodes,
                             double corrVactor, double base_prob = 1, const std::string& extra_token = "");
#else
    std::set<Node> model_create_node_list(Hidden& hidden, double corrVactor, unsigned int HuffMaxExtend,
                                          double huffMin);
    void model_round_to_node(Hidden& hidden, std::set<Node>& IO_Nodes, double corrVactor, double base_prob = 1,
                             const std::string& extra_token = "");
#endif
    std::vector<std::string> BuildMiddel(const Node* currentNode, unsigned int HuffMaxExtend);
#endif
    std::shared_ptr<PytorchModel> parent;
#ifdef SUPPORT_RANGE_CODING
    BigUInt Min;
    BigUInt Max;
#endif
#ifdef SUPPORT_HUFFMAN
#if !defined(USE_CUSTOM_ALLOC) or !defined(MSVCRT_DEL_OPTIMIZE)
    std::vector<Node> NodesVec;
#endif
    // global Garbage collection Speedup
    std::vector<c10::IValue> tensorGCVec;
    std::vector<Node> direct_codes;
#endif
    Hidden hiddenState;
    c10::IValue hiddenStateGC;
    bool finished = false;
};
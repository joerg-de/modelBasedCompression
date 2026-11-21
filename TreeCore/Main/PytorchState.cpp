#include "PytorchState.h"
#include "BigUInt.h"
#include "Settings.h"
#include "helperFunctions.h"

std::shared_ptr<PytorchState> PytorchState::cloneState() const {
    auto ret = make_shared<PytorchState>(parrent);
    ret->hiddenState = hiddenState;
    ret->hiddenStateGC = hiddenStateGC;
    ret->hiddenState.hidden = &ret->hiddenStateGC;
    ret->finished = finished;
#ifdef SUPPORT_RANGE_CODING
    ret->Min = Min;
    ret->Max = Max;
#endif
    return ret;
}

PytorchState::PytorchState(std::shared_ptr<PytorchModel> parrent) {
    this->parrent = parrent;
    resetHidden();
}

void PytorchState::resetHidden() {
    torch::NoGradGuard no_grad;
    if (auto Method = parrent->gettrochModel()->find_method("init_hidden")) {
        hiddenStateGC = Method->operator()({});
        hiddenState = Hidden(&hiddenStateGC);
    } else {
        hiddenStateGC = torch::zeros(parrent->hiddenSize);
        hiddenState = Hidden(&hiddenStateGC);
    }
    finished = false;
#ifdef SUPPORT_RANGE_CODING
    Min = BigUInt();
    Max = BigUInt();
#endif
}

void PytorchState::runForward(int nextChar, double corrVactor, const c10::IValue* in_hidden, c10::IValue* out_hidden,
                              at::Tensor* output_list) {
    static const auto opts = torch::TensorOptions().dtype(torch::kInt32);
    int* castArray = &parrent->IdxTranslates[nextChar];
    at::Tensor data = torch::from_blob(castArray, {1}, opts);
    std::vector<c10::IValue> FowdInput{data, *in_hidden};
    c10::IValue rawResForw;
    rawResForw = parrent->trochModel->forward(FowdInput);
    auto resForw = rawResForw.toTuple()->elements();
    auto output = resForw[0];
    *out_hidden = resForw[1];
    // item is slow use data_ptr on output_list see TensorBase.h TensorBody.h make sure it is contiguous
    if (output_list)
        *output_list = at::pow(output.toTensor()[0], corrVactor);
    return;
}
#ifdef SUPPORT_HUFFMAN
#ifdef USE_COSTOME_ALLOC
void PytorchState::model_round_to_node(Hidden& hidden, std::set<Node, std::less<Node>, stack_allocator<Node>>& IO_Nodes,
                                       double corrVactor, double base_prob, const std::string& extra_token)
#else
void PytorchState::model_round_to_node(Hidden& hidden, std::set<Node>& IO_Nodes, double corrVactor, double base_prob,
                                       const std::string& extra_token)
#endif
{
    if (hidden.nextchar != -1) {
        at::Tensor output_listKeppAlive;
        c10::IValue ai_hidden;
        runForward(hiddenState.nextchar, corrVactor, hidden.hidden, &ai_hidden, &output_listKeppAlive);
        tensorGCVec.emplace_back(std::move(ai_hidden));
        auto ai_hidden_de = &(*tensorGCVec.rbegin());
        // item is slow use data_ptr on output_list see TensorBase.h TensorBody.h make sure it is contiguous
        float* output_list = output_listKeppAlive.data_ptr<float>();

        unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());
        auto BloutIt = hidden.ignor.cbegin();

#ifdef ADJUST_PDF
        // correct factor that cant be used such as he after an t
        double CorrSum = 0;
        for (auto i = 0; i < NumToken; ++i) {
            auto iWord = parrent->token[i];

            while (SimpleSortOperator(*BloutIt, iWord) && !iWord.starts_with(*BloutIt)) [[unlikely]] {
                ++BloutIt;
            }
            if (!iWord.starts_with(*BloutIt)) [[likely]] {
                int idx = parrent->IdxTranslatesInverse[i];
                CorrSum += output_list[idx].item<double>();
            }
        }
        base_prob = base_prob / CorrSum;

        BloutIt = hidden.ignor.begin();
#endif
        // only soted if needed
        std::sort(hidden.ignor.begin(), hidden.ignor.end(), SimpleSortOperator);
        for (unsigned int i = 0; i < NumToken; ++i) {
            // parrent->IdxBlacklists should already be sorted
            // the rest is only soted if needed
            std::vector<std::string> blackList4Next = parrent->IdxBlacklists[i];
            auto iWord = parrent->token[i];
            if (BloutIt != hidden.ignor.cend()) {
                while (SimpleSortOperatorNoSize(*BloutIt, iWord)) [[unlikely]] {
                    ++BloutIt;
                    if (BloutIt == hidden.ignor.cend())
                        break;
                }
                if (BloutIt != hidden.ignor.cend()) {
                    if (iWord.starts_with(*BloutIt)) {
                        continue;
                    }
                    auto NextItter = BloutIt;
                    while (NextItter->starts_with(iWord)) [[unlikely]] {
                        blackList4Next.push_back(NextItter->substr(iWord.size()));
                        ++NextItter;
                        if (NextItter == hidden.ignor.cend())
                            break;
                    }
                }
            }

            double new_prop = output_list[parrent->IdxTranslatesInverse[i]] * base_prob;
            std::string new_token = extra_token + iWord;
            IO_Nodes.emplace(new_prop, new_token, Hidden(ai_hidden_de, std::move(blackList4Next), i));
        }
    } else {
        unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());
        for (unsigned int i = 0; i < NumToken; ++i) {
            std::string idx = parrent->token[i];
            double new_prop = parrent->initailProp.at(idx) * base_prob;
            std::string new_token = extra_token + idx;
            std::vector<std::string> blackout = parrent->IdxBlacklists[i];
            IO_Nodes.emplace(new_prop, new_token, Hidden(hiddenState.hidden, blackout, i));
        }
    }
}
#endif
#ifdef SUPPORT_RANGE_CODING

void PytorchState::UpdateMinMax(BigUInt& Min, BigUInt& Max, unsigned int MinUsefullBitsSize,
                                unsigned int NormalBufferSize) const {
    auto Diff = Max - Min;
    unsigned int ActiveSize = Diff.getSizeNoPadding();
    if (ActiveSize < MinUsefullBitsSize) {
        Min.shiftLeft(MinUsefullBitsSize - ActiveSize, false);
        Max.shiftLeft(MinUsefullBitsSize - ActiveSize, true);
    }
    if (Min.getSize() < NormalBufferSize) {
        Min.shiftLeft(NormalBufferSize - Min.getSize(), false);
        Max.shiftLeft(NormalBufferSize - Max.getSize(), true);
    }
}

void PytorchState::initRangeCoding(unsigned int NormalBufferSize) {
    checkSize(NormalBufferSize);
    Max = BigUInt(NormalBufferSize, true);
    Min = BigUInt(NormalBufferSize, false);
}

double PytorchState::CorrectBlacklist(unsigned int* last, float* output_list) {
    unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());
    *last = 0;
    double ret = 0;
    auto BloutIt = hiddenState.ignor.cbegin();
    for (unsigned int i = 0; i < NumToken; ++i) {
        // parrent->IdxBlacklists should already be sorted
        // the rest is only soted if needed
        auto iWord = &parrent->token[i];
        if (BloutIt != hiddenState.ignor.cend()) {
            while (SimpleSortOperatorNoSize(*BloutIt, *iWord)) [[unlikely]] {
                ++BloutIt;
                if (BloutIt == hiddenState.ignor.cend())
                    break;
            }
            if (BloutIt != hiddenState.ignor.cend()) {
                if (iWord->starts_with(*BloutIt)) {
                    continue;
                }
            }
        }
        ret += output_list[parrent->IdxTranslatesInverse[i]];
        *last = i;
    }
    return ret;
}

void PytorchState::checkSize(unsigned int BufferSize) {
    if (BufferSize <= std::ceil(std::log2(BufferSize)))
        throw std::runtime_error("BufferSize must be bigger then std::log2(BufferSize)");
}

std::tuple<std::string, int>
PytorchState::decodeFromBinaryRangeCoding(const char* data, std::function<void(unsigned long long)> callBack,
                                          bool useEndMarker, unsigned int MinUsefullBitsSize,
                                          unsigned int NormalBufferSize, long long sizeToEncode, double corrVactor) {
    checkSize(MinUsefullBitsSize);
    const char* textInital = data;
    if (finished)
        throw std::runtime_error("call after useEndMarker");
    if (Max.getSize() == 0)
        throw std::runtime_error("need to call initRangeCoding before using decodeFromBinaryRangeCoding");
    finished = useEndMarker;
    torch::NoGradGuard no_grad;
    unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());

    // guess a good value
    std::string res = "";
    signed long long leftSize = strlen(data);
    signed long long leftSizeToEncode = sizeToEncode == 0 ? leftSize : sizeToEncode;
    res.reserve(leftSize * PREALLOC_FACTOR_decodeToBinary);

    while (true) {
        auto Diff = Max - Min;
        auto MaxOld = Max;
        BigUInt DataAsBigUInt(data, Min.getSize());
        if (hiddenState.nextchar != -1) {
            at::Tensor output_listKeppAlive;
            runForward(hiddenState.nextchar, corrVactor, &hiddenStateGC, &hiddenStateGC, &output_listKeppAlive);
            // item is slow use data_ptr on output_list see TensorBase.h TensorBody.h make sure it is contiguous
            float* output_list = output_listKeppAlive.data_ptr<float>();

            // only soted if needed
            std::sort(hiddenState.ignor.begin(), hiddenState.ignor.end(), SimpleSortOperator);
            unsigned int i = 0;

            const std::string* iWord;
            unsigned int last;
            auto corfac = CorrectBlacklist(&last, output_list);
            auto BloutIt = hiddenState.ignor.cbegin();
            for (; i < NumToken; ++i) {
                // parrent->IdxBlacklists should already be sorted
                // the rest is only soted if needed
                iWord = &parrent->token[i];
                if (BloutIt != hiddenState.ignor.cend()) {
                    while (SimpleSortOperatorNoSize(*BloutIt, *iWord)) [[unlikely]] {
                        ++BloutIt;
                        if (BloutIt == hiddenState.ignor.cend())
                            break;
                    }
                    if (BloutIt != hiddenState.ignor.cend()) {
                        if (iWord->starts_with(*BloutIt)) {
                            continue;
                        }
                    }
                }

                float prop = static_cast<float>(output_list[parrent->IdxTranslatesInverse[i]] / corfac);
                auto Current = Diff.div(prop, NumToken - i, MaxOld, Min);
                Max = Current + Min;

                if (i == last) {
                    Max = MaxOld;
                    break;
                }

                if (isInStream(Min, Max, DataAsBigUInt)) {

                    break;
                }
                Min = Max;
            }
            BloutIt = hiddenState.ignor.cbegin();
            std::vector<std::string> blackList4Next = parrent->IdxBlacklists[i];
            if (BloutIt != hiddenState.ignor.cend()) {
                while (SimpleSortOperatorNoSize(*BloutIt, *iWord)) [[unlikely]] {
                    ++BloutIt;
                    if (BloutIt == hiddenState.ignor.cend())
                        break;
                }
                if (BloutIt != hiddenState.ignor.cend()) {
                    auto NextItter = BloutIt;
                    while (NextItter->starts_with(*iWord)) [[unlikely]] {
                        blackList4Next.push_back(NextItter->substr(iWord->size()));
                        ++NextItter;
                        if (NextItter == hiddenState.ignor.cend())
                            break;
                    }
                }
            }

            auto extracted = extract(Min, Max);

            data += extracted.size();
            leftSizeToEncode -= extracted.size();
            leftSize -= extracted.size();
            sizeToEncode -= extracted.size();
            res.append(parrent->token[i]);
            hiddenState = Hidden(hiddenState.hidden, blackList4Next, i);

        } else {
            unsigned int i = 0;

            for (; i < NumToken; ++i) {
                double prop = parrent->initailProp.at(parrent->token[i]);
                auto Current = Diff.div(static_cast<float>(prop), NumToken - i, MaxOld, Min);
                Max = Current + Min;
                if (isInStream(Min, Max, DataAsBigUInt)) {
                    break;
                }
                Min = Max;
            }
            if (i == NumToken) {
                Max = MaxOld;
                --i;
            }
            auto extracted = extract(Min, Max);
            data += extracted.size();
            leftSizeToEncode -= extracted.size();
            leftSize -= extracted.size();
            sizeToEncode -= extracted.size();
            res.append(parrent->token[i]);
            hiddenState = Hidden(&hiddenStateGC, parrent->IdxBlacklists[i], i);
        }
        callBack(data - textInital);
        UpdateMinMax(Min, Max, MinUsefullBitsSize, NormalBufferSize);
        if (!finished) {
            if (leftSizeToEncode <= Max.getSize())
                break;
        } else {
            if (sizeToEncode < 0)
                break;
        }
        if (leftSize < Max.getSize())
            throw std::runtime_error("not enought leftSize");
    }
    if (!finished)
        return std::make_tuple<>(res, leftSizeToEncode);
    else
        return std::make_tuple<>(res, sizeToEncode);
}

std::tuple<std::string, int> PytorchState::decodeToBinaryRangeCoding(const char* text,
                                                                     std::function<void(unsigned long long)> callBack,
                                                                     bool useEndMarker, unsigned int MinUsefullBitsSize,
                                                                     unsigned int NormalBufferSize, double corrVactor,
                                                                     bool ignoreUnkToken) {
    checkSize(MinUsefullBitsSize);
    if (Max.getSize() == 0)
        throw std::runtime_error("need to call initRangeCoding before using decodeToBinaryRangeCoding");
    torch::NoGradGuard no_grad;
    unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());

    const char* textInital = text;

    // guess a good value
    std::string res = "";
    signed long long leftSize = strlen(text);
    res.reserve(leftSize * PREALLOC_FACTOR_decodeFromBinary);

    bool hiddenStateToggel = true;
    c10::IValue hiddenStateGCAlt;
    while (*text) {
        auto Diff = Max - Min;
        // for rollback and limmiting the size of the cumulative elements
        auto MaxOld = Max;
        auto MinOld = Min;
        if (hiddenState.nextchar != -1) {
            at::Tensor output_listKeppAlive;
            if (hiddenStateToggel)
                runForward(hiddenState.nextchar, corrVactor, &hiddenStateGC, &hiddenStateGCAlt, &output_listKeppAlive);
            else
                runForward(hiddenState.nextchar, corrVactor, &hiddenStateGCAlt, &hiddenStateGC, &output_listKeppAlive);
            hiddenStateToggel = !hiddenStateToggel;
            // item is slow use data_ptr on output_list see TensorBase.h TensorBody.h make sure it is contiguous
            auto output_list = output_listKeppAlive.data_ptr<float>();

            unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());

            // only soted if needed
            std::sort(hiddenState.ignor.begin(), hiddenState.ignor.end(), SimpleSortOperator);
            unsigned int iBest = -1;
            signed long long iBestLen = 0;
            BigUInt BestMax;
            BigUInt BestMin;
            unsigned int last;
            auto corfac = CorrectBlacklist(&last, output_list);
            auto BloutIt = hiddenState.ignor.cbegin();
            for (unsigned int i = 0; i < NumToken; ++i) {
                if (SimpleSortOperatorNoSize(text, parrent->token[i]))
                    break;
                if (BloutIt != hiddenState.ignor.cend()) {
                    while (SimpleSortOperatorNoSize(*BloutIt, parrent->token[i])) [[unlikely]] {
                        ++BloutIt;
                        if (BloutIt == hiddenState.ignor.cend())
                            break;
                    }
                    if (BloutIt != hiddenState.ignor.cend()) {
                        if (parrent->token[i].starts_with(*BloutIt)) {
                            continue;
                        }
                    }
                }

                float prop = static_cast<float>(output_list[parrent->IdxTranslatesInverse[i]] / corfac);
                auto Current = Diff.div(prop, NumToken - i, MaxOld, Min);
                Max = Current + Min;

                if (i == last) {
                    Max = MaxOld;
                }

                signed long long len = parrent->token[i].size();
                if (len <= leftSize) [[likely]] {
                    if (memcmp(parrent->token[i].c_str(), text, len) == 0) [[unlikely]] {
                        if (iBestLen < len) {
                            iBest = i;
                            BestMax = Max;
                            BestMin = Min;
                            iBestLen = len;
                        }
                    }
                }

                Min = Max;
            }
            if (iBest == -1) {
                // rollback
                hiddenStateToggel = !hiddenStateToggel;
                Max = MaxOld;
                Min = MinOld;
                if (ignoreUnkToken) {
                    leftSize -= 1;
                    text += 1;
                    continue;
                } else {
                    throw std::runtime_error("Error token can't decoded Token at: " + std::string(text));
                }
            }

            // parrent->IdxBlacklists should already be sorted
            // the rest is only sorted if needed
            std::vector<std::string> blackList4Next;
            blackList4Next = parrent->IdxBlacklists[iBest];
            BloutIt = hiddenState.ignor.cbegin();
            auto iWord = parrent->token[iBest];
            if (BloutIt != hiddenState.ignor.cend()) {
                while (SimpleSortOperatorNoSize(*BloutIt, iWord)) [[unlikely]] {
                    ++BloutIt;
                    if (BloutIt == hiddenState.ignor.cend())
                        break;
                }
                if (BloutIt != hiddenState.ignor.cend()) {
                    auto NextItter = BloutIt;
                    while (NextItter->starts_with(iWord)) [[unlikely]] {
                        blackList4Next.push_back(NextItter->substr(iWord.size()));
                        ++NextItter;
                        if (NextItter == hiddenState.ignor.cend())
                            break;
                    }
                }
            }

            if (!useEndMarker) {
                size_t MaxBlacklistSize = 0;
                for (auto i : blackList4Next) {
                    if (i.size() > MaxBlacklistSize)
                        MaxBlacklistSize = i.size();
                }
                if (leftSize <= iBestLen + MaxBlacklistSize) {
                    // rollback
                    hiddenStateToggel = !hiddenStateToggel;
                    Max = MaxOld;
                    Min = MinOld;
                    break;
                }
            }

            hiddenState = Hidden(hiddenState.hidden, blackList4Next, iBest);
            if (iBest != last)
                Max = BestMax;
            else
                Max = MaxOld;
            Min = BestMin;

            auto extracted = extract(Min, Max);

            leftSize -= iBestLen;
            text += iBestLen;

            for (auto i = extracted.rbegin(); i != extracted.rend(); ++i)
                res.append(*i ? "1" : "0");

        } else {
            unsigned int i = 0;
            unsigned int iBest = 0;
            signed long long iBestLen = 0;
            BigUInt BestMax;
            BigUInt BestMin;
            for (; i < NumToken; ++i) {
                double prop = parrent->initailProp.at(parrent->token[i]);
                auto Current = Diff.div(static_cast<float>(prop), NumToken - i, MaxOld, Min);
                Max = Current + Min;
                signed long long len = parrent->token[i].size();
                if (memcmp(parrent->token[i].c_str(), text, std::min(len, leftSize + 1)) == 0) [[unlikely]] {
                    if (iBestLen < len) {
                        iBest = i;
                        BestMax = Max;
                        BestMin = Min;
                        iBestLen = len;
                    }
                }
                Min = Max;
            }
            if (iBest != NumToken - 1)
                Max = BestMax;
            else
                Max = MaxOld;
            Min = BestMin;

            if (iBest == -1) {
                // rollback
                Max = MaxOld;
                Min = MinOld;
                if (ignoreUnkToken) {
                    leftSize -= 1;
                    text += 1;
                    continue;
                } else {
                    throw std::runtime_error("Error token can't decoded Token at: " + std::string(text));
                }
            }
            if (!useEndMarker) {
                size_t MaxBlacklistSize = 0;
                for (auto i : parrent->IdxBlacklists[iBest]) {
                    if (i.size() > MaxBlacklistSize)
                        MaxBlacklistSize = i.size();
                }
                if (leftSize <= iBestLen + MaxBlacklistSize) {
                    // rollback
                    Max = MaxOld;
                    Min = MinOld;
                    break;
                }
            }

            auto extracted = extract(Min, Max);
            leftSize -= iBestLen;
            text += iBestLen;

            for (auto i = extracted.rbegin(); i != extracted.rend(); ++i)
                res.append(*i ? "1" : "0");
            hiddenState = Hidden(hiddenState.hidden, parrent->IdxBlacklists[iBest], iBest);
        }
        callBack(text - textInital);
        UpdateMinMax(Min, Max, MinUsefullBitsSize, NormalBufferSize);
    }
    if (!hiddenStateToggel)
        hiddenStateGC = hiddenStateGCAlt;
    return std::make_tuple(res, leftSize);
}
#endif

#ifdef SUPPORT_HUFFMAN
bool PytorchState::inNodes(const char* toFind) const noexcept {
#ifdef USE_COSTOME_ALLOC
    auto size = MainNodeAllocer.size();
    for (unsigned int i = 0; i < size; ++i) {
        if (strcmp(toFind, MainNodeAllocer.ItemAt(i)->_Myval.symbol.c_str()) == 0)
            return true;
    }
#else
    for (auto i : NodesVec) {
        if (strcmp(toFind, i.symbol.c_str()) == 0)
            return true;
    }
    for (auto i : direct_codes) {
        if (strcmp(toFind, i.symbol.c_str()) == 0)
            return true;
    }
#endif
    return false;
}

void PytorchState::initWithText(const char* text, std::function<void(unsigned long long)> callBack,
                                bool updateBlacklist) {
    if (finished)
        throw std::runtime_error("call after useEndMarker");
    torch::NoGradGuard no_grad;
    const char* textBegin;

    textBegin = text;

    unsigned int maxWordlen = 0;
    for (auto i : parrent->token)
        if (maxWordlen < i.size())
            maxWordlen = static_cast<unsigned int>(i.size());
    while (text[0]) {
        unsigned int remainingSize;
        if (hiddenState.nextchar != -1) {
            runForward(hiddenState.nextchar, 1, hiddenState.hidden, &hiddenStateGC, nullptr);
            hiddenState.hidden = &hiddenStateGC;

            unsigned int Cns = 0;
            unsigned int lengthLongest = 0;
            unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());
            for (unsigned int Cn = 0; Cn < NumToken; ++Cn) {
                unsigned int len = static_cast<unsigned int>(parrent->token[Cn].size());
                if (memcmp(parrent->token[Cn].c_str(), text, len) == 0) [[unlikely]] {
                    if (len > lengthLongest) {
                        Cns = Cn;
                        lengthLongest = len;
                    }
                }
            }
            text += lengthLongest;
            remainingSize = static_cast<unsigned int>(text - textBegin);
            hiddenState.nextchar = Cns;
            if (maxWordlen >= remainingSize && updateBlacklist) {
                auto BloutIt = hiddenState.ignor.begin();
                std::vector<std::string> blackList4Next = parrent->IdxBlacklists[Cns];
                auto iWord = parrent->token[Cns];
                if (BloutIt != hiddenState.ignor.cend()) {
                    while (SimpleSortOperatorNoSize(*BloutIt, iWord)) [[unlikely]] {
                        ++BloutIt;
                        if (BloutIt == hiddenState.ignor.cend())
                            break;
                    }
                    if (BloutIt != hiddenState.ignor.cend()) {
                        if (iWord.starts_with(*BloutIt))
                            continue;
                        auto NextItter = BloutIt;
                        while (NextItter->starts_with(iWord)) [[unlikely]] {
                            blackList4Next.push_back(NextItter->substr(iWord.size()));
                            ++NextItter;
                            if (NextItter == hiddenState.ignor.cend())
                                break;
                        }
                    }
                }
                std::sort(blackList4Next.begin(), blackList4Next.end(), SimpleSortOperator);

                hiddenState.ignor = blackList4Next;
            }
        } else {
            int Cns = 0;
            unsigned int lengthLongest = 0;
            unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());
            for (unsigned int Cn = 0; Cn < NumToken; ++Cn) {
                unsigned int len = static_cast<unsigned int>(parrent->token[Cn].size());
                if (memcmp(text, parrent->token[Cn].c_str(), len) == 0) [[unlikely]] {
                    if (len > lengthLongest) {
                        Cns = Cn;
                        lengthLongest = len;
                    }
                }
            }
            text += lengthLongest;
            remainingSize = static_cast<unsigned int>(text - textBegin);
            hiddenState.nextchar = Cns;
            if (updateBlacklist)
                hiddenState.ignor = parrent->IdxBlacklists[Cns];
        }
        callBack(remainingSize);
    }
}

std::tuple<std::string, unsigned int, unsigned int>
PytorchState::decodeToBinary(const char* text, std::function<void(unsigned long long)> callBack, bool useEndMarker,
                             double corrVactor, unsigned int HuffMaxExtend, double huffMin) {
    const char* textInital = text;
    if (finished)
        throw std::runtime_error("call after useEndMarker");
    finished = useEndMarker;
    torch::NoGradGuard no_grad;
    unsigned int NumToken = static_cast<unsigned int>(parrent->token.size());
    std::vector<const Node*> stack;
    unsigned int resStak = static_cast<unsigned int>(std::sqrt((HuffMaxExtend + 1) * NumToken) * 10);
    stack.reserve(resStak);

    // guess a good value
    std::string res = "";
    res.reserve(strlen(text) * PREALLOC_FACTOR_decodeToBinary);

#if !defined(USE_COSTOME_ALLOC) or !defined(MSVCRT_DEL_OPTIMIZE)
    // if there is a realloc the Programm crashes so dont do it the times 2 is needed because it has right and left
    NodesVec.reserve((HuffMaxExtend + 1) * parrent->token.size() * 2);
#endif
    tensorGCVec.reserve(HuffMaxExtend);

    size_t leftSize = strlen(text);
    size_t lengthLongest = 0;
    unsigned int maxDestr = 0;
    const Node* bestNode;

    while (text[0]) {
#if !defined(USE_COSTOME_ALLOC) or !defined(MSVCRT_DEL_OPTIMIZE)
        NodesVec.clear();
#endif
#ifdef USE_COSTOME_ALLOC
        MainNodeAllocer.reserve(static_cast<unsigned int>((HuffMaxExtend + 1) * parrent->token.size() * 2 + 1));
#endif
        tensorGCVec.clear();
        auto nodes = model_create_node_list(hiddenState, corrVactor, HuffMaxExtend, huffMin);
        while (nodes.size() > 1) {
            auto it = nodes.begin();
            const Node* right;
#if defined(USE_COSTOME_ALLOC) and defined(MSVCRT_DEL_OPTIMIZE)
            right = &(*it);
#else
            NodesVec.emplace_back(std::move(*it));
            right = &(*NodesVec.rbegin());
#endif

            auto itnext = std::next(it);
            const Node* left;
#if defined(USE_COSTOME_ALLOC) and defined(MSVCRT_DEL_OPTIMIZE)
            left = &(*itnext);
#else
            NodesVec.emplace_back(std::move(*itnext));
            left = &(*NodesVec.rbegin());
#endif

            nodes.erase(it, std::next(itnext));
            nodes.emplace_hint(nodes.begin(), left, right, right->prob + left->prob);
        }

#ifdef DEBUG_TREE
        for (const auto& node : NodesVec) {
            if (node.symbol.length() != 0) {
                std::cout << "node: " << node.symbol << " ";
                std::string code = "";
                const Node* no = &node;
                while (no) {
                    auto NoTemp = no;
                    no = nullptr;
                    for (const auto& nodeIn : NodesVec) {
                        if (nodeIn.right == NoTemp) {
                            code.append("0");
                            no = &nodeIn;
                            break;
                        }
                        if (nodeIn.left == NoTemp) {
                            code.append("1");
                            no = &nodeIn;
                            break;
                        }
                    }
                }
                std::reverse(code.begin(), code.end());
                std::cout << code << " " << node.prob << "\n" << std::flush;
            }
        }
#endif

        if (useEndMarker) {
            auto MidSymbols = BuildMiddel(&(*nodes.begin()), HuffMaxExtend);
            for (size_t i = 0; i < MidSymbols.size(); ++i) {
                if (strcmp(direct_codes[i].symbol.c_str(), text) == 0) [[unlikely]] {
                    res.append(MidSymbols[i]);
                    return std::make_tuple<>(res, 0, strlen(text));
                }
            }
        }

        // at last one must be found
        bestNode = nullptr;
        lengthLongest = 0;
        std::string bestCode;

        // calculte the full codes Tiefensuche Optimized
        auto currentNode = &(*nodes.begin());
        std::string currentCode = "";
        stack.clear();
        currentCode.reserve(resStak);

        while (1 == 1) {
            if (currentNode->left) {
                // do this here so copy knows how much to copy
                currentCode.push_back('0');
                stack.emplace_back(currentNode->left);
                currentNode = currentNode->right;
            } else {
                // check
                size_t len = strlen(currentNode->symbol.c_str());
                if (memcmp(currentNode->symbol.c_str(), text, std::min(len, leftSize)) == 0) [[unlikely]] {
                    if (len > lengthLongest) {
                        if (len > leftSize) {
                            if (!useEndMarker) {
                                return std::make_tuple<>(res, leftSize, 0);
                            } else {
                                if (maxDestr == 0)
                                    maxDestr = static_cast<unsigned int>(leftSize);
                            }
                        }
                        if (len <= leftSize) {
                            bestNode = currentNode;
                            lengthLongest = len;
                            bestCode = currentCode;
                        }
                    }
                }

                // remove leading 1
                unsigned int countLeading1 = 0;
                for (auto i = currentCode.rbegin(); i != currentCode.rend(); ++i) {
                    if (*i == '1')
                        ++countLeading1;
                    else
                        break;
                }
                if (countLeading1 != 0)
                    currentCode.resize(currentCode.size() - countLeading1);
                if (currentCode.size() == 0) [[unlikely]]
                    break;
                currentCode.pop_back();
                currentCode.push_back('1');
                currentNode = stack.back();
                stack.pop_back();
            }
        }
        if (!bestNode)
            break;
        // add it
        res.append(bestCode);
        hiddenStateGC = *(bestNode->hidden.hidden);
        hiddenState = bestNode->hidden;
        hiddenState.hidden = &hiddenStateGC;
        text += lengthLongest;
        leftSize -= lengthLongest;
        callBack(text - textInital);
    }
    return std::make_tuple<>(res, leftSize, maxDestr);
}

// res       , leftover
std::tuple<std::string, int> PytorchState::decodeFromBinary(const char* data,
                                                            std::function<void(unsigned long long)> callBack,
                                                            bool useEndMarker, double corrVactor,
                                                            unsigned int HuffMaxExtend, double huffMin) {
    if (finished)
        throw std::runtime_error("call after useEndMarker");
    finished = useEndMarker;
    torch::NoGradGuard no_grad;
    const char* initData = data;

    // guess a god value
    std::string res = "";
    res.reserve(strlen(data) * PREALLOC_FACTOR_decodeFromBinary);

    const Node* current = nullptr;

#if !defined(USE_COSTOME_ALLOC) or !defined(MSVCRT_DEL_OPTIMIZE)
    // if there is a realloc the Programm crashes so dont do it the times 2 is needed because it has right and left
    NodesVec.reserve((1 + HuffMaxExtend) * parrent->token.size() * 2);
#endif
    tensorGCVec.reserve(HuffMaxExtend + 1);

    while (data[0]) {
#ifdef USE_COSTOME_ALLOC
        MainNodeAllocer.reserve(static_cast<unsigned int>((1 + HuffMaxExtend) * parrent->token.size() * 2 + 1));
#endif
#if !defined(USE_COSTOME_ALLOC) or !defined(MSVCRT_DEL_OPTIMIZE)
        NodesVec.clear();
#endif
        tensorGCVec.clear();

        auto nodes = model_create_node_list(hiddenState, corrVactor, HuffMaxExtend, huffMin);

        while (nodes.size() > 1) {

            auto it = nodes.begin();
            const Node* right;
#if defined(USE_COSTOME_ALLOC) and defined(MSVCRT_DEL_OPTIMIZE)
            right = &(*it);
#else
            NodesVec.emplace_back(std::move(*it));
            right = &(*NodesVec.rbegin());
#endif

            auto itnext = std::next(it);
            const Node* left;
#if defined(USE_COSTOME_ALLOC) and defined(MSVCRT_DEL_OPTIMIZE)
            left = &(*itnext);
#else
            NodesVec.emplace_back(std::move(*itnext));
            left = &(*NodesVec.rbegin());
#endif

            nodes.erase(it, std::next(itnext));

            nodes.emplace_hint(nodes.begin(), left, right, right->prob + left->prob);
        }

        if (useEndMarker) {
            auto MidSymbols = BuildMiddel(&(*nodes.begin()), HuffMaxExtend);
            for (size_t i = 0; i < MidSymbols.size(); ++i) {
                if (strcmp(MidSymbols[i].c_str(), data) == 0) [[unlikely]] {
                    res.append(direct_codes[i].symbol);
                    return std::make_tuple(res, -static_cast<int>(strlen(direct_codes[i].symbol.c_str())));
                }
            }
        }

        // at last one must be found else it crashes

        auto currentTemp = &(*nodes.begin());
        auto dataTemp = data;
        bool found = false;

        while (*dataTemp) {
            if (dataTemp[0] == '0') {
                dataTemp++;
                currentTemp = currentTemp->right;
                if (currentTemp->left == nullptr) {
                    found = true;
                    break;
                }
            } else {
                dataTemp++;
                currentTemp = currentTemp->left;
                if (currentTemp->left == nullptr) {
                    found = true;
                    break;
                }
            }
        }
        if (found) {
            current = currentTemp;
            data = dataTemp;
        } else {
            break;
        }
        unsigned int ValToPy = static_cast<unsigned int>(data - initData);
        callBack(ValToPy);

        // add it
        res.append(current->symbol);
        hiddenStateGC = *(current->hidden.hidden);
        hiddenState = current->hidden;
        hiddenState.hidden = &hiddenStateGC;
    }
    return std::make_tuple(res, strlen(data));
}
#ifdef USE_COSTOME_ALLOC
std::set<Node, std::less<Node>, stack_allocator<Node>>
PytorchState::model_create_node_list(Hidden& hidden, double corrVactor, unsigned int HuffMaxExtend, double huffMin) {
#else
std::set<Node> PytorchState::model_create_node_list(Hidden& hidden, double corrVactor, unsigned int HuffMaxExtend,
                                                    double huffMin) {
#endif
#ifdef USE_COSTOME_ALLOC
    std::set<Node, std::less<Node>, stack_allocator<Node>> IO_ret_list(MainNodeAllocer);
#else
    std::set<Node> IO_ret_list;
#endif
    model_round_to_node(hidden, IO_ret_list, corrVactor);
    direct_codes.clear();

    double huffMinBlackout = huffMin * HUFF_MIN_BLACKOUT_MUL;

    // max to prevent very long loops
    for (unsigned int runs = 0; runs < HuffMaxExtend; ++runs) {
        auto curNode = IO_ret_list.rbegin();
        // getHuffError() is faster than caching huff error
        auto iTemp = curNode->getHuffError();
        for (auto i = IO_ret_list.rbegin(); i != IO_ret_list.rend(); ++i) {
            // huff_err cant be more than HUFF_MIN. HUFF_MIN cant be a bigger error than that of iTemp here
            if (i->prob < huffMinBlackout || i->prob < (iTemp * 3))
                break;
            if (iTemp < i->getHuffError()) {
                curNode = i;
                iTemp = i->getHuffError();
            }
        }
        if (iTemp < huffMin)
            break;

        direct_codes.emplace_back(std::move(*curNode));
        IO_ret_list.erase((++curNode).base());

        model_round_to_node(direct_codes.rbegin()->hidden, IO_ret_list, corrVactor, direct_codes.rbegin()->prob,
                            direct_codes.rbegin()->symbol);
    }
    return IO_ret_list;
}

std::vector<std::string> PytorchState::BuildMiddel(const Node* currentNode, unsigned int HuffMaxExtend) {
    // init
    std::vector<unsigned int> DC_CoPo;
    std::vector<std::string> Found(direct_codes.size(), "");
    DC_CoPo.reserve(HuffMaxExtend);
    for (auto i : direct_codes)
        DC_CoPo.push_back(closest_power(i.prob));

    for (size_t i = 0; i < Found.size(); ++i) {
        int Currdata = 0;
        int CurrentMod = 1;
        while (true) {
            auto data = IntToString(Currdata, DC_CoPo[i] + ((CurrentMod % 2) ? -(CurrentMod / 2) : (CurrentMod / 2)));
            if (data == "") {
                Currdata = 0;
                ++CurrentMod;
            }
            auto dataTemp = data.c_str();
            // copyed code from decodeFromBinary but cant merge for compiler optimisation
            bool found = false;
            const Node* currentTemp = currentNode;
            while (*dataTemp) {
                if (dataTemp[0] == '0') {
                    dataTemp++;
                    currentTemp = currentTemp->right;
                    if (currentTemp->left == nullptr) {
                        found = true;
                        break;
                    }
                } else {
                    dataTemp++;
                    currentTemp = currentTemp->left;
                    if (currentTemp->left == nullptr) {
                        found = true;
                        break;
                    }
                }
            }
            if (found) {
                Currdata++;
                continue;
            }
            if (std::find(Found.begin(), Found.end(), data) != Found.end()) {
                Currdata++;
                continue;
            }
            Found[i] = data;
            break;
        }
    }
    return Found;
}
#endif
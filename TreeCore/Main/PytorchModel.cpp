#include "PytorchModel.h"
#include "PytorchState.h"
#include "Settings.h"
#include "strSort.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>


// token devided by , need to escapte with \, bzw. \\ (.wor)
// initialProp double float pointer for probability of the words (.pro)
// hiddenSize vector of size (.tmp)

std::unique_ptr<PytorchModel> PytorchModel::clone() const {
    auto res = std::make_unique<PytorchModel>();

    res->torchModel = gettorchModel();
    res->embeddedFiles = embeddedFiles;
    res->token = token;
    res->idxBlacklists = idxBlacklists;
    res->idxTranslates = idxTranslates;
    res->idxTranslatesInverse = idxTranslatesInverse;
    res->initialProp = initialProp;
    res->hiddenSize = hiddenSize;
    res->srcPath = srcPath;

    return res;
}

PytorchModel::PytorchModel(std::string path) {
    torchModel = std::make_shared<torch::jit::script::Module>();
    embeddedFiles["wor.wor"] = "";
    embeddedFiles["tmp.tmp"] = "";
    embeddedFiles["pro.pro"] = "";
    embeddedFiles["opt.opt"] = "";

    *torchModel = torch::jit::load(path, std::nullopt, embeddedFiles);

    std::unique_ptr<std::basic_istream<char, std::char_traits<char>>> optSource =
        std::make_unique<std::ifstream>(path.substr(0, path.size() - 3) + ".opt");
    if (optSource->fail()) {
        auto emFind = embeddedFiles.find("opt.opt");
        if (emFind != embeddedFiles.end() && emFind->second.size() != 0)
            optSource = std::make_unique<std::stringstream>(emFind->second);
        else
            throw std::runtime_error("Optimizer settings cant be opend (.opt)");
    }
    int freez;
    int optimize_numerics;
    int optimize_for_inference;
    *optSource >> freez;
    *optSource >> optimize_numerics;
    *optSource >> optimize_for_inference;
    if (freez == 1) {
        if (auto Method = torchModel->find_method("init_hidden")) {
            *torchModel = torch::jit::freeze(*torchModel, {{"init_hidden"}}, (optimize_numerics == 1));
        } else {
            *torchModel = torch::jit::freeze(*torchModel, std::nullopt, (optimize_numerics == 1));
        }
    }
    if (optimize_for_inference == 1)
        *torchModel = torch::jit::optimize_for_inference(*torchModel);

    processLoadMain(path);
}
PytorchModel::PytorchModel(std::string path, const std::shared_ptr<PytorchModel>& loadedModel) {
    torchModel = loadedModel->gettorchModel();
    embeddedFiles = loadedModel->embeddedFiles;
    processLoadMain(path);
}
std::shared_ptr<PytorchState> PytorchModel::generateState(std::shared_ptr<PytorchModel> self) {
    return std::make_shared<PytorchState>(self);
}

std::unique_ptr<std::basic_istream<char, std::char_traits<char>>> PytorchModel::loadCsv(const std::string& basePath,
                                                                                         const std::string& fileKey,
                                         const std::string& ext, bool throwOnError) const {
	std::string csvPath = basePath.substr(0, basePath.size() - 3) + ext;
    auto csvFile = std::make_unique<std::ifstream>(csvPath);
	if (!csvFile->fail())
		return csvFile;

	auto emFind = embeddedFiles.find(fileKey);
	if (emFind != embeddedFiles.end() && !emFind->second.empty()) {
        return std::make_unique<std::stringstream>(emFind->second);
	}
    if (throwOnError)
	    throw std::runtime_error("Word list file (" + std::string(ext) + ") not found");
    return nullptr;
}

void PytorchModel::processLoadMain(std::string path) {
    srcPath = path;
    // load words
    token.clear();
    std::string word;
	auto csvSource = loadCsv(path, "wor.wor", ".wor");
    *csvSource >> std::noskipws;
    typedef std::istream_iterator<char> CharIter;
    for (CharIter it(*csvSource); it != CharIter(); ++it) {
        char c = *it;
        switch (c) {
        case '\\':
            ++it;
            {
                char cNext = *it;
                if (cNext == '\\' || cNext == ',') {
                    word.push_back(cNext);
                } else {
                    throw std::runtime_error("error escaping Words");
                }
            }
            break;
        case ',':
            if (word != "\n\r" && word != "\r\n") {
                token.push_back(word);
            } else {
                token.push_back("\n");
            }
            word.clear();
            break;
        default:
            word.push_back(c);
            break;
        }
    }
    if (!word.empty())
        token.push_back(word);

#ifdef DEBUG_LOAD
    std::cout << "tokens:\n";
    for (const auto& i : token)
        std::cout << i << " " << "\n";
#endif
    // load propabilitys
    initialProp.clear();
    auto proSource = loadCsv(path, "pro.pro", ".pro");
    std::string line;
    unsigned int i = 0;
    while (std::getline(*proSource, line)) {
        double val;
        try {
            val = stod(line);
        } catch (std::runtime_error e) {
            throw std::runtime_error("Failed to decode Propabillity line " + line);
        }
        if (std::isnan(val)) {
            throw std::runtime_error("Failed to decode Propabillity line " + line);
        }
        if (i >= token.size()) {
            throw std::runtime_error("Propabillity list to long");
        }
        initialProp[token[i]] = val;
        ++i;
    }
#ifdef DEBUG_LOAD
    std::cout << "pro:\n";
    for (auto i : initialProp)
        std::cout << i.first << " " << i.second << "\n";
#endif
    // load size of carryover
    hiddenSize.clear();
    auto tmpSource = loadCsv(path, "tmp.tmp", ".tmp", false);
    // tmpSource not need for init as that is done by the model itself
    if (tmpSource && !tmpSource->fail()) {
        while (std::getline(*tmpSource, line)) {
            int val;
            try {
                val = stoi(line);
            } catch (std::runtime_error e) {
                throw std::runtime_error("Failed to decode temp line " + line);
            }
            if (val <= 0) {
                throw std::runtime_error("Failed to decode temp line " + line);
            }
            hiddenSize.push_back(val);
        }
    }
#ifdef DEBUG_LOAD
    std::cout << "tmp\n";
    for (auto i : hiddenSize)
        std::cout << i << " " << "\n";
#endif

    genIdxBlacklists();

#ifdef DEBUG_LOAD
    std::cout << "end Load\n";
#endif
}

int PytorchModel::startsWithEnd(const std::string& token,const std::string& black) const {
    for (unsigned int i = 0; i < black.size(); ++i) {
        if (token.ends_with(black.substr(0, black.size() - i)))
            return static_cast<int>(black.size()) - i;
    }
    return -1;
}

void PytorchModel::loadBlacklistLine(const std::string& line, std::string& errorText, bool force) {
    for (int i = 0; i < token.size(); ++i) {
        auto inLinePos = token[i].find(line);
        if (inLinePos != std::string::npos) {
            if (force) {
                std::string addToAll = token[i].substr(0, inLinePos + line.size());
                // this slows everything down but has to be done
                for (int i = 0; i < token.size(); ++i) {
                    idxBlacklists[i].push_back(addToAll);
                    std::sort(idxBlacklists[i].begin(), idxBlacklists[i].end(), SimpleSortOperator);
                }
            } else {
                errorText += "Can't emplace \"" + line + "\" as it is fully covered by a token.(" + token[i] + ")\n";
                continue;
            }
        }
        auto offset = startsWithEnd(token[i], line);
        if (offset != -1) {
            std::string rest = line.substr(offset);
            idxBlacklists[i].push_back(rest);
            std::sort(idxBlacklists[i].begin(), idxBlacklists[i].end(), SimpleSortOperator);
        }
    }
}

void PytorchModel::loadBlacklist(std::string path, bool Capitalize, bool force) {
    std::string errorText = "";

    std::ifstream blackSource(path);
    if (blackSource.fail()) {
        throw std::runtime_error("Failed to open blacklist");
    }
    std::string line;
    while (std::getline(blackSource, line)) {
        loadBlacklistLine(line, errorText, force);
        if (Capitalize) {
            bool changedSomething = false;
            for (auto& ch : line) {
                if (std::isalpha(static_cast<unsigned char>(ch))) {
                    ch = std::toupper(static_cast<unsigned char>(ch));
                    changedSomething = true;
                    break;
                }
            }
            if (changedSomething)
                loadBlacklistLine(line, errorText, force);
        }
    }
    if (errorText != "")
        throw std::runtime_error(
            errorText +
            "Still apply the blacklist where it is possible. Consider using force True if you use the API.");
}

void PytorchModel::genIdxBlacklists() {
    idxTranslates.resize(token.size());
    std::iota(idxTranslates.begin(), idxTranslates.end(), 0);
    // sort them
    WrapComparator cmp(&token);
    std::sort(idxTranslates.begin(), idxTranslates.end(), cmp);
    std::vector<std::string> newWords;
    newWords.resize(idxTranslates.size());
    idxTranslatesInverse.resize(idxTranslates.size());
    int idx2 = 0;
    for (auto i : idxTranslates) {
#ifdef DEBUG_LOAD
        std::cout << "resort " << i << " " << idx2 << '\n' << std::flush;
#endif
        newWords[idx2] = token[i];
        idxTranslatesInverse[idx2] = i;
        idx2++;
    }
    token = std::move(newWords);

    idxBlacklists.clear();
    idxBlacklists.resize(idxTranslates.size());

#ifdef DEBUG_LOAD
    std::cout << "load";
#endif

    for (int i = 0; i < token.size() - 1; ++i) {
        auto j = i + 1;
        while (j < token.size() && token[j].starts_with(token[i])) {
            auto rest = token[j].substr(token[i].size());

            // if that is blacked out dont use the next
            if (!idxBlacklists[i].empty() && rest.starts_with(*idxBlacklists[i].rbegin())) {
                ++j;
                continue;
            }

            idxBlacklists[i].push_back(rest);
            ++j;
        }
#ifdef DEBUG_LOAD
        std::cout << "blackout: " << token[i] << " " << idxBlacklists[i].size() << " ";
        for (const auto& logBlack : idxBlacklists[i])
            std::cout << "|" << logBlack << "|";
        std::cout << '\n' << std::flush;
#endif
    }
}
std::set<std::string> PytorchModel::listU8Chars() const {
    std::set<std::string> ret;
    for (const auto& i : token) {
        auto it = i.begin();
        auto end = i.end();
        while (it != end) {
            unsigned char current = static_cast<unsigned char>(*it);
            size_t len = 1;
            if (current < 0x80) len = 1;
            else if (current < 0xC0) ++it; // Invalid continuation, skip
            else if (current < 0xE0) len = 2;
            else if (current < 0xF0) len = 3;
            else if (current < 0xF8) len = 4;
            else if (current < 0xFC) len = 5;
            else if (current < 0xFE) len = 6;
            else ++it; // Invalid start byte

            if (it + len > end) {
                ++it; // Truncated sequence, skip safely
                continue;
            }
            ret.emplace(std::string(it, it + len));
            std::advance(it, len);
        }
    }
    return ret;
}

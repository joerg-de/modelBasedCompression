#include "PytorchModel.h"
#include "PytorchState.h"
#include "Settings.h"
#include "strSort.h"
#include <fstream>
#include <stdexcept>

// token devided by , need to escapte with \, bzw. \\ (.wor)
// initailProp double float pointer for probability of the words (.pro)
// hiddenSize vector of size (.tmp)

std::unique_ptr<PytorchModel> PytorchModel::clone() const {
    auto res = std::make_unique<PytorchModel>();

    res->trochModel = gettrochModel();
    res->EmbeddedFiles = EmbeddedFiles;
    res->token = token;
    res->IdxBlacklists = IdxBlacklists;
    res->IdxTranslates = IdxTranslates;
    res->IdxTranslatesInverse = IdxTranslatesInverse;
    res->initailProp = initailProp;
    res->hiddenSize = hiddenSize;
    res->srcPath = srcPath;

    return res;
}

PytorchModel::PytorchModel(std::string path) {
    trochModel = std::make_shared<torch::jit::script::Module>();
    EmbeddedFiles["wor.wor"] = "";
    EmbeddedFiles["tmp.tmp"] = "";
    EmbeddedFiles["pro.pro"] = "";
    EmbeddedFiles["opt.opt"] = "";

    *trochModel = torch::jit::load(path, std::nullopt, EmbeddedFiles);

    std::unique_ptr<std::basic_istream<char, std::char_traits<char>>> optSource =
        std::make_unique<std::ifstream>(path.substr(0, path.size() - 3) + ".opt");
    if (optSource->fail()) {
        auto emFind = EmbeddedFiles.find("opt.opt");
        if (emFind != EmbeddedFiles.end() && emFind->second.size() != 0)
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
        if (auto Method = trochModel->find_method("init_hidden")) {
            *trochModel = torch::jit::freeze(*trochModel, {{"init_hidden"}}, (optimize_numerics == 1));
        } else {
            *trochModel = torch::jit::freeze(*trochModel, std::nullopt, (optimize_numerics == 1));
        }
    }
    if (optimize_for_inference == 1)
        *trochModel = torch::jit::optimize_for_inference(*trochModel);

    ProcessLoadMain(path);
}
PytorchModel::PytorchModel(std::string path, const std::shared_ptr<PytorchModel>& loadedModel) {
    trochModel = loadedModel->gettrochModel();
    EmbeddedFiles = loadedModel->EmbeddedFiles;
    ProcessLoadMain(path);
}
std::shared_ptr<PytorchState> PytorchModel::generateState(std::shared_ptr<PytorchModel> self) {
    return std::make_shared<PytorchState>(self);
}

void PytorchModel::ProcessLoadMain(std::string path) {
    srcPath = path;
    // load words
    token.clear();
    std::string word;
    std::unique_ptr<std::basic_istream<char, std::char_traits<char>>> csvSource =
        std::make_unique<std::ifstream>(path.substr(0, path.size() - 3) + ".wor");
    if (csvSource->fail()) {
        auto emFind = EmbeddedFiles.find("wor.wor");
        if (emFind != EmbeddedFiles.end() && emFind->second.size() != 0)
            csvSource = std::make_unique<std::stringstream>(emFind->second);
        else
            throw std::runtime_error("Word list cant be opend (.wor)");
    }
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
    initailProp.clear();
    std::unique_ptr<std::basic_istream<char, std::char_traits<char>>> proSource =
        std::make_unique<std::ifstream>(path.substr(0, path.size() - 3) + ".pro");
    if (proSource->fail()) {
        auto emFind = EmbeddedFiles.find("pro.pro");
        if (emFind != EmbeddedFiles.end() && emFind->second.size() != 0)
            proSource = std::make_unique<std::stringstream>(emFind->second);
        else
            throw std::runtime_error("Propabillity list cant be opend (.pro)");
    }
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
        initailProp[token[i]] = val;
        ++i;
    }
#ifdef DEBUG_LOAD
    std::cout << "pro:\n";
    for (auto i : initailProp)
        std::cout << i.first << " " << i.second << "\n";
#endif
    // load size of carryover
    hiddenSize.clear();
    std::unique_ptr<std::basic_istream<char, std::char_traits<char>>> tmpSource =
        std::make_unique<std::ifstream>(path.substr(0, path.size() - 3) + ".tmp");
    if (tmpSource->fail()) {
        auto emFind = EmbeddedFiles.find("tmp.tmp");
        if (emFind != EmbeddedFiles.end() && emFind->second.size() != 0)
            tmpSource = std::make_unique<std::stringstream>(emFind->second);
    }
    // tmpSource not need for init as that is done by the model itself
    if (!tmpSource->fail()) {
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

    GenIdxBlacklists();

#ifdef DEBUG_LOAD
    std::cout << "end Load\n";
#endif
}

int PytorchModel::startsWithEnd(std::string token, std::string black) {
    for (unsigned int i = 0; i < black.size(); ++i) {
        if (token.ends_with(black.substr(0, black.size() - i)))
            return static_cast<int>(black.size()) - i;
    }
    return -1;
}

void PytorchModel::loadBlacklistLine(std::string line, std::string& errorText, bool force) {
    for (int i = 0; i < token.size(); ++i) {
        auto inLinePos = token[i].find(line);
        if (inLinePos != std::string::npos) {
            if (force) {
                std::string addToAll = token[i].substr(0, inLinePos + line.size());
                // this slows everything down but has to be done
                for (int i = 0; i < token.size(); ++i) {
                    IdxBlacklists[i].push_back(addToAll);
                    std::sort(IdxBlacklists[i].begin(), IdxBlacklists[i].end(), SimpleSortOperator);
                }
            } else {
                errorText += "Can't emplace \"" + line + "\" as it is fully covered by a token.(" + token[i] + ")\n";
                continue;
            }
        }
        auto offset = startsWithEnd(token[i], line);
        if (offset != -1) {
            std::string rest = line.substr(offset);
            IdxBlacklists[i].push_back(rest);
            std::sort(IdxBlacklists[i].begin(), IdxBlacklists[i].end(), SimpleSortOperator);
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
            for (unsigned int i = 0; i <= line.length(); i++) {
                if (std::isalpha(line[i])) {
                    line[i] = std::toupper(line[i]);
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

void PytorchModel::GenIdxBlacklists() {
    IdxTranslates.resize(token.size());
    for (unsigned int i = 0; i < token.size(); ++i)
        IdxTranslates[i] = i;
    // sort them
    WrapComparator cmp(&token);
    std::sort(IdxTranslates.begin(), IdxTranslates.end(), cmp);
    std::vector<std::string> newWords;
    newWords.resize(IdxTranslates.size());
    IdxTranslatesInverse.resize(IdxTranslates.size());
    int idx2 = 0;
    for (auto i : IdxTranslates) {
#ifdef DEBUG_LOAD
        std::cout << "resort " << i << " " << idx2 << '\n' << std::flush;
#endif
        newWords[idx2] = token[i];
        IdxTranslatesInverse[idx2] = i;
        idx2++;
    }
    token = std::move(newWords);

    IdxBlacklists.clear();
    IdxBlacklists.resize(IdxTranslates.size());

#ifdef DEBUG_LOAD
    std::cout << "load";
#endif

    for (int i = 0; i < token.size() - 1; ++i) {
        auto j = i + 1;
        while (j < token.size() && token[j].starts_with(token[i])) {
            auto rest = token[j].substr(token[i].size());

            // if that is blacked out dont use the next
            if (!IdxBlacklists[i].empty() && rest.starts_with(*IdxBlacklists[i].rbegin())) {
                ++j;
                continue;
            }

            IdxBlacklists[i].push_back(rest);
            ++j;
        }
#ifdef DEBUG_LOAD
        std::cout << "blackout: " << token[i] << " " << IdxBlacklists[i].size() << " ";
        for (const auto& logBlack : IdxBlacklists[i])
            std::cout << "|" << logBlack << "|";
        std::cout << '\n' << std::flush;
#endif
    }
}
std::set<std::string> PytorchModel::listU8Chars() const {
    std::set<std::string> ret;

    for (auto i : token) {
        for (auto j = i.begin(); j != i.end();) {
            std::string res = "";
            unsigned char current = *j;
            if (current < 0x80) {
                res.push_back(*j++);
            } else if (current > 0xfe) {
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
            } else if (current > 0xfc) {
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
            } else if (current > 0xf8) {
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
            } else if (current > 0xf0) {
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
            } else if (current > 0xe0) {
                res.push_back(*j++);
                res.push_back(*j++);
                res.push_back(*j++);
            } else if (current > 0xc0) {
                res.push_back(*j++);
                res.push_back(*j++);
            } else {
                // error not utf8 do nothing
                ++j;
            }
            ret.emplace(std::move(res));
        }
    }
    return ret;
}

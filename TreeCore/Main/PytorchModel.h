#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <string>

#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <torch/torch.h>

class PytorchState;

class PytorchModel {
  public:
    PytorchModel() = default;
    PytorchModel(std::string path);
    PytorchModel(std::string path, const std::shared_ptr<PytorchModel>& loadedModel);
    std::string getSrcPath() const {
        return srcPath;
    }
    std::shared_ptr<torch::jit::script::Module> gettorchModel() const {
        return torchModel;
    }
    std::set<std::string> listU8Chars() const;
    void loadBlacklist(std::string path, bool capitalize = true, bool force = false);
    void loadBlacklistLine(const std::string& line, std::string& errorText, bool force);
    std::shared_ptr<PytorchState> generateState(std::shared_ptr<PytorchModel> self);
    std::unique_ptr<PytorchModel> clone() const;

  private:
    std::unique_ptr<std::basic_istream<char, std::char_traits<char>>>
                      loadCsv(const std::string& basePath, const std::string& fileKey, const std::string& ext, bool throwOnError = true) const;
    void genIdxBlacklists();
    void processLoadMain(std::string path);
    int startsWithEnd(const std::string& token,const std::string& black) const;
    // sort for speedup
    std::vector<std::string> token;
    std::vector<std::vector<std::string>> idxBlacklists;
    std::vector<int> idxTranslates;
    std::vector<int> idxTranslatesInverse;
    std::shared_ptr<torch::jit::script::Module> torchModel;
    std::map<std::string, double> initialProp;
    std::vector<int64_t> hiddenSize;
    std::string srcPath;

  public:
    mutable std::mutex running_mutex;

  private:
    torch::jit::ExtraFilesMap embeddedFiles;
    friend class PytorchState;
};

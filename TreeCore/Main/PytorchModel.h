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
    std::shared_ptr<torch::jit::script::Module> gettrochModel() const {
        return trochModel;
    }
    std::set<std::string> listU8Chars() const;
    void loadBlacklist(std::string path, bool capitalize = true, bool force = false);
    void loadBlacklistLine(std::string line, std::string& errorText, bool force);
    std::shared_ptr<PytorchState> generateState(std::shared_ptr<PytorchModel> self);
    std::unique_ptr<PytorchModel> clone() const;

  private:
    void GenIdxBlacklists();
    void ProcessLoadMain(std::string path);
    int startsWithEnd(std::string token, std::string black);
    // sort for speedup
    std::vector<std::string> token;
    std::vector<std::vector<std::string>> IdxBlacklists;
    std::vector<int> IdxTranslates;
    std::vector<int> IdxTranslatesInverse;
    std::shared_ptr<torch::jit::script::Module> trochModel;
    std::map<std::string, double> initailProp;
    std::vector<int64_t> hiddenSize;
    std::string srcPath;

  public:
    mutable std::mutex running_mutex;

  private:
    torch::jit::ExtraFilesMap EmbeddedFiles;
    friend class PytorchState;
};

#include "PytorchModel.h"
#include "PytorchState.h"
#include <memory>
#include <string>
#include <vector>

static std::mutex Cache_mutex;
static std::vector<std::weak_ptr<PytorchModel>> loadedModels;

std::shared_ptr<PytorchModel> getModelAt(std::string path) {
    std::shared_ptr<PytorchModel> res;
    const std::lock_guard<std::mutex> lock(Cache_mutex);

    // remove unloaded
    for (auto i = loadedModels.begin(); i != loadedModels.end();) {
        if (i->expired()) {
            i = loadedModels.erase(i);
        } else {
            ++i;
        }
    }
    for (auto i = loadedModels.begin(); i != loadedModels.end(); ++i) {
        auto currentPytorchModel = i->lock();
        if (currentPytorchModel->getSrcPath() == path) {
            const std::lock_guard<std::mutex> lock(currentPytorchModel->running_mutex);
            res = std::make_shared<PytorchModel>(path, currentPytorchModel);
            break;
        }
    }
    if (!res.get())
        res = std::make_shared<PytorchModel>(path);
    loadedModels.push_back(res);
    return res;
}
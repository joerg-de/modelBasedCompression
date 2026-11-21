#include "PytorchModelCache.h"
#include "PytorchState.h"
#include <exception>
#include <string>

int main(int argc, char* argv[]) {
    try {

        std::string testString;
        if (argc > 2)
            testString = std::string(argv[1]);
        else
            testString = "This is a test";
        auto model = getModelAt("DEv1.pt");

        // model->loadBlacklist("blacklist.txt", true, true);

        auto enc1 = model->generateState(model);
        enc1->initRangeCoding();
        auto res1 = std::get<0>(enc1->decodeToBinaryRangeCoding(testString.c_str(), [](int a) {}, false, 32, 64));
        std::cout << res1 << "\n";

        long long tempSize = res1.size();
        // dont use all 0 or all 1 as that makes the algo perform worse. Use random data
        res1.append("01010101010101010101010100000000000000000000000000000000000000000000000000000000000000000000000000"
                    "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                    "000000000000000000000000000000000000000000000000000");
        auto dec1 = model->generateState(model);
        dec1->initRangeCoding();
        auto res2 = dec1->decodeFromBinaryRangeCoding(res1.c_str(), [](int a) {}, true, 32, 64, tempSize);
        std::cout << std::get<0>(res2) << "\n";

        // speed test
        /* auto encString = std::string(0xF000, '0');
        auto dec2 = model->generateState(model);
        auto [res3, leftover2] = dec2->decodeFromBinary(encString.c_str(), [](int a) {});
        std::cout << res3 << "\n";

        auto enc2 = model->generateState(model);
        auto res4 = enc2->decodeToBinary(res3.c_str(), [](int a) {});
        std::cout << get<0>(res4) << "\n";*/

        // speed test2
        auto encString = std::string();
        for (int i = 0; i < 0x8000; ++i) {
            encString.append("0");
            encString.append("1");
        }
        auto dec2 = model->generateState(model);
        dec2->initRangeCoding();
        auto res3 = std::get<0>(
            dec2->decodeFromBinaryRangeCoding(encString.c_str(), [](int a) {}, true, 32, 64, encString.size() - 100));
        std::cout << res3 << "\n";

        auto enc2 = model->generateState(model);
        enc2->initRangeCoding();
        auto res4 = std::get<0>(enc2->decodeToBinaryRangeCoding(res3.c_str(), [](int a) {}, true, 32, 64));
        std::cout << res4 << "\n";
    } catch (std::exception& e) {
        std::cout << e.what() << "\n";
    }
    return 0;
}
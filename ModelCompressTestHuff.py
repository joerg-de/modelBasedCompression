#moved anaconda3_New\envs\py_3_12_torch2_6\c10.dll
#moved anaconda3_New\envs\py_3_12_torch2_6\torch_cpu.dll
#moved anaconda3_New\envs\py_3_12_torch2_6\fbgemm.dll
#moved anaconda3_New\envs\py_3_12_torch2_6\asmjit.dll

# maybe later as dose not work with hidden being passed
# pass size to torch.jit.script(test_sum, example_inputs=[(3, 4)])

import unittest
import secrets

from TorchModelCompress import pytorchModel


def nothing(fromCpp):
    return 0


def rand_key(p):
    key1 = ""
    for i in range(p):
        temp = str(secrets.randbits(1))
        key1 += temp

    return key1


testString = b"This is a Test"


class TestStringMethods(unittest.TestCase):

    def test_MutiblePytorchModel(self):
        a = pytorchModel("DEv1.pt")
        b = pytorchModel("DEv1.pt")
        c = pytorchModel("Num.pt")

    def test_getCharList(self):
        a = pytorchModel("DEv0.pt")
        self.assertEqual(a.getCharList(), '\n !"#%&\'()*+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz|§«°²»ÄÉÖ×Üßàáâãäçèéíñóôöøúüýāćčěıłōřšū–‘’‚“„†…')

    def test_loadBlacklist1(self):
        a = pytorchModel("DEv1.pt")
        with self.assertRaises(RuntimeError):
            a.loadBlacklist("notHere", True, True)

    def test_loadBlacklist2(self):
        a = pytorchModel("DEv1.pt")
        a.loadBlacklist("blacklist.txt", True, True)
        aR1 = a.getInitState()
        temp = aR1.decodeToBinary(testString, 1, nothing, True, 1, 0, 0.5)

    def test_loadBlacklist3(self):
        a = pytorchModel("DEv1.pt")
        a.loadBlacklistLine("\n", True)
        aR1 = a.getInitState()
        temp = aR1.decodeToBinary(testString, 1, nothing, True, 1, 0, 0.5)

    def test_decAndBack(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        aR2 = a.getInitState()
        temp = aR2.decodeToBinary(testString, 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(temp[0].translate(str.maketrans('', '', '01')), "")
        self.assertNotEqual(temp[0], "")
        output = aR1.decodeFromBinary(temp[0], 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(output[0], testString)

    def test_clone_decAndBack(self):
        a = pytorchModel("DEv1.pt")
        b = a.clone()
        aR1 = a.getInitState()
        aR2 = b.getInitState()
        temp = aR2.decodeToBinary(testString, 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(temp[0].translate(str.maketrans('', '', '01')), "")
        self.assertNotEqual(temp[0], "")
        output = aR1.decodeFromBinary(temp[0], 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(output[0], testString)

    def test_decAndBackReset1(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        aR1.resetDecState()
        aR2 = a.getInitState()
        temp = aR2.decodeToBinary(testString, 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(temp[0].translate(str.maketrans('', '', '01')), "")
        self.assertNotEqual(temp[0], "")
        output = aR1.decodeFromBinary(temp[0], 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(output[0], testString)

    def test_decAndBackReset2(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        temp = aR1.decodeToBinary(testString, 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(temp[0].translate(str.maketrans('', '', '01')), "")
        self.assertNotEqual(temp[0], "")
        aR1.resetDecState()
        output = aR1.decodeFromBinary(temp[0], 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(output[0], testString)

    # this test Fails if the Module is not statefull --> stateless
    def test_decAndBackStateFull(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        temp = aR1.decodeToBinary(testString, 1, nothing, False, 1, 0, 0.5)
        self.assertEqual(temp[0].translate(str.maketrans('', '', '01')), "")
        self.assertNotEqual(temp[0], "")
        output = aR1.decodeFromBinary(temp[0], 1, nothing, False, 1, 0, 0.5)
        self.assertNotEqual(output[0], testString)

    # this test Fails if the Module is not statefull --> stateless
    def test_runWithoutDecoding(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        aR1.runWithoutDecoding(testString, 1, nothing, True)
        aR2 = aR1.cloneDecState()
        temp = aR1.decodeToBinary(testString, 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(temp[0].translate(str.maketrans('', '', '01')), "")
        self.assertNotEqual(temp[0], "")
        output = aR2.decodeFromBinary(temp[0], 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(output[0], testString)

    def test_runWithoutDecoding2(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        aR1.runWithoutDecoding(testString, 1, nothing, False)
        aR2 = aR1.cloneDecState()
        temp = aR1.decodeToBinary(testString, 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(temp[0].translate(str.maketrans('', '', '01')), "")
        self.assertNotEqual(temp[0], "")
        output = aR2.decodeFromBinary(temp[0], 1, nothing, True, 1, 0, 0.5)
        self.assertEqual(output[0], testString)

    def test_Clone(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        aR2 = aR1.cloneDecState()
        temp = aR1.decodeToBinary(testString, 1, nothing, False, 1, 0, 0.5)
        aR1.resetDecState()
        sink = aR1.decodeFromBinary(temp[0], 1, nothing, False, 1, 0, 0.5)
        aR3 = aR1.cloneDecState()
        self.assertEqual(temp[0].translate(str.maketrans('', '', '01')), "")
        self.assertNotEqual(temp[0], "")
        output1 = aR2.decodeFromBinary(temp[0], 1, nothing, False, 1, 0, 0.5)
        self.assertEqual(output1[0], testString[:len(output1[0])-output1[1]])
        output2 = aR2.decodeFromBinary(temp[0], 1, nothing, False, 1, 0, 0.5)
        self.assertNotEqual(output2[0], testString)
        output3 = aR3.decodeFromBinary(temp[0], 1, nothing, False, 1, 0, 0.5)
        self.assertNotEqual(output3[0], testString)
        self.assertEqual(output3[0], output2[0])

    def test_inLasterCreatedNodes(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        temp = aR1.decodeToBinary(testString, 1, nothing, False, 1, 0, 0.5)
        aR1.inLasterCreatedNodes("test")

    def test_CheckIfFullConsumeInput(self):
        a = pytorchModel("DEv1.pt")
        aR1 = a.getInitState()
        for i in range(1000):
            size = secrets.randbelow(100)
            toDec = rand_key(size)
            aR2 = aR1.cloneDecState()
            (ret, leftover) = aR2.decodeFromBinary(toDec, 1, nothing, True, 1, 32, 0.0078125)
            if (leftover == 0):
                return
        self.fail("did not find any decode ")


if __name__ == '__main__':
    unittest.main()

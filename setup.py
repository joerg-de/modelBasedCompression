from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import os

if os.name == 'nt':  # MSVC (Windows)
    extra_compile_args = ['/O2', '/std:c++20', '/fp:fast', '/Gw', '/GL']
    extra_link_args = ['/LTCG']
else:  # GCC/Clang (Linux/macOS)
    extra_compile_args = ['-O3', '-std=c++20', '-ffast-math', '-flto']
    extra_link_args = ['-flto=auto']

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    setup(name='TorchModelCompress',
          version='1.4.0',
          ext_modules=[
                       CppExtension(
                                    'TorchModelCompress',
                                    [
                                     'TreeCore//Main//PytorchModel.cpp',
                                     'TreeCore//Main//PytorchModelCache.cpp',
                                     'TreeCore//Main//PytorchState.cpp',
                                     'TreeCore//Main//BigUInt.cpp',
                                     'TreeCore//PyBridge//PythonBridgeMain.cpp',
                                     'TreeCore//PyBridge//PythonBridgePytorchModel.cpp',
                                     'TreeCore//PyBridge//PythonBridgePytorchState.cpp',
                                     'TreeCore//PyBridge/PythonCallBackThrottler.cpp',
                                    ],
                                    include_dirs=[this_dir + '//TreeCore//Main', this_dir + '//TreeCore//PyBridge'],
                                    extra_compile_args=extra_compile_args,
                                    extra_link_args=extra_link_args),
                       ],
          cmdclass={'build_ext': BuildExtension},
          )

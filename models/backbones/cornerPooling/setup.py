from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# run this setup file with 
# >> python setup.py install --user

setup(
    name = "cornerPooling",
    ext_modules = [
        CppExtension("topPool", ["source/topPool.cpp"]),
        CppExtension("bottomPool", ["source/bottomPool.cpp"]),
        CppExtension("leftPool", ["source/leftPool.cpp"]),
        CppExtension("rightPool", ["source/rightPool.cpp"])
    ],
    cmdclass = {
        "build_ext": BuildExtension
    }
)

from distutils.core import setup, Extension
MOD = "archess"
module = Extension(MOD, sources = ["archess.cpp"], extra_compile_args=['-std=c++11'])
setup(name=MOD, ext_modules=[module])
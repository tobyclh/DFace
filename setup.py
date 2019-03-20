from setuptools import setup, find_packages, distutils

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


CUDA_FLAGS = []

ext_modules=[
    CUDAExtension('dface.core.nms', [
        # 'dface/core/nms/cpu/nms_cpu.cpp',
        'dface/core/nms/cuda/nms_cuda.cu'
        ])
    ]


setup(name='DFace',
      version='0.1',
      description='',
      url='http://github.com/tobyclh/DFace',
      author='tobyclh',
      author_email='tobyclh@embodyme.com',
      license='Copyright',
      packages=find_packages(),
      zip_safe=False,
      ext_modules=ext_modules,
      cmdclass = {'build_ext': BuildExtension},
      include_package_data=False)

from setuptools import setup, find_packages

setup(
  name = 'mirasol-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.14',
  license='MIT',
  description = 'Mirasol - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/mirasol-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'multimodality'
  ],
  install_requires=[
    'beartype',
    'einops>=0.7.0',
    'x-transformers>=1.25.10',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

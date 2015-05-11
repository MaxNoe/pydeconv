from setuptools import setup

setup(
    name='deconv',
    version='0.1',
    description='Unfolding (deconvolution) algorithms for physics',
    url='http://github.com/maxnoe/pydeconv',
    author='Maximilian Noethe, Kai Bruegge',
    author_email='maximilian.noethe@tu-dortmund.de',
    license='MIT',
    packages=['deconv'],
    install_requires=[
        'numpy',
        'scipy',
    ],
)

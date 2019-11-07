from setuptools import setup, find_packages

setup(
    name='adv_interp_train',
    version='0.0.0',
    install_requires=[
        'torch',
        'torchvision',
        'scipy',
        'tqdm',
    ],
    packages=find_packages(),
)

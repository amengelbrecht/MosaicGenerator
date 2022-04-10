from setuptools import setup

setup(
    name='MosaicGenerator',
    url='https://github.com/amengelbrecht/MosaicGenerator',
    author='Adriaan Engelbrecht',
    author_email='adriaan@vulcansoftworks.com',
    packages=['MosaicGenerator'],
    install_requires=['cv2', 'tqdm', 'pandas', 'os', 'numpy', 'copy'],
    version='0.1',
    license='MIT',
    description='Generates target image from frames of source video(s).'
)
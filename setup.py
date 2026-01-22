import os
import subprocess
from setuptools import setup, find_packages,Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import sys,shutil
import pybind11

def get_version():
    version_file = os.path.join("polsartools", "__version__.py")
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

class BuildExt(build_ext):
    """Custom build_ext to ensure .pyd/.so files are placed inside the package directory."""
    
    def build_extension(self, ext):
        # Get the full path where the extension is originally built
        ext_path = self.get_ext_fullpath(ext.name)
        
        # Correct the destination directory to be inside the package
        ext_name = os.path.basename(ext_path)
        dest_path = os.path.join(os.path.dirname(__file__), "polsartools", ext_name)

        # Call the original method to build the extension
        super().build_extension(ext)

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the built extension to the package directory
        shutil.copy2(ext_path, dest_path)

ext_modules = [
    Extension(
        'polsartools.rflee',
        ['polsartools/cpp/src/rflee.cpp'],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language='c++',
        extra_compile_args=['/std:c++17' if os.name == 'nt' else '-std=c++17'],
    ),
    Extension(
        'polsartools.cprvicpp',
        ['polsartools/cpp/src/cprvicpp.cpp'],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language='c++',
        extra_compile_args=['/std:c++17' if os.name == 'nt' else '-std=c++17'],
    ),
    Extension(
        'polsartools.yam4cpp',
        ['polsartools/cpp/src/yam4cpp.cpp'],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language='c++',
        extra_compile_args=['/std:c++17' if os.name == 'nt' else '-std=c++17'],
    ),
]

# class CMakeBuild(build_ext):
#     def run(self):
#         # Ensure the build directory exists
#         if not os.path.exists(self.build_temp):
#             os.makedirs(self.build_temp)
        
#         # Run CMake to configure the build
#         subprocess.check_call(['cmake', '..'], cwd=self.build_temp)
        
#         # Run the build command
#         subprocess.check_call(['cmake', '--build', '.'], cwd=self.build_temp)
        
#         super().run()

# class CustomInstall(install):
#     def run(self):
#         # Call the install command
#         install.run(self)
#         # Optionally run additional post-installation commands here
with open("README.md", encoding="utf-8") as f: 
    long_description = f.read()
    
setup(
    name='polsartools',
    version=get_version(),
    description='A python package for processing Polarimetric Synthetic Aperture Radar (PolSAR) data.',
    # long_description=open('README.md').read(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Narayanarao Bhogapurapu',
    author_email='bnarayanarao@nitw.ac.in',
    url='https://github.com/polsartools/polsartools',
    packages=find_packages(),
    package_dir={'polsartools': 'polsartools'},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'gdal',
        'scipy',
        'click',
        'tqdm',
        'matplotlib',
        'pybind11',
        'tables',
        'netcdf4',
        'scikit-image',
        
    ],
    extras_require={
        'dev': ['pytest', 'sphinx','pydata-sphinx-theme'],
    },
    entry_points={
        'console_scripts': [
            'polsartools=polsartools.cli:cli',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: GPL-3.0 License',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)
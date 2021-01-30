import os
import subprocess
from setuptools import setup, Extension, find_packages


def git_hash():
    git_path = './' if os.path.exists('.git') else ('../' if os.path.exists('../.git') else None)
    if git_path:
        hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return hash[:7] if (hash is not None) else None
    else:
        return None


def get_version():
    from version import __version__
    hash = git_hash()
    version_str = __version__ + '+' + hash.strip().decode('ascii') if (hash is not None) else __version__
    return version_str


if __name__ == '__main__':
    version_str = get_version()

    long_description = ''
    with open('README.md') as readme:
        long_description = readme.read()

    setup(
        name='jacinto_ai_benchmark',
        version=get_version(),
        description='Accuracy Benchmarking For Deep Learning',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-benchmark/browse',
        author='TIDL & Jacinto AI DevKit Team',
        author_email='jacinto-ai-devkit@list.ti.com',
        classifiers=[
            'Development Status :: 4 - Beta'
            'Programming Language :: Python :: 3.7'
        ],
        keywords = 'artifical intelligence, deep learning, image classification, object detection, semantic segmentation, quantization',
        python_requires='>=3.6',
        packages=find_packages(),
        include_package_data=True,
        install_requires=[],
        project_urls={
            'Source': 'https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-benchmark/browse',
            'Bug Reports': 'https://e2e.ti.com/support/processors/f/791/tags/jacinto_2D00_ai_2D00_devkit',
        },
    )


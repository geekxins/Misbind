from setuptools import setup, find_packages

setup(
    name="misbind",
    version="1.0.0",
    description="Adversarial Perturbation for Protecting Images from Unauthorized Personalization of Diffusion Models",
    author="MisBind Authors",
    url="https://github.com/your-username/MisBind",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "diffusers>=0.25.0",
        "accelerate>=0.25.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "misbind=misbind.core:main",
            "misbind-batch=misbind.batch:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

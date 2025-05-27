from setuptools import setup, find_packages

setup(
    name="model",
    version="0.1",
    author="Yago Moreno",
    author_email="ymoreno.ieu2022@student.ie.edu",
    description="A text classification model for emotion detection in tweets",
    url="https://github.com/ymoreno2022/Natural-Language-Competition/",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "emotion_model": ["roberta_emotion/*"]
    },
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "requests>=2.25.0",
        "click>=8.0.0",
        "tqdm>=4.62.0",
        "safetensors>=0.3.0"
    ],
    entry_points={
        'console_scripts': [
            'inference=model.cli:main',
        ],
    },
)

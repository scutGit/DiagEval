from setuptools import find_packages, setup

required = [
    # Computer Vision
    "opencv-python",
    "matplotlib",
    # UI Automation
    "uiautomator2",
    "pynput",
    "pyautogui",
    "pywinauto",
    # Utils
    "loguru",
    "tenacity",
    "aiohttp",
    "pandas",
    "pyclipper",
    "shapely",
    "tabulate>=0.9.0",
]

extras_require = {
    "ultra": [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "tensorflow==2.17.1",
        "tf_slim",
        "transformers",
        "modelscope[framework]==1.22.3",
        "ultralytics",
    ]
}

setup(
    name="diageval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    extras_require=extras_require,
    description="DiagEval: Diagnostic Retry Framework for GUI Evaluation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

from setuptools import setup, find_packages

setup(
    name='reservoir-history-matching',
    version='1.0.0',
    description='ML-based reservoir production history matching and forecasting',
    author='Ing. Kelvin Cabrera',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask>=2.3.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
    ],
    extras_require={
        'dev': ['pytest>=7.0.0', 'requests>=2.31.0'],
    },
    entry_points={
        'console_scripts': [
            'reservoir-server=app:main',
        ],
    },
)

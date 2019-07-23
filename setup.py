import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name='rlpack',
	version='0.1',
	author='Jacob Pettit',
	author_email='jfpettit@gmail.com',
    short_description='Simple implementations of reinforcement learning algorithms.',
    long_description=long_description,
	url='https://github.com/jfpettit/rlpack',
	install_requires=['numpy', 'torch', 'gym', 'scipy']
	,
)

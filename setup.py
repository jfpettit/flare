import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name='reinforcement_learning-jfpettit',
	version='0.1dev',
	author='Jacob Pettit',
	author_email='jfpettit@gmail.com',
	description='Implementations of reinforcement learning (RL) algorithms and some custom environments.',
	long_description=long_description,
	long_description_content_type = 'text/markdown',
	url='https://github.com/jfpettit/reinforcement_learning',
	packages=setuptools.find_packages(),
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License',
		'Operating System:: OS Independent',
	],
)

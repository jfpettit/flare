import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name='flare',
	version='0.1.5',
	author='Jacob Pettit',
	author_email='jfpettit@gmail.com',
    short_description='Simple implementations of reinforcement learning algorithms.',
    long_description=long_description,
	url='https://github.com/jfpettit/flare',
	install_requires=['numpy', 'torch', 'gym[box2d]', 'scipy', 'roboschool', 'pybullet', 'termcolor', 'joblib', 'tensorboard', 'mpi4py']
	,
)

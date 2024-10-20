from setuptools import setup, Extension

#
setup(name='flightgym',
      version='0.0.1',
      author="Junjie Lu / Yunlong Song",
      author_email='lqzx1998@tju.edu.cn / song@ifi.uzh.ch',
      description="Flightmare: A Quadrotor Simulator",
      long_description='This project is modified based on Flightmare by Yunlong Song, Thanks for his excellent work!',
      packages=[''],
      package_dir={'': './'},
      package_data={'': ['flightgym.cpython-36m-x86_64-linux-gnu.so']},
      zip_fase=True,
      url=None,
)

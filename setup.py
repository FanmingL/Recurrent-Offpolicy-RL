from setuptools import setup, find_packages

setup(
    name='offpolicy_rnn',
    version='0.0.1',
    author="Fan-Ming Luo",
    author_email="luofm@lamda.nju.edu.cn",
    description="An implementation of RNN policy and value function in off-policy RL",
    packages=find_packages(),
    install_requires=[
        "gym", # gym==0.15.4 for rmdp and rl_generalization tasks
        # "smart-logger @ git+https://github.com/FanmingL/SmartLogger.git",
        "torch",
        "tqdm",
        "numpy",
        "pybullet==3.2.5",
        "pycolab==1.2.0",
        'roboschool',
        # "box2d-py",
        # "Box2D"
    ],
)

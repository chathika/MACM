from setuptools  import setup

setup(
    name='MACM',
    version='0.2.0',
    author='Chathika Gunaratne',
    author_email='chathikagunaratne@gmail.com',
    packages=['MACM'],
    include_package_data=True,
    url='https://github.com/chathika/macm',
    license='GPL',
    description='Multi-Action Cascade Model of conversation.',
    long_description="""Multi-Action Cascade Model of conversation. 
        Cite as Gunaratne, C., Baral, N., Rand, W., Garibay, I., Jayalath, C., 
        & Senevirathna, C. (2020). The effects of information overload on 
        online conversation dynamics. Computational and Mathematical 
        Organization Theory, 26(2), 255-276.""",
    long_description_content_type='text/markdown',
    project_urls={
    'Source': 'https://github.com/chathika/macm'
    },
    install_requires=[
        'numba >= 0.55.1',
        'pandas >= 1.4.1',
        'numpy >= 1.21.5',
		'argparse >= 1.4.0',
		'tqdm >= 4.63.0',
    ]
)
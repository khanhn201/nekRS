import datetime
today_str = datetime.date.today().strftime("%Y-%m-%d")

site_configuration = {
    'systems': [
        {
            'name': 'local',
            'hostnames': ['.*'],
            'prefix': f'/home/nekoconn/code/seal/reframe-test/{today_str}',
            'partitions': [
                {
                    'name': 'default',
                    'descr': 'Default partition',
                    'scheduler': 'local',
                    'launcher': 'mpirun',
                    'environs': ['local'],
                }
            ]
        }
    ],
    'storage': [{'enable': True}],
    'environments': [
        {
            'name': 'local',
            'cc': 'mpicc',
            'cxx': 'mpic++',
            'ftn': 'mpif77',
            'env_vars': [
                ['CC', 'mpicc'],
                ['CXX', 'mpic++'],
                ['FC', 'mpif77'],
                ['OMPI_CXX', 'g++-13'],
                ['OMPI_CC',  'gcc-13'],
                ['OMPI_FC',  'gfortran-13'],
                ['CUDAHOSTCXX', 'g++-13'],
            ],
            'extras': {
                'source_dir': '/home/nekoconn/code/seal/nekRS-khanhn',
                'prebuilt_dir': '/home/nekoconn/code/seal/reframe-test/install',
                'cpu_bind': '',
                'ranks_per_node': 1,
                'backend': 'serial',
            }
        }
    ]
}

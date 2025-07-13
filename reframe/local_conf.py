site_configuration = {
    'systems': [
        {
            'name': 'local',
            'hostnames': ['AZKi'],
            'partitions': [
                {
                    'name': 'default',
                    'descr': 'Default partition',
                    'scheduler': 'local',
                    'launcher': 'mpirun',
                    'environs': ['local']
                }
            ]
        }
    ],
    'environments': [
        {
            'name': 'local',
            'modules': ['cmake'],
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
                ['NEKRS_BACKEND', 'serial'],
                ['NEKRS_RANKS_PER_NODE', '1'],
                ['NEKRS_CPU_BIND', '']
            ]
        }
    ]
}

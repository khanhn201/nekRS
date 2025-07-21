site_configuration = {
    'systems': [
        {
            'name': 'aurora',
            'hostnames': ['.*'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'default',
                    'descr': 'default',
                    'scheduler': 'pbs',
                    'launcher': 'mpirun',
                    'environs': ['aurora'],
                    'resources': [
                        {
                            'name': 'account',
                            'options': ['-A {account}']
                        },
                        {
                            'name': 'walltime',
                            'options': ['-l walltime={walltime}']
                        },
                        {
                            'name': 'filesystem',
                            'options': ['-l filesystems={filesystem}']
                        },
                        {
                            'name': 'queue',
                            'options': ['-q {queue}']
                        },
                        {
                            'name': 'nodes',
                            'options': ['-l select={nodes}']
                        }
                    ]
                }
            ]
        }
    ],
    'environments': [
        {
            'name': 'aurora',
            'modules': ['cmake'],
            'cc': 'mpicc',
            'cxx': 'mpic++',
            'ftn': 'mpif77',
            'nvcc': '',
            'env_vars': [
                ['CC', 'mpicc'],
                ['CXX', 'mpic++'],
                ['FC', 'mpif77'],
            ],
            'extras': {
                'prebuilt_dir': '/lus/flare/projects/pe-summer-2025/khanhn/reframe-test/install',
                'cpu_bind': 'list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99',
                'ranks_per_node': 12,
                'backend': 'dpcpp',
            }
        }
    ]
}

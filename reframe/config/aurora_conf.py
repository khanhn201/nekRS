import datetime
today_str = datetime.date.today().strftime("%Y-%m-%d")

site_configuration = {
    'systems': [
        {
            'name': 'aurora',
            'hostnames': ['.*'],
            'prefix': f'/lus/flare/projects/pe-summer-2025/khanhn/reframe-test/{today_str}',
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
                            'name': 'filesystem',
                            'options': ['-l filesystems={filesystem}']
                        },
                        {
                            'name': 'queue',
                            'options': ['-q {queue}']
                        }
                    ]
                }
            ]
        }
    ],
    'storage': [
        {
            'enable': True,
            'sqlite_db_file': '/lus/flare/projects/pe-summer-2025/khanhn/.reframe/reports/results.db'
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
                ['NEKRS_MPI_THREAD_MULTIPLE', '1'],
                ['NEKRS_GPU_MPI', '0'],
                ['OCCA_DPCPP_COMPILER_FLAGS', '\"-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma\"']
            ],
            'extras': {
                'source_dir': '/lus/flare/projects/pe-summer-2025/khanhn/nekRS_khanhn',
                'prebuilt_dir': '/lus/flare/projects/pe-summer-2025/khanhn/reframe-test/install',
                'cpu_bind': 'list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99',
                'ranks_per_node': 12,
                'backend': 'dpcpp',
                'project': 'pe-summer-2025',
                'lhelper_script': """#!/bin/bash
gpu_id=$(( (PALS_LOCAL_RANKID / 2) % 6 ))
tile_id=$(( PALS_LOCAL_RANKID % 2 ))
export ZE_AFFINITY_MASK=$gpu_id.$tile_id
"$@"
"""
            }
        }
    ]
}

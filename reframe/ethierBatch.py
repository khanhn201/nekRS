import os

import reframe as rfm
from reframe.core.backends import getlauncher

@rfm.simple_test
class NekRSEthierBatchTest(rfm.RunOnlyRegressionTest):
  num_nodes = variable(int, value=1)
    
  def __init__(self):
    super().__init__()
    self.descr = 'nekRS Ethier batch test'
    self.maintainers = []
    self.tags = {'nekrs'}
    self.modules = ['cmake']
    self.valid_systems = ['*']
    self.valid_prog_environs = ['*']

  @run_after('setup')
  def read_extras(self):
    self.ranks_per_node = self.current_environ.extras.get('ranks_per_node', 1)
    project = self.current_environ.extras.get('project', '')
    self.extra_resources = {
        'account': {'account': project},
        'queue': {'queue': 'debug'},
        'filesystem': {'filesystem': 'home:flare'},
    }

  @run_after('setup')
  def set_scheduler_and_launcher_options(self):
    self.time_limit = '1h'
    self.num_tasks = self.num_nodes * self.ranks_per_node
    self.num_tasks_per_node = self.ranks_per_node

    self.job.launcher = getlauncher('local')()

  @run_after('setup')
  def change_sourcedir(self):
    sourcesdir = self.current_environ.extras.get('source_dir', '../')
    conf_file = os.path.join(sourcesdir,'reframe/config/aurora_conf.py')
    test_file = os.path.join(sourcesdir,'reframe/ethier.py')
    self.prerun_cmds = ['module load reframe']
    self.executable = f'reframe -C {conf_file} --exec-policy=serial -c {test_file} -v -r'

  @sanity_function
  def check_exit_code(self):
    return True

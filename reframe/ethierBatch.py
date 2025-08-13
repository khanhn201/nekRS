import os

import reframe as rfm
from reframe.core.backends import getlauncher
import reframe.utility.sanity as sn

@rfm.simple_test
class NekRSEthierBatchTest(rfm.RunOnlyRegressionTest):
  num_nodes = variable(int, value=1)
  time_limit = '1h'
    
  def __init__(self):
    super().__init__()
    self.descr = 'nekRS test'
    self.maintainers = []
    self.tags = {'nekrs'}
    self.valid_systems = ['*']
    self.valid_prog_environs = ['*']
    self.modules = ['cmake']
    self.device_id = 0

    self.num_tasks = self.num_nodes * 12
    self.num_tasks_per_node = 12


  @run_after('setup')
  def set_launcher(self):
    self.project = self.current_environ.extras.get('project', '')
    self.extra_resources = {
        'account': {'account': self.project},
        'queue': {'queue': 'debug'},
        'filesystem': {'filesystem': 'home:flare'},
    }
    self.job.launcher = getlauncher('local')()
    sourcesdir = self.current_environ.extras.get('source_dir', '../')
    conf_file = os.path.join(sourcesdir,'reframe/config/aurora_conf.py')
    test_file = os.path.join(sourcesdir,'reframe/ethier.py')
    self.prerun_cmds = ['module load reframe']
    self.executable = f'reframe -C {conf_file} --exec-policy=serial -c {test_file} -v -r'

  @sanity_function
  def check_exit_code(self):
    return True

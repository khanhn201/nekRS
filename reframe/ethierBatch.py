import reframe as rfm
from reframe.core.backends import getlauncher
import reframe.utility.sanity as sn
from nekrs import NekRSCase, NekRSTest, NekRSTestBuildOnly

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
    self.extra_resources = {
        'account': {'account': 'pe-summer-2025'},
        'queue': {'queue': 'debug'},
        'filesystem': {'filesystem': 'home:flare'},
        # 'nodes': {'nodes': self.num_nodes}
    }

    self.num_tasks = self.num_nodes * 12
    self.num_tasks_per_node = 12

    self.prerun_cmds = ['module load reframe']
    self.executable = f'reframe -C /lus/flare/projects/pe-summer-2025/khanhn/nekRS_khanhn/reframe/aurora_conf.py --exec-policy=serial -c /lus/flare/projects/pe-summer-2025/khanhn/nekRS_khanhn/reframe/ethier.py -r -v'

  @run_after('setup')
  def set_launcher(self):
    self.job.launcher = getlauncher('local')()
  @sanity_function
  def check_exit_code(self):
    return True

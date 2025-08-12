import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from nekrs import NekRSCase, NekRSTest, NekRSTestBuildOnly

class NekRSEthierCase(NekRSCase):
  case_name = 'ethier'
  case_root = '../examples'
  num_nodes = 1
  ci_mode = 0

  def __init__(self, num_nodes, ci_mode):
    self.num_nodes = num_nodes
    self.ci_mode = ci_mode
    super().__init__(name=f'{self.case_name}', directory=f'{self.case_root}/{self.case_name}')

class NekRSEthierTestBuildOnly(NekRSTestBuildOnly):
  num_nodes = variable(int, value=1)
  ci_mode = variable(int, value=1)
  local = True
  def __init__(self):
    self.local=True
    super().__init__(nekrs_case=NekRSEthierCase(self.num_nodes, self.ci_mode))

  @run_after('setup')
  def set_launcher(self):
    self.job.launcher = getlauncher('mpirun')()

@rfm.simple_test
class NekRSEthierTest(NekRSTest):
  num_nodes = variable(int, value=1)
  ci_mode = parameter([1, 2, 4, 7, 8, 9, 10, 11, 12, 14, 15, 19, 23, 29, 30])
  case_build = fixture(NekRSEthierTestBuildOnly, scope='environment')
  local = True
    
  def __init__(self):
    nekrs_case=NekRSEthierCase(self.num_nodes, self.ci_mode)
    super().__init__(nekrs_case)

  @run_after('setup')
  def set_launcher(self):
    self.job.launcher = getlauncher('mpirun')()

  @run_after('setup')
  def change_sourcedir(self):
    self.sourcesdir = self.case_build.stagedir

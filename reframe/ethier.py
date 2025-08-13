import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from nekrs import NekRSTest, NekRSTestBuildOnly

class NekRSEthierTestBuildOnly(NekRSTestBuildOnly):
  num_nodes = variable(int, value=1)
  ci_mode = variable(int, value=1)
  local = True
  def __init__(self):
    self.local=True
    super().__init__('ethier', self.num_nodes, self.ci_mode)

  @run_after('setup')
  def set_launcher(self):
    self.job.launcher = getlauncher('mpirun')()
    sourcesdir = self.current_environ.extras.get('source_dir', '../')
    directory = os.path.join(sourcesdir,'examples/ethier')
    self.set_directory(directory)

@rfm.simple_test
class NekRSEthierTest(NekRSTest):
  num_nodes = variable(int, value=1)
  ci_mode = parameter([1, 2, 4, 7, 8, 9, 10, 11, 12, 14, 15, 19, 23, 29, 30])
  case_build = fixture(NekRSEthierTestBuildOnly, scope='environment')
  local = True
    
  def __init__(self):
    super().__init__('ethier', self.num_nodes, self.ci_mode)

  @run_after('setup')
  def set_launcher(self):
    self.job.launcher = getlauncher('mpirun')()

  @run_after('setup')
  def change_sourcedir(self):
    self.sourcesdir = self.case_build.stagedir

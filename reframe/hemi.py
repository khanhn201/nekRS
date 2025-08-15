import os

import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSTest

@rfm.simple_test
class NekRSHemiTest(NekRSTest):
  num_nodes = variable(int, value=8)

  def __init__(self):
    super().__init__('hemi', self.num_nodes, 0)

  @run_after('setup')
  def change_sourcedir(self):
    sourcesdir = self.current_environ.extras.get('source_dir', '../')
    directory = os.path.join(sourcesdir,'examples/hemi')
    self.set_directory(directory)

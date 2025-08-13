import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSCase, NekRSTest

class NekRSHemiCase(NekRSCase):
  case_name = 'hemi'
  case_root = '../examples'
  num_nodes = 1

  def __init__(self, num_nodes):
    self.num_nodes = num_nodes
    super().__init__(name=f'{self.case_name}', directory=f'{self.case_root}/{self.case_name}')

@rfm.simple_test
class NekRSHemiTest(NekRSTest):
  num_nodes = variable(int, value=8)
  maximum_walltime = '01:00:00'
  # time_steps = 8000

  def __init__(self):
    self.sourcesdir = self.current_environ.extras.get('source_dir', '../')
    directory = os.path.join(self.install_path,'/examples/hemi')
    super().__init__(nekrs_case=NekRSHemiCase(self.num_nodes))

  # @run_after('setup')
  # def set_run_parameters(self):
  #   self.set_walltime(self.maximum_walltime)


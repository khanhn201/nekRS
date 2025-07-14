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
  num_nodes = variable(int, value=4)
  maximum_walltime = '01:00:00'
  # time_steps = 8000

  def __init__(self):
    super().__init__(nekrs_case=NekRSHemiCase(self.num_nodes))

  # @run_after('setup')
  # def set_run_parameters(self):
  #   self.set_walltime(self.maximum_walltime)

  @performance_function('fom')
  def calculate_fom(self):
    solve_times = sn.extractall(r'solve\s+(\S+)s', self.stdout, 1, float)
    fom = 1.0 / solve_times[-1]
    return fom

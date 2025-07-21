import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSCase, NekRSTest

class NekRSEthierCase(NekRSCase):
  case_name = 'ethier'
  case_root = '../examples'
  num_nodes = 1
  ci_mode = 0

  def __init__(self, num_nodes, ci_mode):
    self.num_nodes = num_nodes
    self.ci_mode = ci_mode
    super().__init__(name=f'{self.case_name}', directory=f'{self.case_root}/{self.case_name}')

@rfm.simple_test
class NekRSEthierTest(NekRSTest):
  num_nodes = variable(int, value=2)
  ci_mode = parameter(list(range(1, 31)))
  maximum_walltime = '01:00:00'
  # time_steps = 8000

  def __init__(self):
    super().__init__(nekrs_case=NekRSEthierCase(self.num_nodes, self.ci_mode))

  # @run_after('setup')
  # def set_run_parameters(self):
  #   self.set_walltime(self.maximum_walltime)

  @performance_function('fom')
  def calculate_fom(self):
    solve_times = sn.extractall(r'solve\s+(\S+)s', self.stdout, 1, float)
    fom = 1.0 / solve_times[-1]
    return fom

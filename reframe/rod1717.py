import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSCase, NekRSTest

class NekRSRodBundleCase(NekRSCase):

  case_name = 'rod1717'
  case_root = '/lus/flare/projects/Aurora_AT/nekRS/cases/rod1717'

  def __init__(self, layers):
    super().__init__(name=f'{self.case_name}_{layers}', directory=f'{self.case_root}/{self.case_name}_{layers}')

@rfm.simple_test
class NekRSRodBundleTest(NekRSTest):

  layers = variable(int, value=170)
  maximum_walltime = '02:30:00'
  # time_steps = 8000

  valid_layers = [170,680,1780,3550,7100,14200]
  case_constraints = {
    170   : {'min_nodes':8  , 'max_nodes':128 },
    680   : {'min_nodes':32 , 'max_nodes':512 },
    1780  : {'min_nodes':64 , 'max_nodes':1024},
    3550  : {'min_nodes':128, 'max_nodes':2048},
    7100  : {'min_nodes':256, 'max_nodes':4096},
    14200 : {'min_nodes':512, 'max_nodes':8192}
  }

  def __init__(self):
    super().__init__(nekrs_case=NekRSRodBundleCase(self.layers))

  @run_after('init')
  def validate_layers(self):
    if self.layers not in self.valid_layers:
      self.skip(f'rod1717_{self.layers} is not a valid case. "layers" must be one of {self.valid_layers}.')

  # Filter out tests which cannot run on the current node count
  @run_before('setup')
  def check_constraints(self):
    min_nodes = self.case_constraints[self.layers]['min_nodes']
    max_nodes = self.case_constraints[self.layers]['max_nodes']
    self.skip_if(self.num_nodes < min_nodes, f'This test requires at least {min_nodes} nodes.')
    self.skip_if(self.num_nodes > max_nodes, f'This should run on at most {max_nodes} nodes.')

  @run_after('setup')
  def set_run_parameters(self):
    self.set_walltime(self.maximum_walltime)

  @performance_function('fom')
  def calculate_fom(self):
    solve_times = sn.extractall(r'solve\s+(\S+)s', self.stdout, 1, float)
    fom = 1.0 / solve_times[-1]
    return fom

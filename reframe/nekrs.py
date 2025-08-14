import os
import os.path

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext


class NekRSBuild(rfm.CompileOnlyRegressionTest):
  use_prebuilt = variable(bool, value=True)

  def __init__(self):
    super().__init__()
    self.descr = 'nekRS build'
    self.maintainers = []
    self.tags = {'nekrs'}
    self.modules = ['cmake']
    self.valid_systems = ['*']

  # Need stagedir, so must call after setup phase
  # https://reframe-hpc.readthedocs.io/en/stable/regression_test_api.html#reframe.core.pipeline.RegressionTest.stagedir
  @run_before('compile')
  def configure_build(self):
    if self.use_prebuilt:
      prebuilt_dir = self.current_environ.extras.get('prebuilt_dir', '')
      self.build_system = 'CustomBuild'
      self.build_system.commands = ['/bin/true']
      self.prebuild_cmds = [f'cp -r {prebuilt_dir} ./']
      
      self.install_path = os.path.join(f'{self.stagedir}','install')
      self.binary_path = os.path.join(self.install_path,'bin')
    else:
      self.sourcesdir = self.current_environ.extras.get('source_dir', '../')
      self.build_system = 'CMake'
            
      self.build_system.flags_from_environ = True
      self.build_system.builddir = 'build'
      self.build_system.max_concurrency = 8
      self.build_system.make_opts = ['install']
      self.install_path = os.path.join(f'{self.stagedir}','install')
      self.binary_path = os.path.join(self.install_path,'bin')
      self.build_system.config_opts = [
        f'-DCMAKE_INSTALL_PREFIX={self.install_path}',
      ]
            
  @sanity_function
  def validate_build(self):
    nekrs_binary = os.path.join(self.binary_path,'nekrs')
    return sn.assert_true(os.path.isfile(nekrs_binary), f'nekRS binary could not be found in path {nekrs_binary}') 


class NekRSTest(rfm.RunOnlyRegressionTest):
  nekrs_build = fixture(NekRSBuild, scope='environment')

  def __init__(self, case_name, num_nodes, ci_mode):
    super().__init__()
    self.descr = 'nekRS test'
    self.maintainers = []
    self.tags = {'nekrs'}
    self.valid_systems = ['*']
    self.valid_prog_environs = ['*']
    self.modules = ['cmake']
    self.case_name = case_name
    self.num_nodes = num_nodes
    self.ci_mode = ci_mode
    self.device_id = 0

  def set_directory(self, directory):
    self.sourcesdir = directory
    self.readonly_files = [ f'{self.case_name}.re2' ]

  @run_after('setup')
  def read_extras(self):
    self.backend = self.current_environ.extras.get('backend', 'serial')
    self.ranks_per_node = self.current_environ.extras.get('ranks_per_node', 1)
    self.cpu_bind = self.current_environ.extras.get('cpu_bind', '')
    project = self.current_environ.extras.get('project', '')
    self.extra_resources = {
        'account': {'account': project},
        'queue': {'queue': 'debug'},
        'filesystem': {'filesystem': 'home:flare'},
    }

  @run_after('setup')
  def set_paths_exec(self):
    self.nekrs_home = os.path.realpath(self.nekrs_build.install_path)
    self.nekrs_binary = os.path.join(self.nekrs_build.binary_path,'nekrs')
    self.executable = f'{self.nekrs_binary}'

  @run_before('run')
  def make_lhelper(self):
    lhelper_script = self.current_environ.extras.get('lhelper_script', '')
    if lhelper_script != '':
      lhelper_path = os.path.join(self.stagedir, '.lhelper')
      with open(lhelper_path, 'w') as f:
        f.write(lhelper_script)
        os.chmod(lhelper_path, 0o755)
      self.executable = f'./.lhelper {self.executable}'

  @run_before('run')
  def set_environment(self):
    self.env_vars |= {
      'LD_LIBRARY_PATH' : f'$LD_LIBRARY_PATH:{self.nekrs_build.install_path}/lib',
      'NEKRS_HOME' : self.nekrs_home
    }
       
  @run_before('run')
  def set_scheduler_and_launcher_options(self):
    self.time_limit = '1h'
    self.num_tasks = self.num_nodes * self.ranks_per_node
    self.num_tasks_per_node = self.ranks_per_node
    if self.cpu_bind != '':
      self.job.launcher.options += [
        f'-ppn {self.ranks_per_node}',
        f'--cpu-bind={self.cpu_bind}'
      ]

  @run_before('run')
  def set_executable_options(self):
    self.executable_opts += [
      f'--setup {self.case_name}',
      f'--backend {self.backend}',
      f'--device-id {self.device_id}',
      f'--cimode {self.ci_mode}'
    ]

  @sanity_function
  def check_exit_code(self):
    stdout = sn.evaluate(self.stdout)
    last_lines = '\n'.join(osext.tail(stdout, num_lines=5))
    return sn.assert_found_s(r'finished with exit code 0', last_lines, msg='finished with non-zero exit code.')

  @performance_function('fom')
  def fom(self):
    solve_times = sn.extractall(r'solve\s+(\S+)s', self.stdout, 1, float)
    if len(sn.evaluate(solve_times)) > 0:
      fom = 1.0 / solve_times[-1]
    else:
      fom = 1.0
    return fom



class NekRSTestBuildOnly(NekRSTest):
  def __init__(self, case_name, num_nodes, ci_mode):
    super().__init__(case_name, num_nodes, ci_mode)

  @run_before('run')
  def set_build_only(self):
    self.executable_opts += [
      f'--build-only {self.num_tasks}'
    ]

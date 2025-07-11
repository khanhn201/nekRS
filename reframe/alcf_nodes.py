import reframe as rfm
import reframe.utility.typecheck as typ
from reframe.core.backends import getlauncher
import subprocess

good_nodes_cmd = ['pbsnodes -a -F json | jq -r \'.nodes[] | select( (.state=="free" or .state=="job-exclusive") and .resources_available.at_queue=="lustre_scaling" and .resources_available.debug!="True" and .resources_available.validation!="True" and .resources_available.broken!="True") | [.resources_available.host,.state] | @tsv\'']
valid_filesystems_cmd = ['qstat -Bf -F json | jq -r \'.Server[] | [.resources_available.valid_filesystems] | @tsv\'']

def good_nodes():
    cmd = good_nodes_cmd

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, shell=True)

    output, error = process.communicate()

    return output

def valid_filesystems():
    process = subprocess.Popen(valid_filesystems_cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, shell=True)

    output, error = process.communicate()

    return output.rstrip().split(',')

# Class to provide defaults and control for ALCF scheduler options
# Setting a variable from the command line is the highest precedence
# Child classes can overwrite defaults from super().__init__ or the
# set_<option> functions after test setup.
class ALCFTest(rfm.RegressionTest):
    project = variable(str, value='')
    queue = variable(str, value='')
    nodes = variable(typ.List[str], value=['any'])
    num_nodes = variable(int, value=0)
    ppn = variable(int, value=-1)
    build_on_compute = variable(typ.Bool, value=False)
    filesystems = variable(typ.List[str], value=['home'])

    def __init__(self, project='Operations', queue='diag', num_nodes=1, walltime='00:15:00', build_walltime='00:30:00', use_local_launcher=False):
        super().__init__()

        self._project = project
        self._queue = queue
        self._num_nodes = num_nodes
        self._walltime = walltime
        self._build_walltime = build_walltime
        self._use_local_launcher = use_local_launcher
        self._filesystems = None

        self.compute_nodelist = list()
        self.gateway_nodelist = list()
        self.login_nodelist = list()
        self.account_select = None
        self.queue_select = None
        self.vnode_select = None
        self.walltime_select = None
        self.build_walltime_select = None
        self.filesystem_select = None

        # Get the list of nodes on the system
        if self.current_system.name == 'sunspot':
            self.compute_nodelist = ["x%sc%ss%sb0n0" % (rack, chassis, slot) for rack in range(1921,1923) for chassis in range(8) for slot in range(8)]
        elif self.current_system.name == 'aurora':
            self.compute_nodelist = ["x4%s%02dc%ss%sb0n0" % (row, rack, chassis, slot) for row in range(0,8) for rack in range(0,21) for chassis in range(8) for slot in range(8)]
        elif self.current_system.name == 'polaris':
            self.compute_nodelist = ["x%sc0s%sb%sn0" % (rack, slot, blade) for rack in list(range(3001,3017)) + list(range(3101,3113)) + list(range(3201,3213)) for slot in [1,7,13,19,25,31,37] for blade in [0,1]]
            self.gateway_nodelist = [*(f'polaris-gateway-{i:02}' for i in range(1,51))]
            self.login_nodelist = [*(f'polaris-login-{i:02}' for i in range(1,7))]

    @run_after('setup')
    def check_pbs_nodes(self):
        # Early exit if node is not available to be scheduled
        if self.nodes != ['any']:
            if set(self.nodes).issubset(self.compute_nodelist):
                if not set(self.nodes).issubset(good_nodes().split()):
                    bad_nodes = list()
                    for node in self.nodes:
                        if node not in good_nodes().split():
                            bad_nodes.append(node)
                    self.skip(f'\nThe following nodes are down/offline/reserved:\n\t%s' % ('\n\t'.join(bad_nodes)))

            # Nodes were specified that aren't in the system
            elif not set(self.nodes).issubset(self.gateway_nodelist + self.login_nodelist):
                bad_nodes = list()
                for node in self.nodes:
                    if node not in self.compute_nodelist + self.gateway_nodelist + self.login_nodelist:
                        bad_nodes.append(node)
                self.skip(f'\nThe following nodes are not in any of system node lists:\n\t%s' % ('\n\t'.join(bad_nodes)))

    @run_after('setup')
    def check_filesystems(self):
        validated_fs = list()
        valid_fs = valid_filesystems()
        for fs in self.filesystems:
            # First we check if the provided filesystem ends in '_fs' and add it if not because
            # the PBS resources all end in FS
            if not fs.endswith('_fs'):
                fs = fs + '_fs'
            # If the filesystem is valid, let's add it to our list to use
            if fs in valid_fs:
                validated_fs.append(fs)
            else:
                print(f'{fs} is not one of the valid filesystems: {valid_fs}. Ignoring.')
        # If our list of validated filesystems is still empty, no valid filesystems were provided, so skip the test 
        if len(validated_fs) != 0:
            self._filesystems = ':'.join(validated_fs)
        else:
            self.skip('No valid filesystems were provided. See previous output')

    def define_scheduler_options(self):
        # Use variable if set
        self.account_select = f'-A {self.project}' if self.project != '' else f'-A {self._project}'
        self.queue_select = f'-q {self.queue}' if self.queue != '' else f'-q {self._queue}'
        self.walltime_select = f'-l walltime={self._walltime}'
        self.build_walltime_select = f'-l walltime={self._build_walltime}'
        self.filesystem_select = f'-l filesystems={self._filesystems}'

        # If no specific nodes were specified
        if self.nodes == ['any']:
            self.vnode_select = f'-l select={self.num_nodes}' if self.num_nodes != 0 else f'-l select={self._num_nodes}'

        # Early exit if node is not available to be scheduled
        elif set(self.nodes).issubset(self.compute_nodelist):
            self.vnode_select = '-l select=vnode='
            self.vnode_select += '+vnode='.join(self.nodes)
            if len(self.nodes) != self.num_nodes:
                self.vnode_select += f'+%i' % (self.num_nodes - len(self.nodes))

    # Setters for child classes that need to modify after __init__
    def set_walltime(self, walltime):
        self._walltime = walltime

    def set_build_walltime(self, build_walltime):
        self._build_walltime = build_walltime

    def set_num_nodes(self, num_nodes):
        self._num_nodes = num_nodes


    @run_before('run')
    def set_scheduler_options(self):
        self.define_scheduler_options()
        self.job.options = [self.account_select, self.queue_select, self.vnode_select, self.walltime_select, self.filesystem_select]

    @run_before('setup')
    def set_build_locally(self):
        if self.build_on_compute:
            self.build_locally = False

    @run_before('compile')
    def set_build_scheduler_options(self):
        if self.build_on_compute:
            self.define_scheduler_options()
            self.build_job.options = [self.account_select, self.queue_select, self.vnode_select, self.build_walltime_select, self.filesystem_select]

    @run_before('run')
    def set_launcher(self):
        if self._use_local_launcher:
            self.job.launcher = getlauncher('local')()

    @run_before('run')
    def set_post_run_cmds(self):
        self.postrun_cmds = [
            f'cat /sys/module/i915/version >> {self._stagedir}/driver_info.txt', 
            f'rpm -qa | grep fw >> {self._stagedir}/driver_info.txt', 
            f'env | sort > {self._stagedir}/env.txt',
            'module restore',
            f'env | sort > {self._stagedir}/default_env.txt',
            f'diff {self._stagedir}/default_env.txt {self._stagedir}/env.txt > {self._stagedir}/envdiff.txt',
        ]

        self.keep_files += ['envdiff.txt', 'driver_info.txt', 'env.txt']


class ALCFRunOnlyTest(rfm.RunOnlyRegressionTest):
    project = variable(str, value='')
    queue = variable(str, value='')
    nodes = variable(typ.List[str], value=['any'])
    num_nodes = variable(int, value=0)
    ppn = variable(int, value=-1)
    filesystems = variable(typ.List[str], value=['home'])

    def __init__(self, project='Operations', queue='diag', num_nodes=1, walltime='00:15:00', use_local_launcher=False):
        super().__init__()

        self._project = project
        self._queue = queue
        self._num_nodes = num_nodes
        self._walltime = walltime
        self._use_local_launcher = use_local_launcher

        self.system_nodelist = list()
        self.gateway_nodelist = list()
        self.login_nodelist = list()
        self.account_select = None
        self.queue_select = None
        self.vnode_select = None
        self.walltime_select = None
        self.filesystem_select = None

        # Get the list of nodes on the system
        if self.current_system.name == 'sunspot':
            self.compute_nodelist = ["x%sc%ss%sb0n0" % (rack, chassis, slot) for rack in range(1921,1923) for chassis in range(8) for slot in range(8)]
        elif self.current_system.name == 'aurora':
            self.compute_nodelist = ["x4%s%02dc%ss%sb0n0" % (row, rack, chassis, slot) for row in range(0,8) for rack in range(0,21) for chassis in range(8) for slot in range(8)]
        elif self.current_system.name == 'polaris':
            self.compute_nodelist = ["x%sc0s%sb%sn0" % (rack, slot, blade) for rack in list(range(3001,3017)) + list(range(3101,3113)) + list(range(3201,3213)) for slot in [1,7,13,19,25,31,37] for blade in [0,1]]
            self.gateway_nodelist = [*(f'polaris-gateway-{i:02}' for i in range(1,51))]
            self.login_nodelist = [*(f'polaris-login-{i:02}' for i in range(1,7))]

    @run_after('setup')
    def check_pbs_nodes(self):
        # Early exit if node is not available to be scheduled
        if self.nodes != ['any']: 
            if set(self.nodes).issubset(self.compute_nodelist):
                if not set(self.nodes).issubset(good_nodes().split()):
                    bad_nodes = list()
                    for node in self.nodes:
                        if node not in good_nodes().split():
                            bad_nodes.append(node)
                    self.skip(f'\nThe following nodes are down/offline/reserved:\n\t%s' % ('\n\t'.join(bad_nodes)))

            # Nodes were specified that aren't in the system
            elif not set(self.nodes).issubset(self.gateway_nodelist + self.login_nodelist):
                bad_nodes = list()
                for node in self.nodes:
                    if node not in self.compute_nodelist + self.gateway_nodelist + self.login_nodelist:
                        bad_nodes.append(node)
                self.skip(f'\nThe following nodes are not in any of system node lists:\n\t%s' % ('\n\t'.join(bad_nodes)))

    @run_after('setup')
    def check_filesystems(self):
        validated_fs = list()
        valid_fs = valid_filesystems()
        for fs in self.filesystems:
            # First we check if the provided filesystem ends in '_fs' and add it if not because
            # the PBS resources all end in FS
            if not fs.endswith('_fs'):
                fs = fs + '_fs'
            # If the filesystem is valid, let's add it to our list to use
            if fs in valid_fs:
                validated_fs.append(fs)
            else:
                print(f'{fs} is not one of the valid filesystems: {valid_fs}. Ignoring.')
        # If our list of validated filesystems is still empty, no valid filesystems were provided, so skip the test 
        if len(validated_fs) != 0:
            self._filesystems = ':'.join(validated_fs)
        else:
            self.skip('No valid filesystems were provided. See previous output')

    def set_walltime(self, walltime):
        self._walltime = walltime

    def set_num_nodes(self, num_nodes):
        self._num_nodes = num_nodes


    def define_scheduler_options(self):
        self.account_select = f'-A {self.project}' if self.project != '' else f'-A {self._project}'
        self.queue_select = f'-q {self.queue}' if self.queue != '' else f'-q {self._queue}'
        self.walltime_select = f'-l walltime={self._walltime}'
        self.filesystem_select = f'-l filesystems={self._filesystems}'

        # If no specific nodes were specified
        if self.nodes == ['any']:
            self.vnode_select = f'-l select={self.num_nodes}' if self.num_nodes != 0 else f'-l select={self._num_nodes}'

        # Early exit if node is not available to be scheduled
        elif set(self.nodes).issubset(self.compute_nodelist):
            self.vnode_select = '-l select=vnode='
            self.vnode_select += '+vnode='.join(self.nodes)
            if len(self.nodes) != self.num_nodes:
                self.vnode_select += f'+%i' % (self.num_nodes - len(self.nodes))

    @run_before('run')
    def set_scheduler_options(self):
        self.define_scheduler_options()
        self.job.options = [self.account_select, self.queue_select, self.vnode_select, self.walltime_select, self.filesystem_select]

    @run_before('run')
    def set_launcher(self):
        if self._use_local_launcher:
            self.job.launcher = getlauncher('local')()

    @run_before('run')
    def set_post_run_cmds(self):
        self.postrun_cmds = [
            f'cat /sys/module/i915/version >> {self._stagedir}/driver_info.txt', 
            f'rpm -qa | grep fw >> {self._stagedir}/driver_info.txt', 
            f'env | sort > {self._stagedir}/env.txt',
            'module restore',
            f'env | sort > {self._stagedir}/default_env.txt',
            f'diff {self._stagedir}/default_env.txt {self._stagedir}/env.txt > {self._stagedir}/envdiff.txt',
        ]

        self.keep_files += ['envdiff.txt', 'driver_info.txt', 'env.txt']

class ALCFCompileOnlyTest(rfm.CompileOnlyRegressionTest):
    project = variable(str, value='')
    queue = variable(str, value='')
    nodes = variable(typ.List[str], value=['any'])
    num_nodes = variable(int, value=0)
    ppn = variable(int, value=-1)
    build_on_compute = variable(typ.Bool, value=False)
    filesystems = variable(typ.List[str], value=['home'])

    def __init__(self, project='Operations', queue='diag', num_nodes=1, walltime='00:15:00'):
        super().__init__()

        self._project = project
        self._queue = queue
        self._num_nodes = num_nodes
        self._walltime = walltime

        self.compute_nodelist = list()
        self.gateway_nodelist = list()
        self.login_nodelist = list()
        self.account_select = None
        self.queue_select = None
        self.vnode_select = None
        self.walltime_select = None
        self.filesystem_select = None

        # Get the list of nodes on the system
        if self.current_system.name == 'sunspot':
            self.compute_nodelist = ["x%sc%ss%sb0n0" % (rack, chassis, slot) for rack in range(1921,1923) for chassis in range(8) for slot in range(8)]
        elif self.current_system.name == 'aurora':
            self.compute_nodelist = ["x4%s%02dc%ss%sb0n0" % (row, rack, chassis, slot) for row in range(0,8) for rack in range(0,21) for chassis in range(8) for slot in range(8)]
        elif self.current_system.name == 'polaris':
            self.compute_nodelist = ["x%sc0s%sb%sn0" % (rack, slot, blade) for rack in list(range(3001,3017)) + list(range(3101,3113)) + list(range(3201,3213)) for slot in [1,7,13,19,25,31,37] for blade in [0,1]]
            self.gateway_nodelist = [*(f'polaris-gateway-{i:02}' for i in range(1,51))]
            self.login_nodelist = [*(f'polaris-login-{i:02}' for i in range(1,7))]

    @run_after('setup')
    def check_pbs_nodes(self):
        # Early exit if node is not available to be scheduled
        if self.nodes != ['any']: 
            if set(self.nodes).issubset(self.compute_nodelist):
                if not set(self.nodes).issubset(good_nodes().split()):
                    bad_nodes = list()
                    for node in self.nodes:
                        if node not in good_nodes().split():
                            bad_nodes.append(node)
                    self.skip(f'\nThe following nodes are down/offline/reserved:\n\t%s' % ('\n\t'.join(bad_nodes)))

            # Nodes were specified that aren't in the system
            elif not set(self.nodes).issubset(self.gateway_nodelist + self.login_nodelist):
                bad_nodes = list()
                for node in self.nodes:
                    if node not in self.compute_nodelist + self.gateway_nodelist + self.login_nodelist:
                        bad_nodes.append(node)
                self.skip(f'\nThe following nodes are not in any of system node lists:\n\t%s' % ('\n\t'.join(bad_nodes)))

    @run_after('setup')
    def check_filesystems(self):
        validated_fs = list()
        valid_fs = valid_filesystems()
        for fs in self.filesystems:
            # First we check if the provided filesystem ends in '_fs' and add it if not because
            # the PBS resources all end in FS
            if not fs.endswith('_fs'):
                fs = fs + '_fs'
            # If the filesystem is valid, let's add it to our list to use
            if fs in valid_fs:
                validated_fs.append(fs)
            else:
                print(f'{fs} is not one of the valid filesystems: {valid_fs}. Ignoring.')
        # If our list of validated filesystems is still empty, no valid filesystems were provided, so skip the test 
        if len(validated_fs) != 0:
            self._filesystems = ':'.join(validated_fs)
        else:
            self.skip('No valid filesystems were provided. See previous output')

    def set_walltime(self, walltime):
        self._walltime = walltime

    def set_num_nodes(self, num_nodes):
        self._num_nodes = num_nodes

    def define_scheduler_options(self):
        self.account_select = f'-A {self.project}' if self.project != '' else f'-A {self._project}'
        self.queue_select = f'-q {self.queue}' if self.queue != '' else f'-q {self._queue}'
        self.walltime_select = f'-l walltime={self._walltime}'
        self.filesystem_select = f'-l filesystems={self._filesystems}'

        # If no specific nodes were specified
        if self.nodes == ['any']:
            self.vnode_select = f'-l select={self.num_nodes}' if self.num_nodes != 0 else f'-l select={self._num_nodes}'

        # Early exit if node is not available to be scheduled
        elif set(self.nodes).issubset(self.compute_nodelist):
            self.vnode_select = '-l select=vnode='
            self.vnode_select += '+vnode='.join(self.nodes)
            if len(self.nodes) != self.num_nodes:
                self.vnode_select += f'+%i' % (self.num_nodes - len(self.nodes))

    @run_before('setup')
    def set_build_locally(self):
        if self.build_on_compute:
            self.build_locally = False

    @run_before('compile')
    def set_scheduler_options(self):
        if self.build_on_compute:
            self.define_scheduler_options()
            self.job.options = [self.account_select, self.queue_select, self.vnode_select, self.walltime_select, self.filesystem_select]


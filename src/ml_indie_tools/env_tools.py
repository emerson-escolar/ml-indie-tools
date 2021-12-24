'''Tools to configure ML environment for Tensorflow, Pytorch or JAX and 
optional notebook/colab environment'''

import os
import sys
import time
import shutil

class MLEnv():
    """ Initialize platform and accelerator. 
    
    This checks initialization and available accelerator hardware for different ml platforms.
    At return, the following variables are set: `self.is_tensorflow`, `self.is_pytorch`, `self.is_jax`,
    indicating that the ml environment is available for Tensorflow, Pytorch or JAX respectively if `True`.
    `self.is_notebook` and `self.is_colab` indicate if the environment is a notebook or colab environment.
    `self.is_gpu` indicates if the environment is a GPU environment, `self.is_tpu` indicates if the 
    environment is a TPU environment, and `self.is_cpu` that no accelerator is available.
    
    :param platform: Known platforms are: `'tf'` (tensorflow), `'pt'` (pytorch), and `'jax'`
    :param accelerator: known accelerators are: `'fastest'` (pick best available hardware), `'cpu'`, `'gpu'`, `'tpu'`.
    :param verbose: show information about configuration
    """

    def __init__(self, platform='tf', accelerator='fastest', verbose=False):
        self.known_platforms = ['tf', 'pt', 'jax']
        self.known_accelerators = ['cpu', 'gpu', 'tpu', 'fastest']
        if platform not in self.known_platforms:
            print(f"Platform {platform} is not known, please check spelling.")
            return
        if accelerator not in self.known_accelerators:
            print(f"Accelerator {accelerator} is not known, please check spelling.")
            return
        self.platform = platform
        self.accelerator = accelerator
        self.is_tensorflow = False
        self.is_pytorch = False
        self.is_jax = False
        self.is_cpu = False
        self.is_gpu = False
        self.is_tpu = False
        self.is_notebook = False
        self.is_colab = False
        if self.platform == 'tf':
            try:
                import tensorflow as tf
                self.is_tensorflow = True
            except ImportError as e:
                if verbose is True:
                    print(f"Tensorflow not available: {e}")
                return
            try:
                from tensorflow.python.profiler import profiler_client
                self.tf_prof = True
            except:
                self.tf_prof = False
            # %tensorflow_version 2.x
            # import tensorflow as tf
            if verbose is True:
                print("Tensorflow version: ", tf.__version__)
            if self.accelerator == 'tpu' or self.accelerator == 'fastest':
                try:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
                    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
                    self.is_tpu = True
                except ValueError:
                    tpu = None
                    if self.accelerator!= 'fastest':
                        if verbose is True:
                            print("No TPU available")
                if self.is_tpu is True:    
                    tf.config.experimental_connect_to_cluster(tpu)
                    tf.tpu.experimental.initialize_tpu_system(tpu)
                    self.tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
                    self.tpu_num_nodes = len(self.tpu_strategy.extended.worker_devices)
                    if verbose is True:
                        print("TPU strategy available")
            if self.is_tpu is False:
                if self.accelerator == 'gpu' or self.accelerator == 'fastest':
                    try:
                        tf.config.experimental.list_physical_devices('GPU')
                        self.is_gpu = True
                    except RuntimeError as e:
                        if verbose is True:
                            print(f"GPU not available: {e}")
                        self.is_gpu = False
                    if self.is_gpu is True:
                        if verbose is True:
                            print("GPU available")
            if self.is_gpu is False and self.is_tpu is False:
                if verbose is True:
                    print("No GPU or TPU available, this is going to be very slow!")

        if self.platform == 'jax':
            try:
                import jax
                self.is_jax = True
            except ImportError:
                if verbose is True:
                    print("Jax not available")
            if self.is_jax is True:
                if self.accelerator == 'tpu' or self.accelerator == 'fastest':
                    try:
                        import jax.tools.colab_tpu
                        jax.tools.colab_tpu.setup_tpu()
                        self.is_tpu = True
                        if verbose is True:
                            print("JAX TPU detected.")
                    except:
                        if self.accelerator != 'fastest':
                            if verbose is True:
                                print("JAX TPU not detected.")
                                return
                if self.accelerator == 'gpu' or self.accelerator == 'fastest':
                    try:
                        jd=jax.devices()[0]
                        gpu_device_names = ['Tesla', 'GTX', 'Nvidia']  # who knows?
                        for gpu_device_name in gpu_device_names:
                            if gpu_device_name in jd.device_kind:
                                self.is_gpu = True
                                if verbose is True:
                                    print(f"JAX GPU: {jd.device_kind} detected.")
                                break
                        if self.is_gpu is False:
                            if verbose is True:
                                print("JAX GPU not available.")
                    except:
                        if self.accelerator != 'fastest':
                            if verbose is True:
                                print("JAX GPU not available.")
                                return
                if self.accelerator == 'cpu' or self.accelerator == 'fastest':
                    try:
                        jd=jax.devices()[0]
                        cpu_device_names = ['CPU', 'cpu']  
                        for cpu_device_name in cpu_device_names:
                            if cpu_device_name in jd.device_kind:
                                self.is_cpu = True
                                if verbose is True:
                                    print(f"JAX CPU: {jd.device_kind} detected.")
                                break
                        if self.is_cpu is False:
                            if verbose is True:
                                print("JAX CPU not available.")
                    except:
                        print("No JAX CPU available.")
                        return
        if self.platform == 'pt':
            try:
                import torch
                self.is_pytorch = True
            except ImportError:
                print("Pytorch not available.")
                return
            if self.is_pytorch is True:
                if self.accelerator == 'tpu' or self.accelerator == 'fastest':
                    tpu_env=False
                    try:
                        assert os.environ['COLAB_TPU_ADDR']
                        tpu_env = True
                    except:
                        if verbose is True and self.accelerator != 'fastest':
                            print("Pytorch TPU instance not detected.")
                    if tpu_env is True:
                        try:
                            import torch
                            if '1.9.' not in torch.__version__ and verbose is True:
                                print("Pytorch version probably not supported with TPUs. Try (as of 12/2021): ")
                                print("!pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl")
                            import torch_xla.core.xla_model as xm
                            self.is_tpu = True
                            if verbose is True:
                                print("Pytorch TPU detected.")
                        except:
                            print("Pytorch TPU would be available, but failed to\
                                    import torch_xla.core.xla_model.")
                            if self.accelerator != 'fastest':
                                return
                if self.accelerator == 'gpu' or self.accelerator == 'fastest':
                    try:
                        import torch.cuda
                        if torch.cuda.is_available():
                            self.is_gpu = True
                            if verbose is True:
                                print("Pytorch GPU detected.")
                        else:
                            if verbose is True:
                                print("Pytorch GPU not available.")
                    except:
                        if self.accelerator != 'fastest':
                            if verbose is True:
                                print("Pytorch GPU not available.")
                                return
                if self.accelerator == 'cpu' or self.accelerator == 'fastest':
                    self.is_cpu = True
                    if verbose is True:
                        print("Pytorch CPU detected.")
                else:
                    if verbose is True:
                        print("No Pytorch CPU accelerator available.")
                    return
        self.flush_timer = 0
        self.flush_timeout = 180
        self.check_notebook_type(verbose=verbose)
        # self.check_hardware(verbose=verbose)

    def check_notebook_type(self, verbose=False):
        try:
            if 'IPKernelApp' in get_ipython().config:
                self.is_notebook = True
                if verbose is True:
                    print("You are on a Jupyter instance.")
        except NameError:
            self.is_notebook = False
            if verbose is True:
                print("You are not on a Jupyter instance.")
        if self.is_notebook is True:
            try: # Colab instance?
                from google.colab import drive
                self.is_colab = True
                if self.is_tensorflow is True:
                    get_ipython().run_line_magic('load_ext', 'tensorboard')
                    try:
                        get_ipython().run_line_magic('tensorflow_version', '2.x')
                    except:
                        pass
                if verbose is True:
                    print("You are on a Colab instance.")
            except: # Not? ignore.
                self.is_colab = False
                if verbose is True:
                    print("You are not on a Colab instance, so no Google Drive access is possible.")
                pass
        return self.is_notebook, self.is_colab

    def describe(self, return_dict=False, verbose=False):
        """Describe machine learning environment.

        This lists the machine learning environment, os, python version, ml lib versions and hardware
        either as text string or a dictionary of key-value pairs.
        
        Example output for default string: `'Darwin, Python 3.9.9 (conda), Jupyter-instance, Tensorflow 2.7.0 GPU (METAL)'`

        For `return_dict=True`: `{'os': 'Darwin', 'python': '3.9.9', 'conda': True, 'colab': False, 'jupyter': True, 'ml_platform': 'tensorflow', 'ml_version': '2.7.0', 'ml_accelerator': 'GPU', 'ml_accelerator_desc': 'METAL' }`
        
        :param return_dict: If True, return a dictionary of the results, otherwise return a string.
        :param verbose: If True, print debug infos.
        """
        res={}
        ospl=sys.platform
        ospl=ospl[0].upper()+ospl[1:]
        pyver=sys.version.split(' ')[0]
        ospyver = f"{ospl}, Python {pyver}"
        res['os'] = ospl
        res['python'] = pyver
        if 'conda' in sys.version:
            ospyver += ' (conda)'
            res['conda'] = True
        else:
            res['conda'] = False
        if self.is_notebook:
            if self.is_colab:
                ospyver += ', Colab-instance'
                res['colab'] = True
                res['jupyter'] = True
            else:
                ospyver += ', Jupyter-instance'
                res['colab'] = False
                res['jupyter'] = True
        else:
            res['colab'] = False
            res['jupyter'] = False  
        if self.is_tensorflow is True:
            import tensorflow as tf
            desc=f'{ospyver}, Tensorflow {tf.__version__} '
            res['ml_platform'] = 'tensorflow'
            res['ml_version'] = tf.__version__
            if self.is_tpu is True:
                res['ml_accelerator'] = 'TPU'
                tpu_profile_service_address = os.environ['COLAB_TPU_ADDR'].replace('8470', '8466')
                tpu_desc = f"TPU, {self.tpu_num_nodes} nodes"
                res['ml_accelerator_desc'] = tpu_desc
                if self.tf_prof is True:
                    state=profiler_client.monitor(tpu_profile_service_address, 100, 2)
                    if 'TPU v2' in state:
                        tpu_desc=tpu_desc+'v2 (8GB)'  # that's what you currently get on Colab    
                        if verbose is True:
                            print("WARNING: you got old TPU v2 which is limited to 8GB Ram.")
                desc=desc+tpu_desc
            elif self.is_gpu is True:
                res['ml_accelerator'] = 'GPU'
                try:
                    gpu_name=tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0])['device_name']
                    res['ml_accelerator_desc'] = gpu_name
                    desc=desc+f'GPU ({gpu_name})'
                except:
                    desc=desc+'GPU (unknown)'
                    res['ml_accelerator_desc'] = 'unknown'
            elif self.is_cpu is True:
                desc=desc+'CPU'
                res['ml_accelerator'] = 'CPU'
                res['ml_accelerator_desc'] = ''
            else:
                desc=desc+'unknown device (error)'
                res['ml_accelerator'] = 'unknown'
                res['ml_accelerator_desc'] = 'unknown'
        elif self.is_pytorch is True:
            desc='Pytorch '+torch.__version__
        elif self.is_jax is True:
            desc='JAX '+jax.__version__
        else:
            desc='Unknown'
            res['ml_platform'] = 'unknown'
            res['ml_version'] = 'unknown'
            res['accelerator'] = 'unknown'
        if return_dict is True:
            return res
        else:
            return desc

    # Hardware check:
    def check_hardware(self, verbose=True):
        self.is_tpu = False
        self.tpu_is_init = False
        self.is_gpu = False
        self.tpu_address = None

        if self.is_tensorflow is True:
            if self.is_colab:
                try:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
                    if verbose is True:
                        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
                    self.is_tpu = True
                    tpu_profile_service_address = os.environ['COLAB_TPU_ADDR'].replace('8470', '8466')
                    state=profiler_client.monitor(tpu_profile_service_address, 100, 2)
                    if 'TPU v2' in state:
                        print("WARNING: you got old TPU v2 which is limited to 8GB Ram.")

                except ValueError:
                    if verbose is True:
                        print("No TPU available")
                    self.is_tpu = False

            for hw in ["CPU", "GPU", "TPU"]:
                hw_list=tf.config.experimental.list_physical_devices(hw)
                if len(hw_list)>0:
                    if hw=='TPU':
                        self.is_tpu=True
                    if hw=='GPU':
                        self.is_gpu=True
                    if verbose is True:
                        print(f"{hw}: {hw_list} {tf.config.experimental.get_device_details(hw_list[0])}") 

            if not self.is_tpu:
                if not self.is_gpu:
                    if verbose is True:
                        print("WARNING: You have neither TPU nor GPU, this is going to be very slow!")
                else:
                    if verbose is True:
                        print("GPU available")
            else:
                tf.compat.v1.disable_eager_execution()
                if verbose is True:
                    print("TPU: eager execution disabled!")
        else:
            print("Tensorflow not available, so no hardware check (yet).")

    def mount_gdrive(self, mount_point="/content/drive", root_path="/content/drive/My Drive", verbose=True):
        if self.is_colab is True:
            if verbose is True:
                print("You will now be asked to authenticate Google Drive access in order to store training data (cache) and model state.")
                print("Changes will only happen within Google Drive directory `My Drive/Colab Notebooks/ALU_Net`.")
            if not os.path.exists(root_path):
                # drive.flush_and_unmount()
                drive.mount(mount_point) #, force_remount=True)
                return True, root_path
            if not os.path.exists(root_path):
                print(f"Something went wrong with Google Drive access. Cannot save model to {root_path}")
                return False, None
            else:
                return True, root_path
        else:
            if verbose is True:
                print("You are not on a Colab instance, so no Google Drive access is possible.")
            return False, None

    def init_paths(self, project_name, model_name, model_variant=None, log_to_gdrive=False):
        self.save_model = True
        self.model_path=None
        self.cache_path=None
        self.weights_file = None
        self.project_path = None
        self.log_path = "./logs"
        self.log_to_gdrive = log_to_gdrive
        self.log_mirror_path = None
        if self.is_colab:
            self.save_model, self.root_path = self.mount_gdrive()
        else:
            self.root_path='.'

        print(f"Root path: {self.root_path}")
        if self.save_model:
            if self.is_colab:
                self.project_path=os.path.join(self.root_path,f"Colab Notebooks/{project_name}")
                if log_to_gdrive is True:
                    self.log_mirror_path = os.path.join(self.root_path,f"Colab Notebooks/{project_name}/logs")
                    print(f"Logs will be mirrored to {self.log_mirror_path}, they can be used with a remote Tensorboard instance.")
            else:
                self.project_path=self.root_path
            if model_variant is None:
                self.model_path=os.path.join(self.project_path,f"{model_name}")
                self.weights_file=os.path.join(self.project_path,f"{model_name}_weights.h5")
            else:
                self.model_path=os.path.join(self.project_path,f"{model_name}_{model_variant}")
                self.weights_file=os.path.join(self.project_path,f"{model_name}_{model_variant}_weights.h5")
            self.cache_path=os.path.join(self.project_path,'data')
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            if self.is_tpu is False:
                print(f"Model save-path: {self.model_path}")
            else:
                print(f"Weights save-path: {self.weights_file}")
            print(f'Data cache path {self.cache_path}')
        return self.root_path, self.project_path, self.model_path, self.weights_file, self.cache_path, self.log_path

    def gdrive_log_mirror(self):
        # copy directory self.log_path to self.log_mirror_path
        if self.log_to_gdrive is True:
            if self.log_mirror_path is not None:
                if len(self.log_mirror_path)>4 and self.log_mirror_path[-5:]=='/logs':
                    if os.path.exists(self.log_mirror_path) is True:
                        print(f"Removing old log files from {self.log_mirror_path}")
                        shutil.rmtree(self.log_mirror_path)
                    print(f"Staring tree-copy of files from {self.log_mirror_path}. [This can be astonishingly slow!]")
                    shutil.copytree(self.log_path, self.log_mirror_path)
                    print(f"Tensorboard data mirrored to {self.log_mirror_path}")
                else:
                    print(f"Log-mirror path is not valid: {self.log_mirror_path}, it needs to end with '/logs' as sanity-check")

    def epoch_time_func(self, epoch, log):
        if self.log_to_gdrive is True:
            if time.time() - self.flush_timer > self.flush_timeout:
                self.flush_timer=time.time()
                self.gdrive_log_mirror()


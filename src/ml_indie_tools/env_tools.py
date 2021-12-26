'''Tools to configure ML environment for Tensorflow, Pytorch or JAX and 
optional notebook/colab environment'''

import os
import sys
import logging

class MLEnv():
    """ Initialize platform and accelerator. 
    
    This checks initialization and available accelerator hardware for different ml platforms.
    At return, the following variables are set: `self.is_tensorflow`, `self.is_pytorch`, `self.is_jax`,
    indicating that the ml environment is available for Tensorflow, Pytorch or JAX respectively if `True`.
    `self.is_notebook` and `self.is_colab` indicate if the environment is a notebook or colab environment.
    `self.is_gpu` indicates if the environment is a GPU environment, `self.is_tpu` indicates if the 
    environment is a TPU environment, and `self.is_cpu` that no accelerator is available.
    
    The logger `MLEnv` provdides details about the hardware and ml environment.

    :param platform: Known platforms are: `'tf'` (tensorflow), `'pt'` (pytorch), and `'jax'`
    :param accelerator: known accelerators are: `'fastest'` (pick best available hardware), `'cpu'`, `'gpu'`, `'tpu'`.
    """

    def __init__(self, platform='tf', accelerator='fastest'):
        self.log = logging.getLogger('MLEnv')
        self.known_platforms = ['tf', 'pt', 'jax']
        self.known_accelerators = ['cpu', 'gpu', 'tpu', 'fastest']
        if platform not in self.known_platforms:
            self.log.error(f"Platform {platform} is not known, please check spelling.")
            return
        if accelerator not in self.known_accelerators:
            self.log.error(f"Accelerator {accelerator} is not known, please check spelling.")
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
                self.log.error(f"Tensorflow not available: {e}")
                return
            try:
                from tensorflow.python.profiler import profiler_client
                self.tf_prof = True
            except:
                self.tf_prof = False
            self.log.debug("Tensorflow version: ", tf.__version__)
            if self.accelerator == 'tpu' or self.accelerator == 'fastest':
                try:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
                    self.log.debug('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
                    self.is_tpu = True
                except ValueError:
                    tpu = None
                    if self.accelerator!= 'fastest':
                        self.log.debug("No TPU available")
                if self.is_tpu is True:    
                    tf.config.experimental_connect_to_cluster(tpu)
                    tf.tpu.experimental.initialize_tpu_system(tpu)
                    self.tpu_strategy = tf.distribute.TPUStrategy(tpu)
                    self.tpu_num_nodes = len(self.tpu_strategy.extended.worker_devices)
                    self.log.debug("TPU strategy available")
            if self.is_tpu is False:
                if self.accelerator == 'gpu' or self.accelerator == 'fastest':
                    try:
                        tf.config.experimental.list_physical_devices('GPU')
                        self.is_gpu = True
                    except RuntimeError as e:
                        self.log.debug(f"GPU not available: {e}")
                        self.is_gpu = False
                    if self.is_gpu is True:
                        self.log.debug("GPU available")
            if self.is_gpu is False: 
                self.log.info("No GPU or TPU available, this is going to be very slow!")
        if self.platform == 'jax':
            try:
                import jax
                self.is_jax = True
            except ImportError:
                self.log.debug("Jax not available")
            if self.is_jax is True:
                if self.accelerator == 'tpu' or self.accelerator == 'fastest':
                    try:
                        import jax.tools.colab_tpu
                        jax.tools.colab_tpu.setup_tpu()
                        self.is_tpu = True
                        self.log.debug("JAX TPU detected.")
                    except:
                        if self.accelerator != 'fastest':
                            self.log.debug("JAX TPU not detected.")
                            return
                if self.accelerator == 'gpu' or self.accelerator == 'fastest':
                    try:
                        jd=jax.devices()[0]
                        gpu_device_names = ['Tesla', 'GTX', 'Nvidia']  # who knows?
                        for gpu_device_name in gpu_device_names:
                            if gpu_device_name in jd.device_kind:
                                self.is_gpu = True
                                self.log.debug(f"JAX GPU: {jd.device_kind} detected.")
                                break
                        if self.is_gpu is False:
                            self.log.debug("JAX GPU not available.")
                    except:
                        if self.accelerator != 'fastest':
                            self.log.debug("JAX GPU not available.")
                            return
                if self.accelerator == 'cpu' or self.accelerator == 'fastest':
                    try:
                        jd=jax.devices()[0]
                        cpu_device_names = ['CPU', 'cpu']  
                        for cpu_device_name in cpu_device_names:
                            if cpu_device_name in jd.device_kind:
                                self.is_cpu = True
                                self.log.debug(f"JAX CPU: {jd.device_kind} detected.")
                                break
                        if self.is_cpu is False:
                            self.log.debug("JAX CPU not available.")
                    except:
                        self.log.error("No JAX CPU available.")
                        return
        if self.platform == 'pt':
            try:
                import torch
                self.is_pytorch = True
            except ImportError:
                self.log.error("Pytorch not available.")
                return
            if self.is_pytorch is True:
                if self.accelerator == 'tpu' or self.accelerator == 'fastest':
                    tpu_env=False
                    try:
                        assert os.environ['COLAB_TPU_ADDR']
                        tpu_env = True
                    except:
                        self.log.debug("Pytorch TPU instance not detected.")
                    if tpu_env is True:
                        try:
                            import torch
                            if '1.9.' not in torch.__version__:
                                self.log.warning("Pytorch version probably not supported with TPUs. Try (as of 12/2021): ")
                                self.log.warning("!pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl")
                            import torch_xla.core.xla_model as xm
                            self.is_tpu = True
                            self.log.debug("Pytorch TPU detected.")
                        except:
                            self.log.error("Pytorch TPU would be available, but failed to\
                                    import torch_xla.core.xla_model.")
                            if self.accelerator != 'fastest':
                                return
                if self.accelerator == 'gpu' or self.accelerator == 'fastest':
                    try:
                        import torch.cuda
                        if torch.cuda.is_available():
                            self.is_gpu = True
                            self.log.debug("Pytorch GPU detected.")
                        else:
                            self.log.debug("Pytorch GPU not available.")
                    except:
                        if self.accelerator != 'fastest':
                            self.log.error("Pytorch GPU not available.")
                            return
                if self.accelerator == 'cpu' or self.accelerator == 'fastest':
                    self.is_cpu = True
                    self.log.debug("Pytorch CPU detected.")
                else:
                    self.log.error("No Pytorch CPU accelerator available.")
                    return
        self.flush_timer = 0
        self.flush_timeout = 180
        self._check_notebook_type()

    def _check_notebook_type(self):
        """ Internal function, use :func:`describe` instead """
        try:
            if 'IPKernelApp' in get_ipython().config:
                self.is_notebook = True
                self.log.debug("You are on a Jupyter instance.")
        except NameError:
            self.is_notebook = False
            self.log.debug("You are not on a Jupyter instance.")
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
                self.log.debug("You are on a Colab instance.")
            except: # Not? ignore.
                self.is_colab = False
                self.log.debug("You are not on a Colab instance, so no Google Drive access is possible.")
        return self.is_notebook, self.is_colab

    def describe(self, return_dict=False):
        """Describe machine learning environment.

        This lists the machine learning environment, os, python version, ml lib versions and hardware
        either as text string or a dictionary of key-value pairs.
        
        Example output for default string: `'Darwin, Python 3.9.9 (conda), Jupyter-instance, Tensorflow 2.7.0 GPU (METAL)'`

        For `return_dict=True`: `{'os': 'Darwin', 'python': '3.9.9', 'conda': True, 'colab': False, 'jupyter': True, 'ml_platform': 'tensorflow', 'ml_version': '2.7.0', 'ml_accelerator': 'GPU', 'ml_accelerator_desc': 'METAL' }`
        
        :param return_dict: If True, return a dictionary of the results, otherwise return a string.
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
                    from tensorflow.python.profiler import profiler_client
                    state=profiler_client.monitor(tpu_profile_service_address, 100, 2)
                    if 'TPU v2' in state:
                        tpu_desc=tpu_desc+'v2 (8GB)'  # that's what you currently get on Colab    
                        self.log.info("You got old TPU v2 which is limited to 8GB Ram.")
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
            import torch
            desc='Pytorch '+torch.__version__
        elif self.is_jax is True:
            desc='JAX '+jax.__version__
        else:
            import jax
            desc='Unknown'
            res['ml_platform'] = 'unknown'
            res['ml_version'] = 'unknown'
            res['accelerator'] = 'unknown'
        if return_dict is True:
            return res
        else:
            return desc

    def mount_gdrive(self, mount_point="/content/drive", root_path="/content/drive/My Drive"):
        if self.is_colab is True:
            from google.colab import drive
            self.log.info("You will now be asked to authenticate Google Drive access in order to store training data (cache) and model state.")
            self.log.info("Changes will only happen within Google Drive directory `My Drive/Colab Notebooks/ALU_Net`.")
            if not os.path.exists(root_path):
                drive.mount(mount_point)
                return True, root_path
            if not os.path.exists(root_path):
                self.log.error(f"Something went wrong with Google Drive access. Cannot save model to {root_path}")
                return False, None
            else:
                return True, root_path
        else:
            self.log.error("You are not on a Colab instance, so no Google Drive access is possible.")
            return False, None

    def init_paths(self, project_name, model_name, model_variant=None):
        self.save_model = True
        self.model_path=None
        self.cache_path=None
        self.weights_file = None
        self.project_path = None
        self.log_path = "./logs"
        if self.is_colab:
            self.save_model, self.root_path = self.mount_gdrive()
        else:
            self.root_path='.'

        self.log.debug(f"Root path: {self.root_path}")
        if self.save_model:
            if self.is_colab:
                self.project_path=os.path.join(self.root_path,f"Colab Notebooks/{project_name}")
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
                self.log.info(f"Model save-path: {self.model_path}")
            else:
                self.log.info(f"Weights save-path: {self.weights_file}")
            self.log.info(f'Data cache path {self.cache_path}')
        return self.root_path, self.project_path, self.model_path, self.weights_file, self.cache_path, self.log_path


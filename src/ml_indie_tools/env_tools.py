import os
import sys
import time
import shutil

class MLEnv():
    """ Initialize deep learning platform. Known platforms are: 'tf', 'pt',
    'jax', known accelerators are: 'cpu', 'gpu', 'tpu' """
    def __init__(self, platform='tf', accelerator='fastest', verbose=True):
        """ Initialize platform. Known platforms are: 'tf' (tensorflow), 'pt'
        (pytorch), and 'jax', known
        accelerators are: 'fastest' (pick best available hardware), 'cpu', 'gpu', 'tpu' """
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
                    try:
                        assert os.environ['COLAB_TPU_ADDR']
                        import torch_xla.core.xla_model as xm
                        self.is_tpu = True
                        if verbose is True:
                            print("Pytorch TPU detected.")
                    except:
                        if self.accelerator != 'fastest':
                            if verbose is True:
                                print("Pytorch TPU not detected.")
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


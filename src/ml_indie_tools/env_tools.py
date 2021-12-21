import os
import sys
import time
import shutil

class MLEnv():
    """ Initialize deep learning platform. Known platforms are: 'tf', 'pt',
    'jax', known accelerators are: 'cpu', 'gpu', 'tpu' """
    def __init__(self, platform='tf', accelerator='gpu', verbose=True):
        """ Initialize platform. Known platforms are: 'tf', 'pt', 'jax', known
        accelerators are: 'cpu', 'gpu', 'tpu' """
        self.known_platforms = ['tf', 'pt', 'jax']
        self.known_accelerators = ['cpu', 'gpu', 'tpu']
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
            if self.accelerator == 'tpu':
                try:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
                    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
                    self.is_tpu = True
                except ValueError:
                    print('ERROR: Not connected to a TPU runtime!')
                    return  
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                self.tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
                if verbose is True:
                    print("TPU strategy available")
        if self.platform == 'jax':
            if self.accelerator == 'tpu':
                try:
                    import jax.tools.colab_tpu
                    jax.tools.colab_tpu.setup_tpu()
                    self.is_tpu = True
                    if verbose is True:
                        print("JAX TPU detected.")
                except:
                    if verbose is True:
                        print("JAX TPU not available.")
                    return
            elif self.accelerator == 'gpu':
                try:
                    import jax.config
                    jax.config.update_config(jax.config.GPU_DEVICE_NAME, '/gpu:0')
                    self.is_gpu = True
                    if verbose is True:
                        print("JAX GPU detected.")
                except:
                    print("No JAX GPU available.")
                    return
            elif self.accelerator == 'cpu':
                self.is_cpu = True
                if verbose is True:
                    print("JAX CPU detected.")
            try:
                import jax as jnp
                self.is_jax = True
            except ImportError:
                pass
        if self.platform == 'pt':
            try:
                import torch
                self.is_pytorch = True
            except ImportError:
                pass
        self.flush_timer = 0
        self.flush_timeout = 180
        self.is_colab = self.check_colab(verbose=verbose)
        # self.check_hardware(verbose=verbose)

    @staticmethod
    def check_colab(verbose=False):
        try: # Colab instance?
            from google.colab import drive
            is_colab = True
            if self.is_tensorflow is True:
                get_ipython().run_line_magic('load_ext', 'tensorboard')
                try:
                    get_ipython().run_line_magic('tensorflow_version', '2.x')
                except:
                    pass
            if verbose is True:
                print("You are on a Colab instance.")
        except: # Not? ignore.
            is_colab = False
            if verbose is True:
                print("You are not on a Colab instance, so no Google Drive access is possible.")
            pass
        return is_colab

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


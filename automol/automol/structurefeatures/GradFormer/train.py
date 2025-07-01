from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import tensorboard
from torch import optim
import numpy as np
import torch, os ,sys
import time
from tqdm import tqdm
import seaborn as sns
import numpy as np
import subprocess, psutil
import pylab as pl
from IPython import display
import matplotlib.gridspec as gridspec
import pandas as pd
def plot_cpu_perf(axis, cpu_labels, cpu_snapshot):
        axis.cla()
        axis.grid(False)
        axis.set_ylim([0,100])
        axis.set_ylabel('Percent', labelpad=2, fontsize = 16)
        axis.set_xlabel('cpu', labelpad=2, fontsize = 16)
        axis.bar(cpu_labels, cpu_snapshot, color='dodgerblue', edgecolor='none')
        axis.set_title('CPU Utilization', fontsize = 18)

def plot_gpu_perf(axis, gpu_labels, gpu_snapshot):
        axis.cla()
        axis.grid(False)
        axis.set_ylim([0,100])
        axis.set_xticks(gpu_labels)
        axis.set_xlabel('gpu ID', labelpad=2, fontsize = 16)
        #axis.set_ylabel('Percent', labelpad=2, fontsize = 14)
        axis.bar(gpu_labels, gpu_snapshot, width =0.5, color = 'limegreen',align='center', edgecolor='none')
        axis.set_title('GPU Utilization', fontsize = 18)

def plot_gpu_memory(axis, gpu_labels, gpu_mem):
        axis.cla()
        axis.grid(False)
        axis.set_ylim([0,100])
        axis.set_xticks(gpu_labels)
        axis.set_xlabel('gpu ID', labelpad=2, fontsize = 16)
        #axis.set_ylabel('Percent', labelpad=2, fontsize = 14)
        axis.bar(gpu_labels, gpu_mem, width =0.5, color = 'red',align='center', edgecolor='none')
        axis.set_title('GPU Memory usage', fontsize = 18)

def plot_mem(axis,  snapshot):
        axis.cla()
        axis.grid(False)
        axis.set_ylim([0,100])
        axis.set_ylabel('Percent', labelpad=2, fontsize = 16)
        #axis.set_xlabel('Memory', labelpad=2, fontsize = 16)
        axis.bar(['Memory'] ,snapshot, color='dodgerblue', edgecolor='none')
        axis.set_title('Memory Utilization', fontsize = 18)
###
def RunAnimation():
        sys.stdout.flush()
        gpu=False
        if torch.cuda.is_available():
            gpu=True
            import py3nvml
            maxNGPUS = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
            py3nvml.py3nvml.nvmlInit()
            deviceCount = py3nvml.py3nvml.nvmlDeviceGetCount()
        fig = pl.figure(figsize = (16,4))
        pl.rcParams['xtick.labelsize'] = 12
        pl.rcParams['ytick.labelsize'] = 12
        if gpu:
            gs = gridspec.GridSpec(1, 4)
            ax3 = pl.subplot(gs[0,2])
            ax4 = pl.subplot(gs[0,3])
        else:
           gs = gridspec.GridSpec(1, 4)
        ax1 = pl.subplot(gs[0,0])
        ax2 = pl.subplot(gs[0,1])
        pl.tight_layout()
        pl.gcf().subplots_adjust(bottom=0.2)
        from matplotlib.colors import ListedColormap
        cm = ListedColormap(sns.color_palette("RdYlGn", 10).as_hex())
        os.system("mkdir -p images")
        i=0
        while(True):
            sys.stdout.flush()
            #cpu
            cpu_usage = psutil.cpu_percent(percpu=True)
            cpu_labels = range(1,len(cpu_usage)+1)
            plot_cpu_perf(ax1, cpu_labels, cpu_usage)
            plot_mem(ax2,[psutil.virtual_memory()[2]])

            #gpu
            if gpu:
                gpu_snapshot = []
                gpu_mem=[]
                gpu_labels = list(range(1,deviceCount+1))
                import py3nvml
                for j in range(deviceCount):
                    handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(j)
                    util = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_snapshot.append(util.gpu)
                    mem_info = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem.append(round(100*(mem_info.used)/mem_info.total,1))
                gpu_snapshot = gpu_snapshot
                plot_gpu_perf(ax3, gpu_labels, gpu_snapshot)
                plot_gpu_memory(ax4, gpu_labels, gpu_mem)
            #update the graphics
            display.display(pl.gcf())
            display.clear_output(wait=True)
            time.sleep(0.005)
            i=i+1










def set_device(with_cuda=True, cuda_devices= None):
        '''
        cuda_devices should be list of GPU IDs to use such as "0,1,2,3,4,5,6,7"
        cuda_devices= all use all gpus
        '''
        if not torch.cuda.is_available():
            print('Gpus are not avialable....Using Cpu as advice')
            device =torch.device('cpu')
            return device , None
        if torch.cuda.is_available() and with_cuda:
            if (not cuda_devices) or (torch.cuda.device_count()==1) :
                print("Using one GPU")
                device = torch.device("cuda:0")
                return device , [0]
            elif cuda_devices:
                if cuda_devices =='all':
                    print("Using all the avialable %d GPUS" % torch.cuda.device_count())
                    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    device_ids=[d for d in range(torch.cuda.device_count())]
                else:
                    device_ids=[int(d) for d in cuda_devices.split(',')]
                    NRgpus= len(device_ids)
                    assert NRgpus >0 and NRgpus <= torch.cuda.device_count(), 'Number of gpus shoulb be in the range 0- total Gpus'
                device =torch.device(f'cuda:{device_ids[0]}')
                print('Using cuda with the following master device device', device)
                #torch.cuda.set_device(device)
                print('devices list:',device_ids )
                return device, device_ids
def get_gpus_memory():
    CRED    = '\33[31m'
    CEND      = '\33[0m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBOLD     = '\33[1m'
    out =CBOLD+""
    import py3nvml
    py3nvml.py3nvml.nvmlInit()
    numDevices = py3nvml.py3nvml.nvmlDeviceGetCount()
    for i in range(numDevices):
        handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(i)
        util = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util=round(util.gpu,2)
        info = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(handle)
        toGB=(1024*1024*1024)
        used = round(info.used/toGB ,1)
        total=round( info.total/toGB,1)
        frac=round(100*(info.free)/info.total,2)
        if frac > 75 :
            out = out+ CGREEN
        elif frac > 50:
            out = out+ CYELLOW
        else:
            out = out+ CRED
        out = out +"GPU {:2}: ".format(i) +  \
        "{}\t".format(py3nvml.py3nvml.nvmlDeviceGetName(handle)) +\
        "utilization:{:>2}% \t".format (gpu_util) +\
        "Free Memory: {:>2} % ".format(frac) + \
        "(used {}/{} GB)\n".format(used, total)
    print(out+CEND)


class Task_trainer():
    def __init__(self,
                 model= None ,
                 mutitasks=True,
                 out_dir='./out',
                 device=None,
                 device_ids=None,
                 nr_epoch= 1,
                 current_epoch=0,
                 vocab=None,
                 pad_index=0,
                 criterion=None,
                 optimizer=None,
                 lr= 1e-4,
                 scheduler=None,
                 SummaryWriter=None,
                 eval_every_batch= np.inf,
                 print_logs_every=1 ,
                 eval_every_epoch=1,
                 log_freq= 10,
                 clip=1,
                 verbose=1,
                 early_stopping =22
                 ):
        super().__init__()
        self.model = model
        self.mutitasks=mutitasks
        self.device=device
        self.device_ids=device_ids
        self.nr_epoch= nr_epoch
        self.out_dir=out_dir
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.pad_index=pad_index
        self.vocab=vocab
        if self.vocab:
            self.pad_index=vocab.dict_tok2int['<pad>']
            self.vocab_size= len(self.vocab)
        self.lr=lr
        self.optimizer=optimizer
        if self.model and not self.optimizer:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler=scheduler
        if self.model and not self.scheduler:
            self.scheduler= ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.SummaryWriter=SummaryWriter
        
        #if self.model and not self.SummaryWriter: self.SummaryWriter= tensorboard.SummaryWriter(log_dir=self.out_dir+'/run')
        self.eval_every_epoch =eval_every_epoch
        self.eval_every_batch= eval_every_batch
        self.print_logs_every= print_logs_every
        self.log_freq = log_freq
        self.clip=clip
        self.current_epoch =current_epoch
        self.best_epoch= 0
        self.best_loss = None
        self.current_loss=None
        # dict with epoches as keys
        self.eval_losses= {}
        self.eval_metrics={}
        self.avg_train_losses={}
        self.avg_train_metrics={}
        self.uncertainties={}
        self.early_stopping= early_stopping
        self.verbose=verbose
    ###############################################################
    def train_one_batch(self, batch , train=True):
        batch = batch.to(self.device)
        node_num = batch.sph.shape[-1]
        batch.sph = batch.sph.reshape(-1, node_num, node_num)
        if batch.edge_attr is None:
            batch.edge_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        # get train loss
        if train:
            self.optimizer.zero_grad()
            self.model.train()
            #As we want our model to predict the <eos> token but not have it be an input into our model
            # we simply slice the <eos> token off the end of the sequence. this is done inside the model
            out = self.model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, batch.sph)
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, batch.sph)
        losses=dict()
        metrics=dict()
        if hasattr(self.model, 'module'): model= self.model.module
        else: model= self.model
        #Batch_size= tgt.size()[1]
        total_loss=torch.tensor(0.).to(self.device)    
        if model.tasks:
            for task in model.tasks:
                if len(batch[task.task_name][~torch.isnan(batch[task.task_name])]) >2:
                    loss , met= task.loss_fn(out[task.task_name] ,batch[task.task_name])
                    losses[task.task_name]= loss#.type(total_loss.dtype)
                    coef=1.0
                    if task.__class__.__name__ == 'regression_head':
                        metrics[f'{task.task_name}_R2']= met
                        coef= 1./(2.)
                    elif task.__class__.__name__ == 'Classification_head':
                        metrics[f'{task.task_name}_ACC']= met
                    else:
                        metrics[f'{task.task_name}']= met
                    if self.mutitasks:
                        total_loss += (coef*torch.exp(-task.uncertainty) *losses[task.task_name] +0.5* task.uncertainty).mean()
                    else:
                        total_loss += losses[task.task_name].mean()
        #total_loss=losses['Loss']
        if train:
            total_loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            unc={}
            if model.tasks:
                for task in model.tasks: unc[task.task_name]=torch.exp(task.uncertainty).item()
            self.uncertainties[self.current_epoch-1]=unc
        losses['Loss']=total_loss
        losses  = {key: value.mean().item() for key, value in losses.items()}
        return losses , metrics
    
    ###############################################################
    def train_batch_no_backward(self, batch , train=True):
        batch = batch.to(self.device)
        node_num = batch.sph.shape[-1]
        batch.sph = batch.sph.reshape(-1, node_num, node_num)
        if batch.edge_attr is None:
            batch.edge_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        # get train loss
        if train:
            self.optimizer.zero_grad()
            self.model.train()
            #As we want our model to predict the <eos> token but not have it be an input into our model
            # we simply slice the <eos> token off the end of the sequence. this is done inside the model
            out = self.model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, batch.sph)
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, batch.sph)
        losses=dict()
        metrics=dict()
        if hasattr(self.model, 'module'): model= self.model.module
        else: model= self.model
        #Batch_size= tgt.size()[1]
        total_loss=torch.tensor(0.).to(self.device)    
        if model.tasks:
            for task in model.tasks:
                if len(batch[task.task_name][~torch.isnan(batch[task.task_name])]) >2:
                    loss , met= task.loss_fn(out[task.task_name] ,batch[task.task_name])
                    losses[task.task_name]= loss#.type(total_loss.dtype)
                    coef=1.0
                    if task.__class__.__name__ == 'regression_head':
                        metrics[f'{task.task_name}_R2']= met
                        coef= 1./(2.)
                    elif task.__class__.__name__ == 'Classification_head':
                        metrics[f'{task.task_name}_ACC']= met
                    else:
                        metrics[f'{task.task_name}']= met
        
                    if self.mutitasks:
                        total_loss += (coef*torch.exp(-task.uncertainty) *losses[task.task_name] +0.5* task.uncertainty).mean()
                    else:
                        total_loss += losses[task.task_name]#.mean()
        #total_loss=losses['Loss']
        if train:
            total_loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            unc={}
            if model.tasks:
                for task in model.tasks: unc[task.task_name]=torch.exp(task.uncertainty).item()
            self.uncertainties[self.current_epoch-1]=unc
        losses['Loss']=total_loss
        losses  = {key: value.mean().item() for key, value in losses.items()}
        return losses , metrics
        
    ###############################################################
    def train_one_epoch(self, epoch, train_dataloader, test_dataloader):
        all_losses= dict()
        all_metrics= dict()
        number_batch=len(train_dataloader)
        if self.eval_every_batch > number_batch: self.eval_every_batch =number_batch
        leave = False
        if self.verbose:leave = True

        data_iter = tqdm(enumerate(train_dataloader),
                          desc="Train_Epoch:%d:Batch" % epoch, total=number_batch,position=0, leave=leave )
        for b, batch in data_iter:
            losses, metrics =self.train_one_batch(batch , train=True)
            for k in losses:
                if k in all_losses: all_losses[k].append(losses[k])
                else: all_losses[k]= []
            for k in metrics:
                if k in all_metrics: all_metrics[k].append(metrics[k])
                else: all_metrics[k]= []
            if  b%self.log_freq==0 and self.SummaryWriter:
                self.SummaryWriter.add_scalars('All_Batch_Train/Loss', losses, b+ epoch*number_batch ,walltime =True)
                self.SummaryWriter.add_scalars('All_Batch_Train/metrics', metrics, b+ epoch*number_batch ,walltime =True)
                self.SummaryWriter.add_scalars('All_Batch_Train/uncertainties', self.uncertainties[self.current_epoch-1], b+ epoch*number_batch ,walltime =True)
            if (b%self.eval_every_batch==0) and b>0 and  b < (number_batch-1):
                self.avg_train_losses[self.current_epoch-1]= {k:np.mean(all_losses[k]) for k in all_losses}
                self.avg_train_metrics[self.current_epoch-1]= {k:np.mean(all_metrics[k]) for k in all_metrics}
                self.evaluate(test_dataloader)
                if self.verbose:
                    self.print_logs()
        if not self.verbose:
            data_iter.close()
        self.avg_train_losses[self.current_epoch-1]= {k:np.mean(all_losses[k]) for k in all_losses}
        self.avg_train_metrics[self.current_epoch-1]= {k:np.mean(all_metrics[k]) for k in all_metrics}
        if self.SummaryWriter:
            self.SummaryWriter.add_scalars('Epoch_Train/Loss', self.avg_train_losses[self.current_epoch-1], epoch ,walltime =True)
            self.SummaryWriter.add_scalars('Epoch_Train/metrics', self.avg_train_metrics[self.current_epoch-1], epoch ,walltime =True)
            self.SummaryWriter.add_scalars('Epoch_Train/uncertainties', self.uncertainties[self.current_epoch-1], epoch ,walltime =True)
    ###############################################################
    def print_logs(self):
        e=self.current_epoch-1
        print(f'\33[1m Losses and metrics of epoch {e+1}: \33[0m')
        for k in self.uncertainties[e]:
            if f'{k}_ACC' in self.eval_metrics[e]:
                mn='ACC'
                m=self.eval_metrics[e][f'{k}_ACC']
            elif f'{k}_R2' in self.eval_metrics[e]:
                mn='R2'
                m=self.eval_metrics[e][f'{k}_R2']
            else:
                mn=''
                m=self.eval_metrics[e][k]
            print(f'\33[1m{k}\33[0m: Pre-fit train-loss={self.avg_train_losses[e][k]:.7f} |Eval-loss={self.eval_losses[e][k]:.7f} |Eval metric {mn}={m:.4f} |uncertainty={self.uncertainties[e][k]:.4f}')
    ###############################################################
    def evaluate(self, test_dataloader):
        self.model.eval()
        all_losses= dict()
        all_metrics= dict()
        for  batch in test_dataloader:
            losses, metrics = self.train_one_batch(batch , train=False)
            for k in losses:
                if k not in all_losses: all_losses[k]= []
                all_losses[k].append(losses[k])
            for k in metrics:
                if k not in all_metrics: all_metrics[k]= []
                all_metrics[k].append(metrics[k])
        av_losses={k:np.mean(all_losses[k]) for k in all_losses}
        av_metrics={k:np.mean(all_metrics[k]) for k in all_metrics}
        self.eval_losses[self.current_epoch-1] =av_losses
        self.eval_metrics[self.current_epoch-1]=av_metrics
        self.current_loss=self.eval_losses[self.current_epoch-1]['Loss']
        # Save the model if the validation loss is the best we've seen so far.
        if not self.best_loss or self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            if self.verbose:
                print(f'Saving the checkpoint and the best model which is found at epoch {self.current_epoch}...')
            self.save_model( model_file=f'{self.out_dir}/best_model.pkl')
            self.best_epoch = self.current_epoch
            self.save_checkpoint(checkpoint_file=f'{self.out_dir}/checkpoint_best_model.pt',epoch=self.current_epoch)
            #self.print_logs()
        if self.best_loss and self.scheduler:
            self.scheduler.step(self.best_loss)
    ###############################################################
    def fit(self, train_dataloader,test_dataloader , nr_epochs=None ,
            eval_every_epoch=None,eval_every_batch=None, lr=None , nr_gpus=None):
        '''
         nr_epochs, eval_every_batch, eval_every_epoch override the defualt one
        '''
        if lr:
            for g in self.optimizer.param_groups:
                g['lr'] = lr
        if  nr_epochs: self.nr_epoch =nr_epochs
        if eval_every_batch: self.eval_every_batch= eval_every_batch
        else: self.eval_every_batch= np.inf
        if eval_every_epoch: self.eval_every_epoch= eval_every_epoch
        if self.print_logs_every < self.eval_every_epoch:
            print('setting eval_every_epoch to eval_every_epoch')
            self.print_logs_every = self.eval_every_epoch
        if nr_gpus:  self.set_devices(nr_gpus)
        start_time = time.time()
        current_time= start_time
        print('Number of training batchs is',len(train_dataloader))
        print(f'saving the outputs in folder {self.out_dir}')
        print(f'number of training epochs = {self.nr_epoch}')
        early_stopping=0
        for e in range(0,  self.nr_epoch):
            self.current_epoch +=1
            #if self.verbose:print(f'#######################   Epoch {self.current_epoch}    #########################')
            self.train_one_epoch(epoch=self.current_epoch,train_dataloader=train_dataloader,test_dataloader=test_dataloader)
            if (e%self.eval_every_epoch==0 or e ==( self.nr_epoch-1)):
                self.evaluate(test_dataloader)
                if self.SummaryWriter:
                    self.SummaryWriter.add_scalars('Epoch_Eval/Loss', self.eval_losses[self.current_epoch-1], self.current_epoch ,walltime =True)
                    self.SummaryWriter.add_scalars('Epoch_Eval/metrics', self.eval_metrics[self.current_epoch-1], self.current_epoch ,walltime =True)
                    self.SummaryWriter.add_scalars('Epoch_Eval/uncertainties', self.uncertainties[self.current_epoch-1], self.current_epoch ,walltime =True)
            if (e%self.print_logs_every==0 or e ==( self.nr_epoch-1)) and  self.verbose:
                self.print_logs()
            self.save_checkpoint(checkpoint_file=f'{self.out_dir}/checkpoint_last_epoch.pt',epoch=self.current_epoch)
            time_per_epoch= (time.time() - current_time)/60
            if self.verbose > 1:
                print(f'checkpoint saved after epoch {self.current_epoch }...')
                print('--------------------------------------------------------------')
                print(f'Time per epoch {self.current_epoch}= {time_per_epoch:.2f} min ')
                print(f'Best loss={self.best_loss:.7f}')
                current_time= time.time()
            if  (self.current_epoch -self.best_epoch) > self.early_stopping:
                print(f'No improvment after {self.early_stopping} ..stopping...')
                break
        #if not  self.verbose:self.print_logs()
        print(f'Total Training Time: {((time.time() - start_time)/60):.2f} min' )
        if self.SummaryWriter:
            self.SummaryWriter.close()
        torch.cuda.empty_cache()
        print(f'Best model is found at epoch {self.best_epoch}')
    ###############################################################
    def continue_training(self, train_dataloader,test_dataloader ,
                          checkpoint_file=None ,nr_epochs=None ,eval_every_batch=None):
        if not checkpoint_file:
            checkpoint_file= f'{self.out_dir}/checkpoint_last_epoch.pt'
        print(f'loading checkpoint from file: {checkpoint_file}')
        self.load_checkpoint(self, checkpoint_file)
        self.fit(train_dataloader=train_dataloader,test_dataloader=test_dataloader ,
                 nr_epochs=nr_epochs ,eval_every_batch=eval_every_batch)
    ###############################################################
    def set_devices(self,nr_gpus):
        if hasattr(self.model, 'module'):
            self.model=self.model.module
        device_ids=None
        grab_gpus(num_gpus=nr_gpus)
        cuda_devices=os.environ["CUDA_VISIBLE_DEVICES"]
        self.device_ids=[int(d) for d in cuda_devices.split(',')]
        self.device = torch.device(f'cuda:{self.device_ids[0]}')
        print('Using cuda with the following master device device', self.device)
        #torch.cuda.set_device(self.device)
        self.model=self.model.to(self.device)
        if self.device_ids and len(self.device_ids) > 1:
            print("Using", len(self.device_ids) , "GPUs for DataParallel")
            self.model = nn.DataParallel(self.model ,device_ids=self.device_ids , dim=1)
    ###############################################################
    def save_model(self, model_file='best_model.pkl' ):
        out={}
        if hasattr(self.model, 'module'): mdel2save= self.model.module
        else:   mdel2save= self.model
        out['model_state_dict']=mdel2save.state_dict()
        out['model']=mdel2save
        torch.save(out, model_file)
    ###############################################################
    def save_checkpoint(self, checkpoint_file='checkpoint.pt', epoch=None):
        out={}
        for att in self.__dict__:
            if att not in ['SummaryWriter']:
                out[att]= getattr(self, att)#tr.f'{att}'
        if hasattr(self.model, 'module'): mdel2save= self.model.module
        else: mdel2save= self.model
        out['model_state_dict']= mdel2save.state_dict()
        if self.optimizer:
            out['optimizer_state_dict']=self.optimizer.state_dict()
        if self.scheduler:
            out['scheduler_state_dict']=self.scheduler.state_dict()
        if self.vocab:
                out['vocab']=self.vocab
        torch.save(out, checkpoint_file)
    ###############################################################
    def load_checkpoint(self, checkpoint_file=None , best_model=False):
        '''
        best_model: load checkpoint of best model otherwise checkpoint_last_epoch
        '''
        if not checkpoint_file:
            if best_model: checkpoint_file= f'{self.out_dir}/checkpoint_best_model.pt'
            else:  checkpoint_file= f'{self.out_dir}/checkpoint_last_epoch.pt'
        print(f'loading checkpoint from file: {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        for k in checkpoint:
            if hasattr(self, k) and k not in ['model', 'SummaryWriter','optimizer','scheduler']:
                setattr(self,k, checkpoint[k])
        if not self.model:
            print('Model structure is loaded from the checkpoint')
            setattr(self,'model', checkpoint['model'])
        if hasattr(self.model, 'module'):
            self.model=self.model.module
        print('Model parameters are loaded from the checkpoint')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint :
            print('Optimizer parameters are loaded from the checkpoint')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            print('Scheduler parameters are loaded from the checkpoint')
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

###################################################################

############################################################################
def save_model(model, model_file='best_model.pkl' ):
    out={}
    if hasattr(model, 'module'):
        mdel2save= model.module
    else:
        mdel2save= model
    out['model_state_dict']=mdel2save.state_dict()
    out['model']=mdel2save
    torch.save(out, model_file)
############################################################################
def load_model(model_file='best_model.pkl'):
    print(f'loading checkpoint from file: {model_file}')
    checkpoint = torch.load(model_file)
    model= checkpoint['model']
    print('The model class is loaded...')
    if hasattr(model, 'module'):
        print('model is data parallel...')# should not happen
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    print('Model parameters are loaded from the saved model_state_dict...\nDone!')
    return model
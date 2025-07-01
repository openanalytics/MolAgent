import torch
from torch import optim, nn, utils, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
import lightning as L
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import regression, MetricCollection, functional, classification
from building_elements_encoders import  mol_atomstic_encoder, mol_residues_level_encoder, mol_residues_light_level_encoder,mol_light_atomstic_encoder
from interaction_encoders import InteractionRelativeGraphEncoder, FullParameterEmbedding



class RayLightningTrainer(L.LightningModule):
    def __init__(self,lr=1e-4):
        super(RayLightningTrainer, self).__init__()        
        self.lr=lr
        
    def forward(self, batch):
        pass
    
    def loss_fn(self,out,y_true):
        pass
    
    def additional_training_logs(self,out,y_true):
        pass
        
    def retrieve_y_true(self,batch):
        pass
    
    def retrieve_y_pred(self,out):
        pass
    
    def training_step(self, batch, batch_idx):
        out=self.forward(batch)
        y_true=self.retrieve_y_true(batch)
        y_pred=self.retrieve_y_pred(out)
        loss = self.loss_fn(out,y_true)
        self.log("train_loss", loss, prog_bar=True)
        self.additional_training_logs(out,y_true)
        
        output = self.train_metrics(y_pred, y_true['values'])
        self.log_dict(output)
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        out=self.forward(batch)
        y_true=self.retrieve_y_true(batch)
        y_pred=self.retrieve_y_pred(out)
        loss = self.loss_fn(out,y_true)
        self.log("val_loss", loss, prog_bar=True)

        self.valid_metrics.update(y_pred, y_true['values'])

    def on_validation_epoch_end(self):
        # use log_dict instead of log
        # metrics are logged with keys: val_Accuracy, val_Precision and val_Recall
        output = self.valid_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        # remember to reset metrics at the end of the epoch
        self.valid_metrics.reset()    
    
    def test_step(self, batch, batch_idx):
        out=self.forward(batch)
        y_true=self.retrieve_y_true(batch)
        y_pred=self.retrieve_y_pred(out)
        loss = self.loss_fn(out,y_true)
        self.log("test_loss", loss, prog_bar=True)
        self.test_metrics.update(y_pred, y_true['values'])
        
    def on_test_epoch_end(self):
        # use log_dict instead of log
        # metrics are logged with keys: val_Accuracy, val_Precision and val_Recall
        output = self.test_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False)
        # remember to reset metrics at the end of the epoch
        self.test_metrics.reset()   

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Configure optimizer and learning rate scheduler
        o = AdamW(self.parameters(), lr=self.lr, weight_decay=2e-4)
        s = CyclicLR(o, self.lr,self.lr*10, 1000, mode='triangular', cycle_momentum=False)
        return {'optimizer': o,
                'lr_scheduler': {'scheduler': s, 'interval': 'epoch', 'name': 'Training/LearningRate'}}

            
class MHARayRelativeDecoderhead(RayLightningTrainer):
    def __init__(self,config,emb_dim):
        super(MHARayRelativeDecoderhead, self).__init__(lr=config['lr'])
        ligand_model=mol_atomstic_encoder( hidden_dim=100, sasa_hidden_dim=10,  
                         mpnn='GINE',
                         pe_origin_dim=20,pe_dim=20 ,
                         gamma=0.5, slope=0.0, hops_list=[2.,5.,10.,15.],
                         nhead=4,n_hop=5,num_layers=10, 
                         dropout=0.1, drop_prob=0.0, attn_dropout=0.1
                        )

        protein_model=mol_residues_light_level_encoder( hidden_dim=100, sasa_hidden_dim=10,
                         emb_add=True, ## add the emb for pe and rotamers to (residues,sasa) and res_res_BondEncoder+ Bakbone_dihs_enc
                         rotamers_hidden_dim=20,   pe_dim=20 ,# ignore with emb_add
                         mpnn='GIN',#mpnn='GINE'
                         gamma=0.5, slope=0.0, hops_list=[5.,7.,8.,10.], #<hops between 5 and 10
                         nhead=4,n_hop=5,num_layers=10, 
                         dropout=0.1, drop_prob=0.0, attn_dropout=0.1
                        )
        param_encoder=FullParameterEmbedding()

        interaction_encoder=mol_light_atomstic_encoder( hidden_dim=132, edge_dim=param_encoder.emb_dim*len(param_encoder.used_indices),  
                         mpnn='GINE',
                         pe_origin_dim=20,pe_dim=20 , sasa_in=2,
                         gamma=0.5, slope=0.0, hops_list=[1.,2.,3.,1.],
                         nhead=4,n_hop=5,num_layers=10, 
                         dropout=0.1, drop_prob=0.0, attn_dropout=0.1
                        )
        self.model=InteractionRelativeGraphEncoder( 100,protein_model,ligand_model,param_encoder,interaction_encoder,pre_transform=pre_transform, inc_padding_val=-510)
        self.emb_dim=emb_dim
        self.start=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.end=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.linear1=torch.nn.Linear(emb_dim,3)        
        decoder_layer= torch.nn.TransformerDecoderLayer(d_model=emb_dim,nhead=config['nhead'],batch_first=True)
        self.transformer_decoder=torch.nn.TransformerDecoder(decoder_layer,num_layers=config['n_layers'])
        self.mse_loss = torch.nn.MSELoss()
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.loss_weight=config['loss_w']        
        self.mse_loss = torch.nn.MSELoss()
        metrics = MetricCollection({'R2': regression.R2Score(),
                 'MAE': regression.MeanAbsoluteError(),
                 'Pearson': regression.PearsonCorrCoef(),
                 'MAPE': regression.MeanAbsolutePercentageError(),
                 'MSE': regression.MeanSquaredError(),
                 'WMAPE': regression.WeightedMeanAbsolutePercentageError()}
            )
        train_metrics = MetricCollection({
                 'MAE': regression.MeanAbsoluteError(),
                 'Pearson': regression.PearsonCorrCoef(),
                 'MAPE': regression.MeanAbsolutePercentageError(),
                 'MSE': regression.MeanSquaredError(),
                 'WMAPE': regression.WeightedMeanAbsolutePercentageError()}
            ,prefix='train_')
        self.acc = classification.BinaryAccuracy()
        self.AUROC = classification.AUROC(task="binary")
        self.train_metrics = train_metrics
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()
        
    def forward(self, batch):
        out1,mask1,out2,mask2=self.model(batch)
        sequences_tgt=[ torch.vstack([self.start,out1[b,:,:][mask1[b]],self.end]) for b in range(out1.shape[0])]
        sequences_mem=[ torch.vstack([self.start,out2[b,:,:][mask2[b]],self.end]) for b in range(out2.shape[0])]
        tgt=torch.nn.utils.rnn.pad_sequence(sequences_tgt, batch_first=True, padding_value=0.0)
        memory=torch.nn.utils.rnn.pad_sequence(sequences_mem, batch_first=True, padding_value=0.0)
        return self.linear1(self.transformer_decoder(tgt,memory)[:,0,:])    
    
    def loss_fn(self,out,y_true):
        y_pred=out[:,0]
        y_prob=out[:,1:]
        y_true_val=y_true['values']
        y_true_labels=y_true['labels']
        mse_loss = self.mse_loss(y_pred,y_true_val)
        label_loss = self.entropy_loss(y_prob,y_true_labels)
        loss=(self.loss_weight*mse_loss + (1-self.loss_weight)*label_loss)
        return loss
        
    def retrieve_y_true(self,batch):
        y_true=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity
        y_label=(batch['ligands_one'].ligand_affinity>batch['ligands_two'].ligand_affinity).long()
        return {'values':y_true,'labels':y_label}
    
    def retrieve_y_pred(self,out):
        return out[:,0]
        
    def additional_training_logs(self,out,y_true):
        train_acc=self.acc(out[:,1:].argmax(dim=1),y_true['labels'])
        train_auc=self.AUROC(out[:,2],y_true['labels'])
        self.log("train_accuracy", train_acc, prog_bar=False)
        self.log("train_auc", train_auc, prog_bar=False)


    

    


import torch
from torch import optim, nn, utils, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
import lightning as L
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import regression, MetricCollection, functional, classification



class LightningTrainer(L.LightningModule):
    def __init__(self,lr=1e-4):
        super(LightningTrainer, self).__init__()        
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
        
        if y_true['values'].shape[0]>1:
            output = self.train_metrics(y_pred, y_true['values'])
            self.log_dict(output)
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        out=self.forward(batch)
        y_true=self.retrieve_y_true(batch)
        y_pred=self.retrieve_y_pred(out)
        loss = self.loss_fn(out,y_true)
        self.log("val_loss", loss, prog_bar=True)
        if hasattr(self,'mse_loss'):
            val_mse_loss = self.mse_loss(y_pred,y_true['values'])
            self.log("val_mse_loss", val_mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        if y_true['values'].shape[0]>1:
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
        s = CyclicLR(o, base_lr=self.lr,max_lr=self.lr*15, step_size_up=5, mode='triangular2', cycle_momentum=False)
        return {'optimizer': o,
                'lr_scheduler': {'scheduler': s, 'interval': 'epoch', 'name': 'Training/LearningRate'}}
    
###############################################################################
class MHARelativehead(LightningTrainer):
    def __init__(self,model,emb_dim,nhead=4,n_layers=3,loss_w=0.66,lr=1e-4,pred_hidden_dropout=0.1,noise_lvl=0.3):
        super(MHARelativehead, self).__init__(lr=lr)
        self.model=model
        self.emb_dim=emb_dim
        self.start=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.split_graphs=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.end=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.linear1=torch.nn.Linear(emb_dim,3)
        encoder_layer= torch.nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_encoder=torch.nn.TransformerEncoder(encoder_layer,num_layers=n_layers)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(pred_hidden_dropout)
        self.mse_loss = torch.nn.MSELoss()
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.loss_weight=loss_w      
        self.noise_lvl=noise_lvl  
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
        self.acc = classification.MulticlassAccuracy(num_classes=3)
        self.AUROC = classification.MulticlassAUROC(num_classes=3)
        self.train_metrics = train_metrics
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()
        
    def forward(self, batch):
        out1,mask1,out2,mask2=self.model(batch)
        sequences=[ torch.vstack([self.start,out1[b,:,:][mask1[b]],self.split_graphs,out2[b,:,:][mask2[b]],self.end]) for b in range(out1.shape[0])]
        hidden=torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
        return self.linear1(self.dropout(self.activation(self.transformer_encoder(hidden)[:,0,:])))
        #return self.linear1(self.dropout(self.activation(torch.mean(self.transformer_encoder(hidden),dim=1))))

    def loss_fn(self,out,y_true):
        y_pred=out[:,0]
        y_prob=out[:,1:]
        y_true_val=y_true['values']
        y_true_labels=y_true['labels']
        
        #no loss from within noise lvl
        #y_true_val[y_true['noise_mask']]=y_pred[y_true['noise_mask']]
        #no loss for predictions within 0.3 
        prediction_mask=torch.abs(y_true_val-y_pred)<self.noise_lvl
        #y_true_val[prediction_mask]=y_pred[prediction_mask]
        
        mse_loss = self.mse_loss(y_pred,y_true_val)
        label_loss = self.entropy_loss(y_prob,y_true_labels)
        
        loss_weights=self.loss_weight*~(prediction_mask | y_true['noise_mask'])
        #loss=( (1./2.)*torch.exp(-self.uncertainty_reg)*mse_loss  +0.5*self.uncertainty_reg + torch.exp(-self.uncertainty_clf)*label_loss+ +0.5*self.uncertainty_clf)
        loss=(loss_weights*mse_loss + (1-loss_weights)*label_loss).mean()
        return loss
        
    def retrieve_y_true(self,batch):
        y_true=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity
        y_label=(batch['ligands_one'].ligand_affinity>batch['ligands_two'].ligand_affinity).long()
        mask0=torch.abs(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)<=self.clf_lvl
        mask1=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity>self.clf_lvl
        mask2=-(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)>self.clf_lvl
        y_label[mask0]=0
        y_label[mask1]=1
        y_label[mask2]=2
        return {'values':y_true,'labels':y_label,'noise_mask':torch.abs(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)<self.noise_lvl}
    
    def retrieve_y_pred(self,out):
        return out[:,0]
        
    def additional_training_logs(self,out,y_true):
        train_acc=self.acc(out[:,1:].argmax(dim=1),y_true['labels'])
        train_auc=self.AUROC(out[:,1:],y_true['labels'])
        self.log("train_accuracy", train_acc, prog_bar=False)
        self.log("train_auc", train_auc, prog_bar=False)
            
            
###############################################################################        
class MHARelativePharmaRegressionhead(LightningTrainer):
    def __init__(self,model,hidden_dim=39,pred_hidden_dropout=0.1, lr=1e-4,loss_w=0.66,noise_lvl=0.3,clf_lvl=0.6):
        super(MHARelativePharmaRegressionhead, self).__init__(lr=lr)
        self.model=model
        self.linear=torch.nn.Linear(hidden_dim, 3)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(pred_hidden_dropout)
        self.loss_weight=loss_w
        self.noise_lvl=noise_lvl
        self.clf_lvl=clf_lvl
        self.mse_loss = torch.nn.MSELoss()
        self.entropy_loss = torch.nn.CrossEntropyLoss()
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
        self.acc = classification.MulticlassAccuracy(num_classes=3)
        self.AUROC = classification.MulticlassAUROC(num_classes=3)
        self.uncertainty_reg  = torch.nn.Parameter(torch.zeros(1))
        self.uncertainty_clf  = torch.nn.Parameter(torch.zeros(1))
        self.train_metrics = train_metrics
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()
        
    def forward(self, batch):
        out=self.model(batch)
        return self.linear(self.dropout(self.activation(out)))
    
    def loss_fn(self,out,y_true):
        y_pred=out[:,0]
        y_prob=out[:,1:]
        y_true_val=y_true['values']
        y_true_labels=y_true['labels']
        
        #no loss from within noise lvl
        #y_true_val[y_true['noise_mask']]=y_pred[y_true['noise_mask']]
        #no loss for predictions within 0.3 
        prediction_mask=torch.abs(y_true_val-y_pred)<self.noise_lvl
        #y_true_val[prediction_mask]=y_pred[prediction_mask]
        
        mse_loss = self.mse_loss(y_pred,y_true_val)
        label_loss = self.entropy_loss(y_prob,y_true_labels)
        
        #loss_weights=self.loss_weight*~(prediction_mask | y_true['noise_mask'])
        loss_weights=self.loss_weight*~(y_true['noise_mask'])
        #loss=( (1./2.)*torch.exp(-self.uncertainty_reg)*mse_loss  +0.5*self.uncertainty_reg + torch.exp(-self.uncertainty_clf)*label_loss+ +0.5*self.uncertainty_clf)
        loss=(loss_weights*mse_loss + (1-loss_weights)*label_loss).mean()
        return loss
        
    def retrieve_y_true(self,batch):
        y_true=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity
        y_label=(batch['ligands_one'].ligand_affinity>batch['ligands_two'].ligand_affinity).long()
        mask0=torch.abs(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)<=self.clf_lvl
        mask1=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity>self.clf_lvl
        mask2=-(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)>self.clf_lvl
        y_label[mask0]=0
        y_label[mask1]=1
        y_label[mask2]=2
        return {'values':y_true,'labels':y_label,'noise_mask':torch.abs(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)<self.noise_lvl}
    
    def retrieve_y_pred(self,out):
        return out[:,0]
        
    def additional_training_logs(self,out,y_true):
        train_acc=self.acc(out[:,1:].argmax(dim=1),y_true['labels'])
        train_auc=self.AUROC(out[:,1:],y_true['labels'])
        self.log("train_accuracy", train_acc, prog_bar=False)
        self.log("train_auc", train_auc, prog_bar=False)
            
###############################################################################
class MHARelativeDecoderhead(LightningTrainer):
    def __init__(self,model,emb_dim,nhead=4,n_layers=3,loss_w=0.66,lr=1e-4,pred_hidden_dropout=0.1,noise_lvl=0.3,clf_lvl=0.6):
        super(MHARelativeDecoderhead, self).__init__(lr=lr)
        self.model=model
        self.emb_dim=emb_dim
        self.start=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.end=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.linear1=torch.nn.Linear(emb_dim,4)        
        decoder_layer= torch.nn.TransformerDecoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_decoder=torch.nn.TransformerDecoder(decoder_layer,num_layers=n_layers)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(pred_hidden_dropout)
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss_weight=loss_w
        self.noise_lvl=noise_lvl
        self.clf_lvl=clf_lvl
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
        self.acc = classification.MulticlassAccuracy(num_classes=3)
        self.AUROC = classification.MulticlassAUROC(num_classes=3)
        self.uncertainty_reg  = torch.nn.Parameter(torch.zeros(1))
        self.uncertainty_clf  = torch.nn.Parameter(torch.zeros(1))
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
        return self.linear1(self.dropout(self.activation(self.transformer_decoder(tgt,memory)[:,0,:])))
    
    def generate_features(self, batch):
        out1,mask1,out2,mask2=self.model(batch)
        sequences_tgt=[ torch.vstack([self.start,out1[b,:,:][mask1[b]],self.end]) for b in range(out1.shape[0])]
        sequences_mem=[ torch.vstack([self.start,out2[b,:,:][mask2[b]],self.end]) for b in range(out2.shape[0])]
        tgt=torch.nn.utils.rnn.pad_sequence(sequences_tgt, batch_first=True, padding_value=0.0)
        memory=torch.nn.utils.rnn.pad_sequence(sequences_mem, batch_first=True, padding_value=0.0)
        return self.transformer_decoder(tgt,memory)[:,0,:]
    
    def loss_fn(self,out,y_true):
        y_pred=out[:,0]
        y_prob=out[:,1:]
        y_true_val=y_true['values']
        y_true_labels=y_true['labels']
        
        #no loss from within noise lvl
        #y_true_val[y_true['noise_mask']]=y_pred[y_true['noise_mask']]
        #no loss for predictions within 0.3 
        prediction_mask=torch.abs(y_true_val-y_pred)<self.noise_lvl
        #y_true_val[prediction_mask]=y_pred[prediction_mask]
        
        mse_loss = self.mse_loss(y_pred,y_true_val)
        label_loss = self.entropy_loss(y_prob,y_true_labels)
        
        #loss_weights=self.loss_weight*~(prediction_mask | y_true['noise_mask'])
        loss_weights=self.loss_weight*~(y_true['noise_mask'])
        #loss=( (1./2.)*torch.exp(-self.uncertainty_reg)*mse_loss  +0.5*self.uncertainty_reg + torch.exp(-self.uncertainty_clf)*label_loss+ +0.5*self.uncertainty_clf)
        loss=(loss_weights*mse_loss + (1-loss_weights)*label_loss).mean()
        return loss
        
    def retrieve_y_true(self,batch):
        y_true=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity
        y_label=(batch['ligands_one'].ligand_affinity>batch['ligands_two'].ligand_affinity).long()
        mask0=torch.abs(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)<=self.clf_lvl
        mask1=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity>self.clf_lvl
        mask2=-(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)>self.clf_lvl
        y_label[mask0]=0
        y_label[mask1]=1
        y_label[mask2]=2
        return {'values':y_true,'labels':y_label,'noise_mask':torch.abs(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)<self.noise_lvl}
    
    def retrieve_y_pred(self,out):
        return out[:,0]
        
    def additional_training_logs(self,out,y_true):
        train_acc=self.acc(out[:,1:].argmax(dim=1),y_true['labels'])
        train_auc=self.AUROC(out[:,1:],y_true['labels'])
        self.log("train_accuracy", train_acc, prog_bar=False)
        self.log("train_auc", train_auc, prog_bar=False)
        
###############################################################################    
class LinearRelativehead(LightningTrainer):
    def __init__(self,model,emb_dim,loss_w=0.66,lr=1e-4,pred_hidden_dropout=0.1,noise_lvl=0.3,clf_lvl=0.6):
        super(LinearRelativehead, self).__init__(lr=lr)
        self.model=model
        self.emb_dim=emb_dim
        self.linear1=torch.nn.Linear(emb_dim,4)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(pred_hidden_dropout)
        self.mse_loss = torch.nn.MSELoss()
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.loss_weight=loss_w 
        self.noise_lvl=noise_lvl
        self.clf_lvl=clf_lvl
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
        self.acc = classification.MulticlassAccuracy(num_classes=3)
        self.AUROC = classification.MulticlassAUROC(num_classes=3)
        self.uncertainty_reg  = torch.nn.Parameter(torch.zeros(1))
        self.uncertainty_clf  = torch.nn.Parameter(torch.zeros(1))
        self.train_metrics = train_metrics
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()
        
    def forward(self, batch):
        out1,mask1,out2,mask2=self.model(batch)
        return self.linear1(self.dropout(self.activation(out1[:,0,:]-out2[:,0,:]))) 
    
    def generate_features(self, batch):
        out1,mask1,out2,mask2=self.model(batch)
        return out1[:,0,:]-out2[:,0,:]
    
    
    def loss_fn(self,out,y_true):
        y_pred=out[:,0]
        y_prob=out[:,1:]
        y_true_val=y_true['values']
        y_true_labels=y_true['labels']
        
        #no loss from within noise lvl
        #y_true_val[y_true['noise_mask']]=y_pred[y_true['noise_mask']]
        #no loss for predictions within 0.3 
        prediction_mask=torch.abs(y_true_val-y_pred)<self.noise_lvl
        #y_true_val[prediction_mask]=y_pred[prediction_mask]
        
        mse_loss = self.mse_loss(y_pred,y_true_val)
        label_loss = self.entropy_loss(y_prob,y_true_labels)
        
        #loss_weights=self.loss_weight*~(prediction_mask | y_true['noise_mask'])
        loss_weights=self.loss_weight*~(y_true['noise_mask'])
        #loss=( (1./2.)*torch.exp(-self.uncertainty_reg)*mse_loss  +0.5*self.uncertainty_reg + torch.exp(-self.uncertainty_clf)*label_loss+ +0.5*self.uncertainty_clf)
        loss=(loss_weights*mse_loss + (1-loss_weights)*label_loss).mean()
        return loss
        
    def retrieve_y_true(self,batch):
        y_true=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity
        y_label=(batch['ligands_one'].ligand_affinity>batch['ligands_two'].ligand_affinity).long()
        mask0=torch.abs(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)<self.clf_lvl
        mask1=batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity>self.clf_lvl
        mask2=-(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)>self.clf_lvl
        y_label[mask0]=0
        y_label[mask1]=1
        y_label[mask2]=2
        return {'values':y_true,'labels':y_label,'noise_mask':torch.abs(batch['ligands_one'].ligand_affinity-batch['ligands_two'].ligand_affinity)<self.noise_lvl}
    
    def retrieve_y_pred(self,out):
        return out[:,0]
        
    def additional_training_logs(self,out,y_true):
        train_acc=self.acc(out[:,1:].argmax(dim=1),y_true['labels'])
        train_auc=self.AUROC(out[:,1:],y_true['labels'])
        self.log("train_accuracy", train_acc, prog_bar=False)
        self.log("train_auc", train_auc, prog_bar=False)
    
###############################################################################    
class MHAAbsolutePharmaRegressionhead(LightningTrainer):
    def __init__(self,model,hidden_dim=13,pred_hidden_dropout=0.1, lr=1e-4,noise_lvl=0.3):
        super(MHAAbsolutePharmaRegressionhead, self).__init__(lr=lr)
        self.model=model
        self.linear=torch.nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(pred_hidden_dropout)
        self.lnorm=torch.nn.LayerNorm(hidden_dim)
        self.noise_lvl=noise_lvl
        self.mse_loss = torch.nn.MSELoss()
        metrics = MetricCollection({'R2': regression.R2Score(),
                 'MAE': regression.MeanAbsoluteError(),
                 'Pearson': regression.PearsonCorrCoef(),
                 'MAPE': regression.MeanAbsolutePercentageError(),
                 'MSE': regression.MeanSquaredError(),
                 'WMAPE': regression.WeightedMeanAbsolutePercentageError()}
            )
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()
        
    def forward(self, batch):
        out=self.model(batch)
        return self.linear(self.dropout(self.activation(self.lnorm(out)))).squeeze(dim=1)
    
    def generate_features(self, batch):
        return self.model(batch)
    
    def loss_fn(self,out,y_true):
        #no loss for predictions within 0.3
        y_true_val=y_true['values']
        prediction_mask=torch.abs(y_true_val-out)<self.noise_lvl
        y_true_val[prediction_mask]=out[prediction_mask]
        
        mse= self.mse_loss(out,y_true_val)
        return mse
        
    def retrieve_y_true(self,batch):
        y_true=batch['ligands'].ligand_affinity
        return {'values':y_true}
    
    def retrieve_y_pred(self,out):
        return out
        
    def additional_training_logs(self,out,y_true):
        return 0
        
###############################################################################    
class MHAAbsolutehead(LightningTrainer):
    def __init__(self,model,emb_dim,nhead=4,n_layers=3,lr=1e-4,pred_hidden_dropout=0.1,noise_lvl=0.3):
        super(MHAAbsolutehead, self).__init__(lr)
        self.model=model
        self.emb_dim=emb_dim
        self.start=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.end=torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.ones(1,emb_dim)))
        self.linear1=torch.nn.Linear(emb_dim,1)
        encoder_layer= torch.nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_encoder=torch.nn.TransformerEncoder(encoder_layer,num_layers=n_layers)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(pred_hidden_dropout)
        self.mse_loss = torch.nn.MSELoss()
        self.noise_lvl=noise_lvl
        metrics = MetricCollection({'R2': regression.R2Score(),
                 'MAE': regression.MeanAbsoluteError(),
                 'Pearson': regression.PearsonCorrCoef(),
                 'MAPE': regression.MeanAbsolutePercentageError(),
                 'MSE': regression.MeanSquaredError(),
                 'WMAPE': regression.WeightedMeanAbsolutePercentageError()}
            )
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()
        
    def forward(self, batch):
        out,mask=self.model(batch)
        sequences=[ torch.vstack([self.start,out[b,:,:][mask[b]],self.end]) for b in range(out.shape[0])]
        hidden=torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
        return self.linear1(self.dropout(self.activation(self.transformer_encoder(hidden)[:,0,:]))).squeeze(dim=1)
    
    def generate_features(self, batch):
        out,mask=self.model(batch)
        sequences=[ torch.vstack([self.start,out[b,:,:][mask[b]],self.end]) for b in range(out.shape[0])]
        hidden=torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
        return self.transformer_encoder(hidden)[:,0,:]
    
    def loss_fn(self,out,y_true):
        #no loss for predictions within 0.3
        y_true_val=y_true['values']
        prediction_mask=torch.abs(y_true_val-out)<self.noise_lvl
        try:
            y_true_val[prediction_mask]=out[prediction_mask]
        except:
            print(y_true_val)
            print(prediction_mask)
            print(out)
        
        mse= self.mse_loss(out,y_true_val)
        return mse
        
    def retrieve_y_true(self,batch):
        y_true=batch['ligands'].ligand_affinity
        return {'values':y_true}
    
    def retrieve_y_pred(self,out):
        return out
        
    def additional_training_logs(self,out,y_true):
        return 0
    
################################################################################    
class LinearAbsolutehead(LightningTrainer):
    def __init__(self,model,emb_dim,lr=1e-4,pred_hidden_dropout=0.1,noise_lvl=0.3):
        super(LinearAbsolutehead, self).__init__(lr)
        self.model=model
        self.emb_dim=emb_dim
        self.linear1=torch.nn.Linear(emb_dim,1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(pred_hidden_dropout)
        self.mse_loss = torch.nn.MSELoss()
        self.noise_lvl=noise_lvl
        metrics = MetricCollection({'R2': regression.R2Score(),
                 'MAE': regression.MeanAbsoluteError(),
                 'Pearson': regression.PearsonCorrCoef(),
                 'MAPE': regression.MeanAbsolutePercentageError(),
                 'MSE': regression.MeanSquaredError(),
                 'WMAPE': regression.WeightedMeanAbsolutePercentageError()}
            )
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()
        
    def forward(self, batch):
        out,mask=self.model(batch)
        return self.linear1(self.dropout(self.activation(out[:,0,:]))).squeeze(dim=1)
    
    def generate_features(self, batch):
        out,mask=self.model(batch)
        return out[:,0,:]
    
    def loss_fn(self,out,y_true):
        #no loss for predictions within 0.3
        y_true_val=y_true['values']
        prediction_mask=torch.abs(y_true_val-out)<self.noise_lvl
        y_true_val[prediction_mask]=out[prediction_mask]
        
        mse= self.mse_loss(out,y_true_val)
        return mse
        
    def retrieve_y_true(self,batch):
        y_true=batch['ligands'].ligand_affinity
        return {'values':y_true}
    
    def retrieve_y_pred(self,out):
        return out
        
    def additional_training_logs(self,out,y_true):
        return 0
    
    
        
    
  
    
    

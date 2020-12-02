from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# custom
from models.nets.basic_net import BasicNet
from datasets.datasets import MyCustomDataset
from utils.train_utils import Configs
from datasets.utils import inv_norm

class Eval():
    
    def __init__(self, test_dataset, load_path, has_ground_truth=False, batch_size=128, num_workers=0):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model(load_path).to(self.device)
        self.pretrained = test_dataset.pretrained
        self.batch_size = min(batch_size, len(test_dataset))
        self.num_workers = num_workers
        
        self.loader, self.predicted_classes, self.predicted_probas, self.ground_truth = None, None, None, None
        self.inference(test_dataset, has_ground_truth) 
        #automacally done for the init with test_loader, but can also be re-called with other loaders as long as net is the same
        
    def _init_model(self, load_path):
        
        print('init model from ', load_path)
        
        # get configs to init model
        # to be custom too
        configs = Configs(config_file = os.path.join(load_path, 'config.yaml'), load_only=True).get_configs()
        config = {'name' : configs['net_from_name'], 'out_cls': configs['num_classes'] , 'bn_momentum': 1.0 - configs['ema_m']}
        
        # get weights
        checkpoint = torch.load(os.path.join(load_path, 'model_best_f1_score.pth'), map_location=self.device)
        
        # init model and weights
        net = BasicNet(**config).model
        net.load_state_dict(checkpoint['eval_model'], strict=True)

        return net
    
    
    def inference(self, test_dataset, has_ground_truth=False):
        # set model to eval mode
        self.model.eval()
        
        # init other attributes (done each time inference is called)
        self.loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.predicted_classes, self.predicted_probas = [], []
        self.ground_truth = [] if has_ground_truth else None
        
        with torch.no_grad():
            for _, each_ in enumerate(tqdm(self.loader)): 
                images = each_[0] #images should be the first element of tuple returned by loader
                images = images.to(self.device)
                logits = self.model(images)

                p_classes = torch.max(logits, dim=-1)[1] #get the predicted class

                p_probas = torch.max(torch.softmax(logits, dim=-1), dim=-1)[0] #get the predicted probas

                self.predicted_classes.extend(p_classes.detach().cpu().numpy())
                self.predicted_probas.extend(p_probas.detach().cpu().numpy())
                
                try:
                    self.ground_truth.extend(each_[1].numpy())# targets should be the second element if available
                except:
                    continue
                
    
    def _can_compute_metrics(self):
        if self.ground_truth is None:
            raise Exception('has_ground_truth was set to False')
        elif len(self.ground_truth) != len(self.predicted_classes):
            raise Exception('mismatch sizes %d between targets list and predictions %d ' % (len(self.ground_truth), len(self.predicted_classes)))
        else:
            return True
            
    def get_accuracy(self):
        if self._can_compute_metrics():
            return accuracy_score(self.ground_truth, self.predicted_classes)
    
    def get_precision(self):
        if self._can_compute_metrics():
            return precision_score(self.ground_truth, self.predicted_classes)
            
    def get_recall(self):
        if self._can_compute_metrics():
            return recall_score(self.ground_truth, self.predicted_classes)
        
    def get_f1_score(self):
        if self._can_compute_metrics():
            return f1_score(self.ground_truth, self.predicted_classes)
    
    def get_confusion_matrix(self):
        if self._can_compute_metrics():
            return confusion_matrix(self.ground_truth, self.predicted_classes)
    
    def show_image_and_answers(self, index):

        image = self.loader.dataset[index][0]
        if self.pretrained : image = inv_norm(image)
        display(transforms.ToPILImage()(image).convert("LA"))
        print('predicted label is :', self.predicted_classes[index])
        print('with a proba of :', self.predicted_probas[index])
        
        if self.ground_truth is not None: print('ground truth is :', self.ground_truth[index])
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    '''
    Data Configurations
    '''
    parser.add_argument('--load_path', type=str, default='./saved_models/fixmatch_weighted_loss_th95_rn34_pretrained/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--has_ground_truth', action='store_true')
    args = parser.parse_args()
    
    # custom dataset
    test_dataset = MyCustomDataset()

    # create Eval instance
    evals = Eval(test_dataset=test_dataset, 
                 load_path=args.load_path, 
                 has_ground_truth=args.has_ground_truth, 
                 batch_size=args.batch_size)
    
    # example of calls, find them all in Eval class
    print(evals.get_accuracy())
    print(evals.get_confusion_matrix())
    
    # you can also recall the inference function with another dataset (for the same network), in this case re-call:
    # evals.inference(test_dataset=new_test_dataset)
    # then you can recall then all the metrics as aboven 
    # print(evals.get_f1score())
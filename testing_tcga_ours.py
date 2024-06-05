import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
import json

class BagLabels():
    def __init__(self,file_path):
        self.file_path = file_path
        self.labels_dict = self._fetch_labels()

    def _fetch_labels(self):
        data = pd.read_csv(self.file_path, header=0, names=['file_path', 'label'])
        return {row['file_path']: row['label'] for index, row in data.iterrows()}

    def get_label(self, id):
        filename = id.split('/')[-1]
        return self.labels_dict[filename]


class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]
        img_pos = np.asarray([int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])]) # row, col
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def save_metrics(args, predictions, ground_truth, scores):
    # Convert predictions and ground_truth to single integer labels for confusion matrix
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.array(ground_truth)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    sensitivity = recall_score(true_labels, pred_labels, labels=[0, 1, 2], average=None)
    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))

    f1 = f1_score(true_labels, pred_labels, labels=[0, 1, 2], average='weighted')
    auroc = roc_auc_score(true_labels, scores, multi_class='ovo')
    results = {
        'F1 Score': f1,
        'AUC': auroc,
        'Sensitivity': sensitivity.tolist(),
        'Specificity': specificity
    }
    filepath = os.path.join(args.score_path, 'metrics.json')
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['LUAD', 'LUSC', 'benign'], yticklabels=['LUAD', 'LUSC', 'benign'])
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.savefig(f'{args.plt_path}/confusion_matrix.png')
    plt.close()

    # Binarize the output for ROC curve
    true_labels_binarized = label_binarize(true_labels, classes=[0, 1, 2])
    scores = np.array(scores)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 3
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(f'{args.plt_path}/roc_curve.png')
    plt.close()

def save_prediction(args, predictions, ground_truth, scores):
    df = pd.DataFrame({
    'Prediction': predictions,
        'Ground Truth': ground_truth,
        'LUAD Score': [score[0] for score in scores],
        'LUSC Score': [score[1] for score in scores],
        'Benign Score': [score[2] for score in scores]
    })
    filepath = os.path.join(args.score_path,'predictions.json')
    df.to_csv(filepath,index=False)


def test(args, bags_list, bag_labels, milnet):
    milnet.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    predictions = []
    ground_truth = []
    scores = []
    for i in range(0, num_bags):
        print(f'Bag{i}')
        feats_list = []
        pos_list = []
        classes_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))
        label = bag_labels.get_label(bags_list[i])
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            bag_prediction, A, _ = milnet.b_classifier(bag_feats, ins_classes)
            bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            if args.average:
                max_prediction, _ = torch.max(ins_classes, 0) 
                bag_prediction = (bag_prediction+torch.sigmoid(max_prediction).cpu().numpy())/2
            print(bag_prediction)
            ground_truth.append(label)
            attentions = np.array([]).astype(np.float32)
            prediction = [0, 0, 0]  
            score = [bag_prediction[0], bag_prediction[1], 1 - bag_prediction[0] - bag_prediction[1]]
            gt_dict = {0:'LUAD',1:'LUSC'}
            if bag_prediction[0] >= args.thres_luad and bag_prediction[1] < args.thres_lusc:
                print(bags_list[i] + ' is detected as: LUAD')
                color = [1, 0, 0]
                attentions = A[:, 0].cpu().numpy().astype(np.float32)
                prediction[0] = 1
            elif bag_prediction[1] >= args.thres_lusc and bag_prediction[0] < args.thres_luad:
                print(bags_list[i] + ' is detected as: LUSC')
                color = [0, 1, 0]
                attentions = A[:, 1].cpu().numpy().astype(np.float32)
                prediction[1] = 1
            elif bag_prediction[0] < args.thres_luad and bag_prediction[1] < args.thres_lusc:
                print(bags_list[i] + ' is detected as: benign')
                prediction[2] = 1
            else:
                print(bags_list[i] + ' is detected as: both LUAD and LUSC')
                prediction[2] = 1
            print('ground truth: '+ gt_dict[label])
            
            predictions.append(prediction)
            scores.append(score)
            
            if attentions.size > 0:
                attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
                color_map = np.zeros((np.amax(pos_arr, 0)[0] + 1, np.amax(pos_arr, 0)[1] + 1, 3))
                for k, pos in enumerate(pos_arr):
                    tile_color = np.asarray(color) * attentions[k]
                    color_map[pos[0], pos[1]] = tile_color
                slide_name = bags_list[i].split(os.sep)[-1]
                color_map = transform.resize(color_map, (color_map.shape[0] * 32, color_map.shape[1] * 32), order=0)
                io.imsave(os.path.join('test', 'output', slide_name + '.png'), img_as_ubyte(color_map), check_contrast=False)
            else:
                print(f'No attentions for bag {bags_list[i]}, skipping color map generation.')
            
    return predictions, ground_truth, scores
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_luad', type=float, default=0.5912618041038513)
    parser.add_argument('--thres_lusc', type=float, default=0.4408007264137268)
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--embedder_weights',type=str,default='test/weights/embedder.pth')
    parser.add_argument('--aggregator_weights', type=str, default='test/weights/aggregator.pth')
    parser.add_argument('--bag_path', type=str, default='test/patches')
    parser.add_argument('--patch_ext', type=str, default='jpeg')
    parser.add_argument('--map_path', type=str, default='test/output')
    parser.add_argument('--export_scores', type=int, default=1)
    parser.add_argument('--score_path', type=str, default='test/score')
    parser.add_argument('--gt_path', type=str, default='test/label.csv')
    parser.add_argument('--plt_path', type=str, default='test/plts')
    args = parser.parse_args()
    
    # if args.embedder_weights == 'ImageNet':
    #     print('Use ImageNet features')
    #     resnet = models.resnet18(pretrained=True, norm_layer=nn.BatchNorm2d)
    # else:
    #     resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    # for param in resnet.parameters():
    #     param.requires_grad = False
    # resnet.fc = nn.Identity()
    # i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    # b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    # milnet = mil.MILNet(i_classifier, b_classifier).cuda()

    # if args.embedder_weights !=  'ImageNet':
    #     state_dict_weights = torch.load(args.embedder_weights)
    #     new_state_dict = OrderedDict()
    #     for i in range(4):
    #         state_dict_weights.popitem()
    #     state_dict_init = i_classifier.state_dict()
    #     for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
    #         name = k_0
    #         new_state_dict[name] = v
    #     i_classifier.load_state_dict(new_state_dict, strict=False)

    # state_dict_weights = torch.load(args.aggregator_weights) 
    # state_dict_weights["i_classifier.fc.weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    # state_dict_weights["i_classifier.fc.bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    # milnet.load_state_dict(state_dict_weights, strict=False)

    resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    aggregator_weights = torch.load(os.path.join('test', 'weights', 'aggregator.pth'))
    milnet.load_state_dict(aggregator_weights, strict=False)
    
    state_dict_weights = torch.load(os.path.join('test', 'weights', 'embedder.pth'))
    new_state_dict = OrderedDict()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = i_classifier.state_dict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    new_state_dict["fc.weight"] = aggregator_weights["i_classifier.fc.0.weight"]
    new_state_dict["fc.bias"] = aggregator_weights["i_classifier.fc.0.bias"]
    i_classifier.load_state_dict(new_state_dict, strict=True)
    milnet.i_classifier = i_classifier

    bags_list = glob.glob(os.path.join(args.bag_path, '*'))
    os.makedirs(args.map_path, exist_ok=True)
    if args.export_scores:
        os.makedirs(args.score_path, exist_ok=True)

    bag_labels = BagLabels(args.gt_path)
    predictions, ground_truth, scores = test(args, bags_list, bag_labels, milnet)
    
    os.makedirs(args.plt_path, exist_ok=True)
    save_prediction(args, predictions, ground_truth, scores)
    save_metrics(args, predictions, ground_truth, scores)


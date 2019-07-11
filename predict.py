'''

Running this file properly with an image of a flower returns a list of probabilities and a list of flower names (name of the flower species)

The user needs to specify the number of predictions he wants (known as topk predictions) and the program will return the top n class predictions (probabilities + names)

Parameters: -i: image file path. This is mandatory
            -c: PyTorch model checkpoint. This is mandatory
            -tk: aka topk: number of top probability classes to print. This is mandatory
            -json: json file for class names. This is mandatory
            -gpu: predict on gpu or cpu: need to type 'gpu' or 'cpu'. This is mandatory
            -m: optional parameter: if your saved model is not densenet161 you must provide the imagenet name of the model so that it can load properly
            -hu: optional parameter: number of hidden layers. If you changed the architecture in train.py then you need to use the same -hu here.

Returns: 2 lists. first list is the probabilities, second list is the class names
            
'''


import torch   #imports PyTorch
from utils import load_model, predict
import argparse


parser = argparse.ArgumentParser(description = 'Predict tool')

parser.add_argument('-i', '--image', type = str, required=True, help = 'Image file path')
parser.add_argument('-c', '--checkpoint', type = str, required=True, help = 'PyTorch model checkpoint')
parser.add_argument('-tk', '--topk', type = int, required=True, help = 'Number of Top probability classes to print')
parser.add_argument('-json', '--json', type = str, required=True, help = 'JSON file for class names')
parser.add_argument('-gpu', '--gpu', type = str, required=True, help = 'Train on GPU? type gpu or cpu')
parser.add_argument('-m', '--model', type=str, required=False, help='Optional: model type from ImageNet, else will default to densenet161')
parser.add_argument('-hu', '--hidden_units', type= int, required=False, help='Optional: hidden units in first layer, must be bigger then 1656')

args = parser.parse_args()

if __name__ == '__main__':
    
    if args.gpu == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            print('No GPU detected, predictions will be generated on CPU')
        else:
                print('Predictions will be generated on GPU')
                
    elif args.gpu =='cpu':
        print('Predictions will be generated on CPU')
        device = torch.device('cpu')
    else :
        print('inputs were not undertsood, predictions will be generated on CPU')
        device = torch.device('cpu')
        
    if args.model == None:
        args.model = 'densenet161'
        
    if args.hidden_units == None:
        args.hidden_units = 2208
        
    model = load_model(args.checkpoint, imagenet = args.model, hidden_units = args.hidden_units)
    
    print(predict(args.image, model, device, args.json, args.topk))
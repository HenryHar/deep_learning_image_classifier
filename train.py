'''

Running this file will train a neural network (classifier), the architecture is defined in utils.py class model_classifier...

We pass the following arguments: 
    
    -m: provide the name of an imagenet nn, the program will download it and use it to create the features 
    So far only works if the nn has a .classifier module, the densenet and inception do but the googlenet has it under something else which bugs out
    -lr: learning_rate: you need to set the learning rate for the optimizer, uses SGD at the moment.
    -hu: number of hidden units in the first layer of the classifier neural network
    -e: epochs: number of training/validation epochs for the model, the program automatically saves the model every 5 epochs
    -gpu: a command used to train the network on the GPU or CPU. need to type 'gpu' or 'cpu', anything else and the program will use the cpu   

    example: python train.py -m densetnet161 -lr 0.015 -hu 2208 -e 25 -gpu gpu
    run the command above in your conda cmd prompt
    
'''

import torch   #imports PyTorch
import argparse
from utils import create_model

parser = argparse.ArgumentParser(description = 'Currently supports densenet and inception models from ImageNet')

parser.add_argument('-m', '--imagenet', type= str, required=True, help='ImageNet model for pytorch, type string')
parser.add_argument('-d', '--directory', type= str, required=True, help='directory containing train, validate and test data')
parser.add_argument('-lr', '--learning_rate', type =float, required=True, help='Learning rate for optimizer')
parser.add_argument('-hu', '--hidden_units', type = int, required=True, help ='Number of hidden units in first layer')
parser.add_argument('-e', '--epochs', type = int, required=True, help ='Number of Training Epochs')


parser.add_argument('-gpu', '--gpu', type = str, required=True, help = 'Train on GPU? type gpu or cpu')

args = parser.parse_args()

    
'''parser.add_argument('-top_preds', '--top_preds', type = int32, help ='Provide N for top n class predictions')

parser.add_argument('-load_json', '--load_json', help = 'Specify JSON file to map class predictions to other categorical names')
'''

if __name__ == '__main__':
    
    model_choices = ['densenet161','densenet201']   # 201 has 1920, densenet161 has 2208 both have 1.000 output
    
    while args.imagenet not in model_choices:
        print('Please choose one of the following models:', model_choices)
        args.imagenet = input()

    if args.gpu == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            print('No GPU detected, will train on CPU')
        else:
            print('Training on GPU')
            
    elif args.gpu =='cpu':
        print('Training on CPU')
        device = torch.device('cpu')
    else :
        print('inputs were not undertsood, training on CPU')
        device = torch.device('cpu')
    
    
    create_model(args.imagenet, args.directory, args.learning_rate, args.hidden_units, args.epochs, device)
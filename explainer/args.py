import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="unrexplainer")
    
    parser.add_argument('--path', type=str, default='./result/',
                    help='Name of the path. Default is home/neutor/result.')

    parser.add_argument('--dataset', type=str, default='Cora',
                    help='Name of the dataset. Default is Cora.')
        
    parser.add_argument('--model', type=str, default='graphsage',
                    help='Name of the node representation learning model. Default is graphsage.')
    
    parser.add_argument('--perturb', type=float, default=0.0,
                    help='Hyperparameter for perturbation. Default is 0.0')    
    
    parser.add_argument('--maxiter', type=int, default=1000,
                       help='Number of epochs for MCTS with restart.')
    
    parser.add_argument('--explainer', type=str, default='mctsrestart',
                    help='Name of mcts-based-explainer. Default is mctsrestart.')
    
    parser.add_argument('--task', type=str, default='link',
                    help='Name of downstream task. Default is link.')
                
    parser.add_argument('--gpu', type=str, default='0',
                    help='Set the gpu to use. Default is 0.')
    
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='Number of the hidden dimension. Default is 128.')
    
    parser.add_argument('--c1', type=float, default=1.0,
                    help='Hyperparameter for the exploration term. Default is 1.0')    
    
    parser.add_argument('--neighbors_cnt', type=int, default=5,
                    help='Number of top-k neighbors in the embedding space. Default is 5')
    
    parser.add_argument('--restart', type=float, default=0.2,
                    help='Number of probability of restart. Default is .')
    
    parser.add_argument('--iter', default=300, type=int,
                      help='Number of epochs for training')
    
    parser.add_argument('--lr', default=0.001, type=float,
                      help='Number of learning rate')
    
    parser.set_defaults(directed=False)

    return parser.parse_args()
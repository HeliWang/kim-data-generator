import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Kim CNN")
    parser.add_argument('--no_cuda', action="store_false", help="do not use cuda.", dest="cuda")
    parser.add_argument('--gpu', type=int, help="selected cuda device id. Use -1 for CPU.", default=0)
    parser.add_argument('--batch_size', type=int, help="the training batch size used in training a model by torch text.", default=1000)
    parser.add_argument('--mode', type=str, help="""
        rand: All words are randomly initialized and then modified during training.
        static: A model with pre-trained vectors from word2vec. All words -- including the unknown ones that 
          are initialized with zero -- are kept static and only the other parameters of the model are learned.
        non-static: Same as above but the pre-trained vectors are fine-tuned for each task.
        multichannel: A model with two sets of word vectors. Each set of vectors is treated as a 'channel' and
          each filter is applied to both channels, but gradients are back-propagated only through one of the channels.
          Hence the model is able to fine-tune one set of vectors while keeping the other static. Both channels are 
          initialized with word2vec.# text-classification-cnn Implementation for Convolutional Neural Networks for 
          Sentence Classification of Kim (2014) with PyTorch.
    """, default='multichannel')
    parser.add_argument('--lr', type=float, help="learning rate in torch.optim methods when training a model", default=1.0)
    parser.add_argument('--seed', type=int, help="random seed", default=3435)
    parser.add_argument('--dataset', help="dataset to use in building vocabulary and training a model", type=str, default='SST-1')
    parser.add_argument('--resume_snapshot',
                        help="reload saved model snapshot, a file-like object used in torch.load()",
                        type=str, default=None)
    parser.add_argument('--dev_every', help="evaluate performance on validation set in model training for every `dev_every` iterations",
                        type=int, default=30)
    parser.add_argument('--log_every', help="print progress message in model training for every `log_every` iterations",
                        type=int, default=10)
    parser.add_argument('--patience', help="early stop model training if validation accuracy is not rising (or even decreasing) for `patience` iterations",
                        type=int, default=50)
    parser.add_argument('--save_path', help="save snapshop of model after model training",
                        type=str, default='saves')
    parser.add_argument('--output_channel', help="in_channels used in torch.nn.ConvTranspose2d to reduce word dimension",
                        type=int, default=100)
    parser.add_argument('--words_dim', help="words dimension in word embedding", type=int, default=300)
    parser.add_argument('--embed_dim', help="words dimension in static word embedding", type=int, default=300)
    parser.add_argument('--dropout', help="dropout - probability of an element to be zeroed", type=float, default=0.5)
    parser.add_argument('--epoch_decay', help="number of epochs after which the Learning rate is decayed exponentially", type=int, default=15)
    parser.add_argument('--vector_cache', help="load vocab vectors from the path", type=str, default="data/word2vec.sst-1.pt")
    parser.add_argument('--trained_model', help="load trained model for prediction", type=str, default="")
    parser.add_argument('--weight_decay', help="weight decay (L2 penalty) in torch optimizer", type=float, default=0)
    args = parser.parse_args()
    return args
import argparse
import numpy as np
import time
import torch
import utils
import os
from global_model import RENet_global
from sklearn.utils import shuffle
import pickle


def train(args):
    # load data
    num_nodes, num_rels = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    train_data, train_times_origin = utils.load_quadruples('./data/' + args.dataset, 'train.txt')

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)


    os.makedirs('models', exist_ok=True)
    os.makedirs('models/' + args.dataset +str(args.runnr), exist_ok=True)

    # if args.model == 0: 
    #     model_state_file = 'models/' + args.dataset + 'attn.pth'
    # elif args.model == 1:
    #     model_state_file = 'models/' + args.dataset + 'mean.pth'
    # elif args.model == 2:
    #     model_state_file = 'models/' + args.dataset + 'gcn.pth'
    # elif args.model == 3:
    #     model_state_file = 'models/' + args.dataset + '/max' + str(args.maxpool) + 'rgcn_global.pth'
    #MODIFIED eval_paper_authors
    if args.model == 0: # model_state_file = 'models/' + args.dataset + '/' +str(args.runnr)+  '/rgcn.pth'  #MODIFIED eval_paper_authors
        model_state_file = 'models/' + args.dataset + str(args.runnr)+ 'attn.pth'
    elif args.model == 1:
        model_state_file = 'models/' + args.dataset + str(args.runnr)+'mean.pth'
    elif args.model == 2:
        model_state_file = 'models/' + args.dataset  + str(args.runnr)+ 'gcn.pth'
    elif args.model == 3:
        model_state_file = 'models/' + args.dataset + '/' +str(args.runnr)+ '/' +'/max' + str(args.maxpool) + 'rgcn_global.pth'
    #end MODIFIED eval_paper_authors
    # with open('./RE-Net/data/' + args.dataset + '/train_graphs.txt', 'rb') as f:
    with open('./data/' + args.dataset + '/train_graphs.txt', 'rb') as f:
        graph_dict = pickle.load(f)
    num_nodes, num_rels = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    train_data, train_times_origin = utils.load_quadruples('./data/' + args.dataset, 'train.txt')

    print("start training...")
    model = RENet_global(num_nodes,
                  args.n_hidden,
                  num_rels,
                  dropout=args.dropout,
                  model=args.model,
                  seq_len=args.seq_len,
                  num_k=args.num_k,
                  maxpool=args.maxpool,
                  use_cuda=use_cuda) #modified eval_paper_authors: added use_cuda

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)


    seed = 999
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    if use_cuda: # modified eval_paper_authors: changed order.
        torch.cuda.set_device(args.gpu) # modified eval_paper_authors to set this
        model.cuda()


    true_prob_s, true_prob_o = utils.get_true_distribution(train_data, num_nodes)

    epoch = 0
    loss_small = 10000
    while True:
        model.train()
        if epoch == args.max_epochs:
            break
        epoch += 1
        loss_epoch = 0
        t0 = time.time()
        # print(graph_dict.keys())
        # print(train_times_origin)

        train_times, true_prob_s, true_prob_o = shuffle(train_times_origin, true_prob_s, true_prob_o)


        for batch_data, true_s, true_o in utils.make_batch(train_times, true_prob_s, true_prob_o, args.batch_size):
            
            batch_data = torch.from_numpy(batch_data)
            true_s = torch.from_numpy(true_s)
            true_o = torch.from_numpy(true_o)
            if use_cuda:
                batch_data = batch_data.cuda()
                true_s = true_s.cuda()
                true_o = true_o.cuda()

            loss = model(batch_data, true_s, true_o, graph_dict)  # computes softmax internally
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch += loss.item()

        t3 = time.time()
        model.global_emb = model.get_global_emb(train_times_origin, graph_dict)

        print("Epoch {:04d} | Loss {:.4f} | time {:.4f}".
              format(epoch, loss_epoch / (len(train_times) / args.batch_size), t3 - t0))

        if loss_epoch < loss_small:
            loss_small = loss_epoch
            # if args.model == 3:
            torch.save({'state_dict': model.state_dict(), 'global_emb': model.global_emb},
                        model_state_file)
                # with open(model_graph_file, 'wb') as fp:
                #     pickle.dump(model.graph_dict, fp)
            # else:
            #     torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
            #                 's_hist': model.s_hist_test, 's_cache': model.s_his_cache,
            #                 'o_hist': model.o_hist_test, 'o_cache': model.o_his_cache,
            #                 'latest_time': model.latest_time},
            #                model_state_file)

    print("training done")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RENet')
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS18', help="dataset to use")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--max-epochs", type=int, default=100, help="maximum epochs")
    parser.add_argument("--model", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k", type=int, default=10, help="cuttoff position")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--maxpool", type=int, default=1)
    parser.add_argument('--runnr', type=int, default=0) #added eval_paper_authors: which run out of x seeds - for logging

    args = parser.parse_args()
    print(args)
    train(args)

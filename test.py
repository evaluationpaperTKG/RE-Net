import argparse
import numpy as np
import torch
import utils
import os
from model import RENet
from global_model import RENet_global
import pickle

##added eval_paper_authors
import inspect
import sys
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, currentdir) 
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
import evaluation_utils #for logging
## end eval_paper_authors


def today(): # added eval_paper_authors for logging
    import datetime
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    now = datetime.datetime.now()
    day, month, year = str(now.day), str(now.month), str(now.year)[2:]
    filename = 'results_test_' + day + '-' + month + '-' + year
    return filename


def test(args):
    ###### eval_paper_authors
    # root_dir = os.getcwd()
    # os.chdir(root_dir+'/RE-Net')
    import logging
    root_dir = os.getcwd()
    filename = today()
    log_dir = os.path.join(root_dir, 'logs', '_testresults' + '.log')
    logging.basicConfig(filename=log_dir, filemode='a',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.DEBUG)
    logging.debug("-------------------start---------------------"+str(filename))
    ####### eval_paper_authors

    # ---------------------------------------------------------------------------Initializations----------------------------
    ####### eval_paper_authors
    settingsinfo = 'n'
    if args.feedgt == 'False':
        settingsinfo = 'multistep' + '-' + args.setting #if you do NOT feed the gt, you do multistep
    else:
        settingsinfo = 'singlestep' + '-' + args.setting #single vs multistep plus filter setting. for scoreswriting
    
    experiment_nr = args.runnr
    ### end eval_paper_authors
    # model_state_file = 'models/' + args.dataset + '/rgcn.pth'
    # model_graph_file = 'models/' + args.dataset + '/rgcn_graph.pth'
    # model_state_global_file2 = 'models/' + args.dataset + '/max' + str(args.maxpool) + 'rgcn_global2.pth'
    model_state_file = 'models/' + args.dataset + '/' +str(args.runnr)+  '/rgcn.pth' # eval_paper_authors - one folder per run!
    model_graph_file = 'models/' + args.dataset + '/' +str(args.runnr)+  '/rgcn_graph.pth' # eval_paper_authors
    model_state_global_file2 = 'models/' + args.dataset + '/' +str(args.runnr) +  '/max' + str(args.maxpool) + 'rgcn_global2.pth' #eval_paper_authors

    num_nodes, num_rels = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    if args.dataset == 'icews_know':
        train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        test_data, test_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'test.txt')
    else:
        train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'valid.txt')
        test_data, test_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt', 'test.txt')
    total_data = torch.from_numpy(total_data)
    test_data = torch.from_numpy(test_data)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available() #modified eval_paper_authors: shifted
    model = RENet(num_nodes,
                  args.n_hidden,
                  num_rels,
                  model=args.model,
                  seq_len=args.seq_len,
                  num_k=args.num_k,
                  use_cuda =use_cuda) #added eval_paper_authors: use_cuda

    global_model = RENet_global(num_nodes,
                                args.n_hidden,
                                num_rels,
                                model=args.model,
                                seq_len=args.seq_len,
                                num_k=args.num_k, maxpool=args.maxpool,
                                use_cuda =use_cuda)  #added eval_paper_authors: use_cuda

    with open('data/' + args.dataset + '/test_history_sub.txt', 'rb') as f:
        s_history_test_data = pickle.load(f)
    with open('data/' + args.dataset + '/test_history_ob.txt', 'rb') as f:
        o_history_test_data = pickle.load(f)
    with open(model_graph_file, 'rb') as f:
        model.graph_dict = pickle.load(f)

    
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        # torch.cuda.manual_seed_all(999)
        model.cuda()
        global_model.cuda()
        total_data = total_data.cuda()

    checkpoint_global = torch.load(model_state_global_file2, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    print("Using best epoch: {}".format(checkpoint['epoch']))
    logging.debug("Using best epoch: {}".format(checkpoint['epoch']))
    global_model.load_state_dict(checkpoint_global['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    model.s_hist_test = checkpoint['s_hist']
    model.s_his_cache = checkpoint['s_cache']
    model.o_hist_test = checkpoint['o_hist']
    model.o_his_cache = checkpoint['o_cache']
    model.s_hist_test_t = checkpoint['s_hist_t']
    model.s_his_cache_t = checkpoint['s_cache_t']
    model.o_hist_test_t = checkpoint['o_hist_t']
    model.o_his_cache_t = checkpoint['o_cache_t']
    model.global_emb = checkpoint['global_emb']
    model.latest_time = checkpoint['latest_time']
    if args.dataset == "icews_know":
        model.latest_time = torch.LongTensor([4344])[0]
    model.eval()
    global_model.eval()

    total_loss = 0
    latest_time = test_times[0]
    ranks = []
    hits = []
    total_ranks = np.array([])
    total_ranks_filter = np.array([])
    s_history_test = s_history_test_data[0]
    s_history_test_t = s_history_test_data[1]
    o_history_test = o_history_test_data[0]
    o_history_test_t = o_history_test_data[1]


    for ee in range(num_nodes):
        while len(model.s_hist_test[ee]) > args.seq_len:
            model.s_hist_test[ee].pop(0)
            model.s_hist_test_t[ee].pop(0)
        while len(model.o_hist_test[ee]) > args.seq_len:
            model.o_hist_test[ee].pop(0)
            model.o_hist_test_t[ee].pop(0)

    for i in range(len(test_data)):

        batch_data = test_data[i]
        s_hist = s_history_test[i]
        o_hist = o_history_test[i]
        s_hist_t = s_history_test_t[i]
        o_hist_t = o_history_test_t[i]
        if latest_time != batch_data[3]:
            ranks.append(total_ranks_filter)
            latest_time = batch_data[3]
            total_ranks_filter = np.array([])

        if use_cuda:
            batch_data = batch_data.cuda()

        with torch.no_grad():
            # Filtered metric
            if args.setting == 'raw': #modified eval_paper_authors
                ranks_filter, loss = model.evaluate(batch_data, (s_hist, s_hist_t), (o_hist, o_hist_t),
                                                    global_model, datasetname= args.dataset, settingsinfo=settingsinfo, experiment_nr=experiment_nr)
            elif args.setting == 'static':
                ranks_filter, loss = model.evaluate_filter(batch_data, (s_hist, s_hist_t), (o_hist, o_hist_t),
                                                           global_model, total_data, datasetname= args.dataset, settingsinfo=settingsinfo, experiment_nr=experiment_nr)


            total_ranks_filter = np.concatenate((total_ranks_filter, ranks_filter))
            total_loss += loss.item()
    ranks.append(total_ranks_filter)


    ### logging eval_paper_authors
    method = 'renet'
    logname = method + '-' + args.dataset + '-' +str(args.runnr) + '-' + settingsinfo 
    import pathlib
    dirname = os.path.join(pathlib.Path().resolve(), 'results' )
    newfilename = os.path.join(dirname, logname + ".pkl")
    # if not os.path.isfile(filename):
    with open(newfilename,'wb') as file:
        pickle.dump(model.new_logging_dict, file, protocol=4) 
        file.close()
    ### end eval_paper_authors

    for rank in ranks:
        total_ranks = np.concatenate((total_ranks, rank))

    for hit in [1, 3, 10]:
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print('filter setting: ', args.setting)
        print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
        logging.debug('filter setting: '+ args.setting)
        logging.debug("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
        
    mrr = np.mean(1.0 / total_ranks)
    mr = np.mean(total_ranks)
    
    print("MRR (filtered): {:.6f}".format(mrr))
    print("MR (filtered): {:.6f}".format(mr))

    # added eval_paper_authors
    from datetime import datetime
    now = datetime.now()    
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    # end added eval_paper_authors
    logging.debug("MRR (filtered): {:.6f}".format(mrr)) 
    logging.debug("MR (filtered): {:.6f}".format(mr))
    logging.debug("-------------------------DONE---------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RENet')
    # parser.add_argument("-d", "--dataset", type=str, default='WIKI', help="dataset to use")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14', help="dataset to use")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--model", type=int, default=3)
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k", type=int, default=1000, help="cuttoff position")
    parser.add_argument("--maxpool", type=int, default=1)
    # parser.add_argument('--raw', action='store_true')
    # all below have been added by eval_paper_authors:
    parser.add_argument('--setting', type=str, default='raw', choices=['raw', 'static', 'time'],
                        help="filtering setting")
    parser.add_argument('--feedgt', type=str, default='False') #added eval_paper_authors
    parser.add_argument('--runnr', type=int, default=0) #added eval_paper_authors: which run out of x seeds - for logging

    args = parser.parse_args()
    test(args)


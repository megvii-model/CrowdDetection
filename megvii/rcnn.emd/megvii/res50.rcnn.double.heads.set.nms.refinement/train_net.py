import argparse
import torch
from config import *
from network import Network
import time
from data.CrowdHuman import CrowdHuman
from utils import misc_utils, SGD_bias
import torch.multiprocessing as mp
import torch.distributed as dist
import pdb
def find_free_port():

    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def do_train_epoch(net, data_iter, optimizer, rank, epoch, config):
    
    if rank == 0:
        fid_log = open(config.log_path,'a')

    basic_lr = config.learning_rate
    lr_decay = [0] + config.lr_decay
    ind = [i for i, s in enumerate(lr_decay) if epoch >= s][-1]
    learning_rate = basic_lr * (0.1 ** ind)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    tic = time.time()
    workspace = osp.split(osp.realpath(__file__))[0]
    steps = range(0, config.iter_per_epoch)
    for t, step in zip(data_iter, steps):
        
        images, gt_boxes, im_info = t
        if images is None:
            continue

        # warm up
        if epoch == 1 and step < config.warm_iter:
            alpha = step / config.warm_iter
            lr_new = (0.33 + 0.67 * alpha) * learning_rate
            for group in optimizer.param_groups:
                group['lr'] = lr_new
        else:
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

        # get training data
        optimizer.zero_grad()
        # forward
        outputs = net(images.cuda(rank), im_info.cuda(rank), gt_boxes.cuda(rank))
        # collect the loss
        total_loss = sum([outputs[key].mean() for key in outputs.keys()])

        outputs = {k:v.data.cpu().numpy() for k, v in outputs.items()}
        print_str = ''
        for k, v in outputs.items():
            print_str += '{}: {:.3f}, '.format(k, v)
        print_str += 'total_loss: {:.3f}.'.format(total_loss.data.cpu().numpy())

        total_loss.backward()
       
        optimizer.step()
        
        # stastic
        if rank == 0:
            if step % config.log_dump_interval == 0:
                stastic_total_loss = total_loss.item()
                elt = images.shape[0] * config.log_dump_interval / (time.time() - tic)
                line = 'Epoch:{}, iter:{}/{}, speed:{:.2f} mb/s, lr:{:.4f}, {}\n{}'.format(
                    epoch, 
                    step, 
                    config.iter_per_epoch,
                    elt,
                    optimizer.param_groups[0]['lr'],
                    print_str,
                    workspace,
                    )
                print(line)
                fid_log.write(line + '\n')
                fid_log.flush()
                tic = time.time()
    if rank == 0:
        fid_log.close()

def train_worker(rank, network, config, args):
    # set the parallel
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.num_gpus,
                            rank=rank)
    # initialize model
    net = network()
    # load pretrain model
    backbone_dict = torch.load(config.init_weights)
    backbone_dict = backbone_dict['state_dict']
    del backbone_dict['fc.weight']
    del backbone_dict['fc.bias']    
    net.resnet50.load_state_dict(backbone_dict)

    net.cuda(rank)
    begin_epoch = 1

    # build optimizer
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=config.learning_rate,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    if args.resume_weights:
        model_file = args.resume_weights
        assert osp.exists(model_file)
        check_point = torch.load(model_file, map_location='cpu')
        net.load_state_dict(check_point['state_dict'])
        num = osp.basename(model_file).split('.')[0].split('-')[-1]
        begin_epoch = int(num) + 1

    # using distributed data parallel
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], broadcast_buffers=False)
    # build data provider
    crowdhuman = CrowdHuman(config, if_train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(crowdhuman)
    data_iter = torch.utils.data.DataLoader(dataset=crowdhuman,
            batch_size=config.train_batch_per_gpu,
            num_workers=4,
            collate_fn=crowdhuman.collate_fn,
            sampler = train_sampler,
            pin_memory=False)
    for epoch in range(begin_epoch, config.max_epoch + 1):
        train_sampler.set_epoch(epoch)
        do_train_epoch(net, data_iter, optimizer, rank, epoch, config)
        if rank == 0:
            #save the model
            model_file = osp.join(config.model_dir, 'model-{}.pth'.format(epoch))
            model = dict(epoch = epoch,
                state_dict = net.module.state_dict(),
                optimizer = optimizer.state_dict())
            torch.save(model,model_file)

            model_file = osp.join(config.model_dir, 'model-{}.pth'.format(epoch-1))


def main(args, config, network):
    # check gpus
    if not torch.cuda.is_available():
        print('No GPU exists!')
        return
    else:
        num_gpus = torch.cuda.device_count()
    torch.set_default_tensor_type('torch.FloatTensor')
    num_gpus = min(num_gpus, args.num_gpus)
    config.iter_per_epoch //= num_gpus
    config.learning_rate = config.base_lr * config.train_batch_per_gpu * num_gpus

    line = 'network.lr.{:.4f}.train.{}'.format(config.learning_rate, config.max_epoch)
    config.log_path = osp.join(config.model_dir, '..', line + '.logger')

    misc_utils.ensure_dir(config.model_dir)
    if not osp.exists('output'):
        os.symlink(config.output_dir,'output')
    
    # print the training config
    line = 'Num of GPUs:{}, learning rate:{:.5f}, mini batch size:{}, \
            \ntrain_epoch:{}, iter_per_epoch:{}, decay_epoch:{}'.format(
            num_gpus, config.learning_rate, config.train_batch_per_gpu,
            config.max_epoch, config.iter_per_epoch, config.lr_decay)
    print(line)
    print("Init multi-processing training...")
    
    assert num_gpus > 0
    if num_gpus > 1:
        mp.spawn(train_worker, nprocs=num_gpus, args=(network, config, args))
    else:
        train_worker(0, network, config, args)


def run_train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_weights', '-r', default=None, type=str)
    parser.add_argument('--num-gpus', '-d', default = 1, type=int)
    port =find_free_port()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '{}'.format(port)
    os.environ['NCCL_IB_DISABLE'] = '1'

    print('communication link:{}:{}'.format('127.0.0.1', port))
    args = parser.parse_args()
    main(args, config, Network)

if __name__ == '__main__':
    run_train()

